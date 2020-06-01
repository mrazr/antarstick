from queue import Queue
from typing import List, Dict

import cv2 as cv
import numpy as np
from PyQt5 import QtWidgets
from PyQt5.QtCore import (QMarginsF, QModelIndex, QPointF, QRectF, Qt,
                          pyqtSignal, QByteArray, QThreadPool)
from PyQt5.QtCore import pyqtSlot as Slot
from PyQt5.QtGui import QImage, QPixmap, QPainter, QBrush, QColor, QFont, QPen
from PyQt5.QtWidgets import QGraphicsScene, QSpinBox

from camera import Camera
from camera_processing.antarstick_processing import get_sticks_in_folder
from camera_processing.widgets import ui_camera_view
from camera_processing.widgets.button import Button
from camera_processing.widgets.button_menu import ButtonMenu
from camera_processing.widgets.cam_graphics_view import CamGraphicsView
from camera_processing.widgets.custom_pixmap import CustomPixmap
from camera_processing.widgets.overlay_gui import OverlayGui
from camera_processing.widgets.stick_link_manager import StickLinkManager
from camera_processing.widgets.stick_widget import StickMode, StickWidget
from dataset import Dataset
from image_list_model import ImageListModel
from my_thread_worker import MyThreadWorker
from stick import Stick
from camera_processing.widgets.stick_length_input import StickLengthInput


class CameraViewWidget(QtWidgets.QWidget):

    sticks_changed = pyqtSignal()
    initialization_done = pyqtSignal(Camera)

    def __init__(self, dataset: Dataset):
        QtWidgets.QWidget.__init__(self)

        self.ui = ui_camera_view.Ui_CameraView()
        self.ui.setupUi(self)
        self.ui.detectionSensitivitySlider.sliderReleased.connect(self._handle_slider_released)
        self.ui.detectionSensitivitySlider.valueChanged.connect(self._handle_slider_value_changed)

        self.image_list = ImageListModel()
        self.ui.image_list.setModel(self.image_list)
        self.ui.image_list.selectionModel().currentChanged.connect(self.handle_list_model_current_changed)
        self.ui.image_list.setEnabled(False)

        self.ui.splitter.setStretchFactor(0, 1)
        self.ui.splitter.setStretchFactor(1, 12)
        self.ui.splitter.splitterMoved.connect(self.handle_splitter_moved)

        self.dataset = dataset
        self.dataset.cameras_linked.connect(self.handle_cameras_linked)
        self.dataset.cameras_unlinked.connect(self.handle_cameras_unlinked)
        self.dataset.camera_added.connect(self.handle_camera_added)
        self.dataset.camera_removed.connect(self.handle_camera_removed)
        self.camera: Camera = None
        self.graphics_scene = QGraphicsScene()

        self.stick_link_manager = StickLinkManager(self.dataset, self.camera)
        self.graphics_scene.addItem(self.stick_link_manager)
        self.stick_link_manager.setZValue(2)
        self.stick_link_manager.setVisible(False)

        self.cam_view = CamGraphicsView(self.stick_link_manager, self)
        self.ui.cam_view_placeholder.addWidget(self.cam_view)
        self.cam_view.setScene(self.graphics_scene)

        self.current_viewed_image: np.ndarray = None
        self.gpixmap = CustomPixmap(self.dataset)
        self.gpixmap.setAcceptHoverEvents(False)
        self.gpixmap.setZValue(1)

        self.gpixmap.right_add_button.clicked.connect(self.handle_link_camera_button_clicked)
        self.gpixmap.left_add_button.clicked.connect(self.handle_link_camera_button_clicked)

        self.graphics_scene.addItem(self.gpixmap)

        self.stick_widgets: List[StickWidget] = []
        self.detected_sticks: List[Stick] = []
        self.link_menus = dict({"right": None, "left": None})
        self.left_link: CustomPixmap = None
        self.right_link: CustomPixmap = None

        self.overlay_gui = OverlayGui(self.cam_view)
        self.overlay_gui.reset_view_requested.connect(self._recenter_view)
        self.overlay_gui.edit_sticks_clicked.connect(self.handle_edit_sticks_clicked)
        self.overlay_gui.link_sticks_clicked.connect(self.handle_link_sticks_clicked)
        #self.overlay_gui.sticks_length_clicked.connect(self.handle_sticks_length_clicked)

        #self.sticks_length_input = QSpinBox(None)
        ##self.sticks_length_input.setStyleSheet("border: 1px solid red; border-radius: 5px 5px;")
        #self.g_sticks_length_input = self.graphics_scene.addWidget(self.sticks_length_input)
        #self.g_sticks_length_input.setVisible(False)
        self.graphics_scene.addItem(self.overlay_gui)
        #self.stick_length_input = StickLengthInput()
        #self.graphics_scene.addItem(self.stick_length_input)
        #self.stick_length_input.adjust_layout()
        #self.stick_length_input.setVisible(False)
        #self.stick_length_input.setZValue(15)
        #self.stick_length_input.input_entered.connect(self.handle_sticks_length_clicked)
        #self.stick_length_input.input_cancelled.connect(self.handle_sticks_length_clicked)
        self.overlay_gui.initialize()

        self.link_menu = ButtonMenu()
        self.link_menu_position: str = None
        self.graphics_scene.addItem(self.link_menu)
        self.link_menu.setZValue(100)
        self.link_menu.setVisible(False)
        self.link_menu.set_layout_direction("vertical")

        self.return_queue = Queue()

    def initialise_with(self, camera: Camera):
        self.camera = camera
        self.stick_link_manager.camera = self.camera
        self.stick_link_manager.update_links()
        self.image_list.initialize(self.camera.folder)
        if len(self.camera.sticks) == 0:
            print('detecting')
            self._detect_sticks()
        else:
            self.initialize_rest_of_gui()

    def initialize_rest_of_gui(self):
        self.ui.image_list.setModel(self.image_list)
        viewport_rect = self.cam_view.viewport().rect()
        _re = self.cam_view.mapToScene(viewport_rect)
        self.graphics_scene.setSceneRect(QRectF(_re.boundingRect()))

        self.gpixmap.initialise_with(self.camera)

        self.gpixmap.set_show_title(True)

        self.gpixmap.stick_link_requested.connect(self.stick_link_manager.handle_stick_widget_link_requested)

        x_center = self.cam_view.viewport().rect().width() / 2
        self.gpixmap.setPos(x_center - self.gpixmap.boundingRect().width() / 2, 0)
        self.cam_view.fitInView(self.gpixmap.boundingRect(), Qt.KeepAspectRatio)
        self.cam_view.centerOn(self.gpixmap)
        self.graphics_scene.update()

        self.stick_link_manager.primary_camera = self.gpixmap

        #if len(self.gpixmap.stick_widgets) == 0:
        #    self._detect_sticks()

        self.initialize_link_menu()
        self.ui.image_list.setEnabled(True)
        self.initialization_done.emit(self.camera)

    @Slot(bool)
    def link_cameras_enabled(self, value: bool):
        self.gpixmap.set_link_cameras_enabled(value)


    def _detect_sticks(self):
        #img_sticks = get_sticks_in_folder(self.camera.folder)

        worker = MyThreadWorker(get_sticks_in_folder, args=(self.camera.folder,), kwargs={'return_queue': self.return_queue})
        worker.signals.finished.connect(self.handle_first_time_init_done)
        QThreadPool.globalInstance().start(worker)

        #self.camera.rep_image_path = img_sticks[1]
        #self.camera.rep_image = cv.imread(str(self.camera.rep_image_path))
        #self.camera.rep_image = cv.resize(self.camera.rep_image, (0, 0), fx=0.25, fy=0.25)

        #lines = img_sticks[0]
        #sticks: List[Stick] = self.dataset.create_new_sticks(self.camera, len(lines))
        #for i, stick in enumerate(sticks):
        #    line = lines[i]
        #    print(line)
        #    stick.set_endpoints(*(line[0]), *(line[1]))
        #self.camera.add_sticks(sticks)

        #print(sticks)
        
    #def _detect_sticks(self):
    #    non_snow = self.current_viewed_image
    #    if non_snow is None:
    #        non_snow = get_non_snow_images(self.camera.folder)
    #        if non_snow is None:
    #            return
    #        non_snow = non_snow[0]

    #    img = cv.cvtColor(non_snow, cv.COLOR_BGR2GRAY)
    #    img = cv.pyrDown(img)
    #    perc = self.ui.detectionSensitivitySlider.value() / 100.0
    #    height = perc * img.shape[0]
    #    prep, _ = preprocess_phase(img)
    #    lines = get_lines_from_preprocessed(prep, img)
    #    lines = list(filter(lambda l: np.linalg.norm(l[0] - l[1]) >= height, lines))

    #    if len(lines) == 0:
    #        return
    #
    #    self.camera.remove_sticks()

    #    sticks: List[Stick] = self.dataset.create_new_sticks(self.camera, len(lines))
    #    for i, stick in enumerate(sticks):
    #        line = lines[i]
    #        stick.set_endpoints(*(line[0]), *(line[1]))
    #    self.camera.add_sticks(sticks)

    @Slot()
    def _handle_slider_released(self):
        self._detect_sticks()
    
    @Slot(int)
    def _handle_slider_value_changed(self, value: int):
        self.gpixmap.set_reference_line_percentage(value / 100.0)
    
    def _recenter_view(self):
        rect_to_view = self.gpixmap.sceneBoundingRect()

        if self.left_link is not None:
            rect_to_view = rect_to_view.united(self.left_link.sceneBoundingRect())
        if self.right_link is not None:
            rect_to_view = rect_to_view.united(self.right_link.sceneBoundingRect())
        
        self.graphics_scene.setSceneRect(rect_to_view.marginsAdded(QMarginsF(50, 50, 50, 50)))

        self.cam_view.fitInView(rect_to_view, Qt.KeepAspectRatio)
        self.graphics_scene.update(rect_to_view)
    
    @Slot(int, int)
    def handle_splitter_moved(self, pos: int, index: int):
        self._recenter_view()
    
    @Slot(QModelIndex, QModelIndex)
    def handle_list_model_current_changed(self, current: QModelIndex, previous: QModelIndex):
        print('list model current changed')
        image_path = self.image_list.data(current, Qt.UserRole)
        self.current_viewed_image = cv.pyrDown(cv.imread(str(image_path)))
        self.gpixmap.set_image(cv.resize(self.current_viewed_image, (0, 0), fx=0.5, fy=0.5))

    @Slot()
    def handle_edit_sticks_clicked(self):
        edit_sticks_on = self.overlay_gui.edit_sticks_button_pushed()
        if edit_sticks_on:
            self.gpixmap.set_stick_widgets_mode(StickMode.EDIT)
        else:
            self.gpixmap.set_stick_widgets_mode(StickMode.DISPLAY)
        self.graphics_scene.update()
    
    @Slot()
    def handle_sticks_changed(self):
        if self.left_link is not None:
            self.left_link.update_stick_widgets()
        if self.right_link is not None:
            self.right_link.update_stick_widgets()
        self.sync_stick_link_manager()
        self.graphics_scene.update()
    
    @Slot('PyQt_PyObject')
    def handle_stick_link_requested(self, stick_widget: StickWidget):
        for sw in self.stick_widgets:
            sw.set_available_for_linking(False)

        if self.left_link is not None:
            for sw in self.left_link.stick_widgets:
                sw.set_available_for_linking(True)
        
        if self.right_link is not None:
            for sw in self.right_link.stick_widgets:
                sw.set_available_for_linking(True)

        self.stick_link_manager.handle_stick_widget_link_requested(stick_widget)
        self.graphics_scene.update()

    def stick_widgets_set_mode(self, mode: StickMode):
        for sw in self.stick_widgets:
            sw.set_mode(mode)
    
    def handle_link_sticks_clicked(self):
        if self.overlay_gui.link_sticks_button_pushed():
            self.stick_link_manager.start()
        else:
            self.stick_link_manager.stop()

    def sync_stick_link_manager(self):
        cams = []
        if self.left_link is not None:
            cams.append(self.left_link)
        if self.right_link is not None:
            cams.append(self.right_link)

        self.stick_link_manager.set_secondary_cameras(cams)
    
    def _destroy(self):
        self.graphics_scene.removeItem(self.stick_link_manager)

    def handle_cameras_linked(self, cam1: Camera, cam2: Camera):
        if cam1.id != self.camera.id and cam2.id != self.camera.id:
            return
        c_pixmap = CustomPixmap(self.dataset)
        self.graphics_scene.addItem(c_pixmap)
        c_pixmap.setAcceptHoverEvents(False)

        c_pixmap.set_display_mode()
        c_pixmap.set_show_stick_widgets(True)
        cam: Camera = None
        cam_position = "right"

        if self.camera.id == cam1.id: # self.camera is on the left
            cam = cam2
        else: # self.camera is on the right
            cam = cam1
            cam_position = "left"

        c_pixmap.initialise_with(cam)

        pos: QPointF  = self.gpixmap.pos()
        if cam_position == "left":
            pos.setX(pos.x() - self.gpixmap.boundingRect().width())
            if self.left_link is not None:
                self.dataset.unlink_cameras(self.camera, self.left_link.camera)
            self.gpixmap.left_add_button.set_role("UNLINK")
            self.gpixmap.left_add_button.setVisible(True)
            self.left_link = c_pixmap
        else:
            pos.setX(pos.x() + self.gpixmap.boundingRect().width())
            if self.right_link is not None:
                self.dataset.unlink_cameras(self.camera, self.right_link.camera)
            self.gpixmap.right_add_button.set_role("UNLINK")
            self.gpixmap.right_add_button.setVisible(True)
            self.right_link = c_pixmap

        c_pixmap.setPos(pos)

        self._recenter_view()

        self.sync_stick_link_manager()

    def handle_cameras_unlinked(self, cam1: Camera, cam2: Camera):
        to_remove = cam1 if cam2.id == self.camera.id else cam2

        if self.left_link is not None and self.left_link.camera.id == to_remove.id:
            self.left_link.setParentItem(None)
            self.graphics_scene.removeItem(self.left_link)
            self.left_link = None
            self.gpixmap.left_add_button.set_role("LINK")
        elif self.right_link is not None and self.right_link.camera.id == to_remove.id:
            self.right_link.setParentItem(None)
            self.graphics_scene.removeItem(self.right_link)
            self.right_link = None
            self.gpixmap.right_add_button.set_role("LINK")
        self._recenter_view()

    def handle_link_camera_button_clicked(self, btn_position: str, button_role: str):
        if button_role.lower() == "unlink":
            self.dataset.unlink_cameras(self.camera,
                                        self.left_link.camera if btn_position == "left" else self.right_link.camera)
            return

        self.link_menu_position = btn_position
        self.adjust_link_menu_position()
        self.link_menu.setVisible(True)

    def adjust_link_menu_position(self):
        pos = self.gpixmap.left_add_button.sceneBoundingRect().center()
        if self.link_menu_position == "right":
            self.gpixmap.left_add_button.link_cam_text.setVisible(False)
            if self.link_menu_position is not None:
                self.gpixmap.left_add_button.setVisible(True)
            pos = self.gpixmap.right_add_button.sceneBoundingRect().center()
            pos = pos - QPointF(self.link_menu.boundingRect().width(), self.link_menu.boundingRect().height() * 0.5)
        else:
            self.gpixmap.right_add_button.link_cam_text.setVisible(False)
            if self.link_menu_position is not None:
                self.gpixmap.right_add_button.setVisible(True)
            pos = pos - QPointF(0.0 * self.link_menu.sceneBoundingRect().width() * 0.5,
                                self.link_menu.boundingRect().height() * 0.5)
        self.gpixmap.disable_link_button(self.link_menu_position)
        self.link_menu.setPos(pos)
        #self.link_menu.setVisible(True)

    def initialize_link_menu(self):
        for cam in self.dataset.cameras:
            if cam.id == self.camera.id:
                continue
            self.handle_camera_added(cam)

    def handle_camera_added(self, camera: Camera):
        img = camera.rep_image
        pixmap = None
        if img is None:
            pixmap = QPixmap(170, 128)
            painter = QPainter()
            pixmap.fill(QColor(0, 125, 125, 100))
            painter.begin(pixmap)
            painter.setRenderHint(QPainter.TextAntialiasing)
            font = QFont(Button.font)
            font.setPixelSize(14)
            painter.setFont(font)
            brush = QBrush(QColor(255, 255, 255, 100))
            pen = QPen()
            pen.setWidth(1.5)
            pen.setBrush(brush)
            painter.setPen(pen)
            painter.drawText(pixmap.rect(), Qt.AlignCenter, "initializing")
            painter.end()
        else:
            barray = QByteArray(img.tobytes())
            image = QImage(barray, img.shape[1], img.shape[0], QImage.Format_BGR888)
            pixmap = QPixmap.fromImage(image)
        btn = self.link_menu.get_button(btn_id=str(camera.folder))
        if btn is None:
            btn = self.link_menu.add_button(btn_id=str(camera.folder), label=str(camera.folder.name), pixmap=pixmap)
            btn.clicked.connect(self.handle_camera_button_clicked)
        else:
            btn.set_pixmap(pixmap)
            self.link_menu.set_layout_direction(self.link_menu.layout_direction)
        self.adjust_link_menu_position()

    def handle_camera_removed(self, camera: Camera):
        if self.camera.id == camera.id:
            return
        button = self.link_menu.get_button(str(camera.folder))
        self.link_menu.remove_button(str(camera.folder))
        button.setParentItem(None)
        self.graphics_scene.removeItem(button)
        button.deleteLater()

    def handle_camera_button_clicked(self, btn_dict: Dict[str, str]):
        btn_id = btn_dict["btn_id"]
        camera = self.dataset.get_camera(btn_id)

        if self.link_menu_position == "right":
            self.dataset.link_cameras(self.camera, camera)
        else:
            self.dataset.link_cameras(camera, self.camera)

        self.link_menu_position = None
        self.link_menu.setVisible(False)

    def handle_first_time_init_done(self):
        img_sticks = self.return_queue.get()

        self.camera.rep_image_path = img_sticks[1]
        self.camera.rep_image = cv.imread(str(self.camera.rep_image_path))
        self.camera.rep_image = cv.resize(self.camera.rep_image, (0, 0), fx=0.25, fy=0.25)

        lines = img_sticks[0]
        sticks: List[Stick] = self.dataset.create_new_sticks(self.camera, len(lines))
        for i, stick in enumerate(sticks):
            line = lines[i]
            stick.set_endpoints(*(line[0]), *(line[1]))
        self.camera.add_sticks(sticks)

        self.initialize_rest_of_gui()
