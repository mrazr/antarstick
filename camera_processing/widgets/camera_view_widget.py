import json
import time
from multiprocessing import Pool
from queue import Queue
from typing import List, Dict, Optional, Any

import cv2 as cv
import numpy as np
from PyQt5 import QtWidgets
from PyQt5.QtCore import (QMarginsF, QModelIndex, QPointF, QRectF, Qt,
                          pyqtSignal, QByteArray, QThreadPool, QRect)
from PyQt5.QtCore import pyqtSlot as Slot, QTimer
from PyQt5.QtGui import QImage, QPixmap, QPainter, QBrush, QColor, QFont, QPen
from PyQt5.QtWidgets import QGraphicsScene, QGraphicsItem
from pandas import DataFrame

import camera_processing.antarstick_processing as antar
from camera import Camera
from camera_processing.antarstick_processing import process_batch
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
from stick_detection_dialog import StickDetectionDialog


class CameraViewWidget(QtWidgets.QWidget):

    sticks_changed = pyqtSignal()
    initialization_done = pyqtSignal(Camera)

    def __init__(self, dataset: Dataset):
        QtWidgets.QWidget.__init__(self)

        self.ui = ui_camera_view.Ui_CameraView()
        self.ui.setupUi(self)
        #self.ui.detectionSensitivitySlider.sliderReleased.connect(self._handle_slider_released)
        #self.ui.detectionSensitivitySlider.valueChanged.connect(self._handle_slider_value_changed)

        self.image_list = ImageListModel()
        self.ui.image_list.setModel(self.image_list)
        self.ui.image_list.selectionModel().currentChanged.connect(self.handle_list_model_current_changed)
        #self.ui.image_list.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        #self.ui.image_list.resizeColumnToContents(1)
        #self.ui.image_list.resizeColumnToContents(2)
        self.ui.image_list.setEnabled(False)

        #self.ui.splitter.setStretchFactor(0, 1)
        #self.ui.splitter.setStretchFactor(1, 4)

        self.ui.splitter.setStretchFactor(0, 0)
        self.ui.splitter.setStretchFactor(1, 1)

        self.ui.splitter.splitterMoved.connect(self.handle_splitter_moved)

        self.dataset = dataset
        self.dataset.cameras_linked.connect(self.handle_cameras_linked)
        self.dataset.cameras_unlinked.connect(self.handle_cameras_unlinked)
        #self.dataset.camera_added.connect(self.handle_camera_added)
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
        self.cam_view.rubberBandChanged.connect(self.handle_cam_view_rubber_band_changed)
        self.cam_view.rubber_band_started.connect(self.handle_cam_view_rubber_band_started)

        self.current_viewed_image: np.ndarray = None
        self.gpixmap = CustomPixmap(self.dataset)
        self.gpixmap.setAcceptHoverEvents(False)
        self.gpixmap.setZValue(1)

        self.gpixmap.right_add_button.clicked.connect(self.handle_link_camera_button_right_clicked)
        self.gpixmap.left_add_button.clicked.connect(self.handle_link_camera_button_left_clicked)

        self.graphics_scene.addItem(self.gpixmap)

        self.stick_widgets: List[StickWidget] = []
        self.detected_sticks: List[Stick] = []
        self.link_menus = dict({"right": None, "left": None})
        self.left_link: Optional[CustomPixmap] = None
        self.right_link: Optional[CustomPixmap] = None

        self.overlay_gui = OverlayGui(self.cam_view)
        self.overlay_gui.reset_view_requested.connect(self._recenter_view)
        self.overlay_gui.edit_sticks_clicked.connect(self.handle_edit_sticks_clicked)
        self.overlay_gui.link_sticks_clicked.connect(self.handle_link_sticks_clicked)
        self.overlay_gui.delete_sticks_clicked.connect(self.handle_delete_sticks_clicked)
        #self.overlay_gui.redetect_sticks_clicked.connect(self.handle_redetect_sticks_clicked)
        self.overlay_gui.process_photos_clicked.connect(self.handle_process_photos_clicked)
        self.overlay_gui.clicked.connect(self.handle_overlay_gui_clicked)
        self.overlay_gui.find_sticks_clicked.connect(self.handle_find_sticks_clicked)
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
        self.link_menu.show_close_button(True)
        self.link_menu_position: str = None
        self.graphics_scene.addItem(self.link_menu)
        self.link_menu.setZValue(100)
        self.link_menu.setVisible(False)
        self.link_menu.set_layout_direction("vertical")
        self.link_menu.close_requested.connect(self.handle_link_menu_close_requested)

        self.return_queue = Queue()
        self.worker_pool = Pool(processes=2)
        self.image_loading_time: float = -1.0

        self.photo_count_to_process: int = 0
        self.photo_count_processed: int = 0
        self.next_batch_start: int = 0
        self.photo_batch: List[str] = []
        self.timer: QTimer = QTimer()

        #self.stick_width_input = QInputDialog(parent=None, flags=Qt.Dialog)
        #self.stick_width_input.setInputMode(QInputDialog.IntInput)
        #self.stick_width_input.setIntMinimum(3)
        #self.stick_width_input.setIntMaximum(15)
        #self.stick_width_input.intValueChanged.connect(self.detect_sticks)
        #self.stick_width_input.intValueSelected.connect(lambda _: cv.destroyAllWindows())

        self.stick_detection_dialog = StickDetectionDialog()
        self.stick_detection_dialog.spinLength.valueChanged.connect(self.detect_sticks)
        self.stick_detection_dialog.spinWidth.valueChanged.connect(self.detect_sticks)
        self.stick_detection_dialog.spinP0.valueChanged.connect(self.detect_sticks)
        self.stick_detection_dialog.buttonBox.clicked.connect(lambda _: cv.destroyAllWindows())
        self.stick_detection_dialog.spinSensitivity.valueChanged.connect(self.detect_sticks)
        #self.overlay_gui.redetect_sticks_clicked.connect(lambda: self.stick_detection_dialog.show())
        self.overlay_gui.redetect_sticks_clicked.connect(self.handle_redetect_sticks_clicked_)
        self.stick_detection_dialog.btnApply.clicked.connect(self.detect_sticks2)
        #self.overlay_gui.redetect_sticks_clicked.connect(lambda: self.stick_detection_dialog.show())

    def handle_redetect_sticks_clicked_(self):
        param_json = json.dumps(antar.params)
        self.stick_detection_dialog.paramsText.setPlainText(param_json)
        self.stick_detection_dialog.show()

    def initialise_with(self, camera: Camera):
        self.camera = camera
        if self.camera.rep_image is None:
            self.camera.rep_image = cv.resize(cv.imread(str(self.camera.folder / self.camera.rep_image_path)), (0, 0), fx=0.25, fy=0.25)
        self.stick_link_manager.camera = self.camera
        self.stick_link_manager.update_links()
        self.image_list.initialize(self.camera, self.camera.get_processed_count())
        self.initialize_rest_of_gui()
        if len(self.camera.sticks) == 0:
            select_index = self.image_list.index(0, 0)
            self.ui.image_list.setCurrentIndex(select_index)
            self.handle_find_sticks_clicked()
        #else:
        #    if self.camera.rep_image is None:
        #        self.camera.rep_image = cv.resize(cv.imread(str(self.camera.rep_image_path)),
        #                                          (0, 0), fx=0.25, fy=0.25)
        #    self.initialize_rest_of_gui()

    def initialize_rest_of_gui(self):
        self.ui.image_list.setModel(self.image_list)
        viewport_rect = self.cam_view.viewport().rect()
        _re = self.cam_view.mapToScene(viewport_rect)
        self.graphics_scene.setSceneRect(QRectF(_re.boundingRect()))

        self.gpixmap.initialise_with(self.camera)

        self.gpixmap.set_show_title(True)

        #self.gpixmap.stick_link_requested.connect(self.stick_link_manager.handle_stick_widget_link_requested)

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
        self.overlay_gui.show_loading_screen(False)
        self.overlay_gui.process_photos_count_clicked.connect(self.handle_process_photos_clicked)
        self.overlay_gui.initialize_process_photos_popup(self.camera.get_photo_count(), self.image_loading_time)
        self.overlay_gui.handle_cam_view_changed()
        #self.overlay_gui.top_menu._center_buttons()
        self.initialization_done.emit(self.camera)

    @Slot(bool)
    def link_cameras_enabled(self, value: bool):
        self.gpixmap.set_link_cameras_enabled(value)


    def _detect_sticks(self):
        #img_sticks = get_sticks_in_folder_non_mp(self.camera.folder)
        #self.return_queue.put_nowait(img_sticks)
        #self.handle_first_time_init_done()
        worker = MyThreadWorker(antar.get_sticks_in_folder2, args=(self.camera.folder,), kwargs={'return_queue': self.return_queue})
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
        #rect_to_view = self.gpixmap.sceneBoundingRect()
        rect_to_view = self.gpixmap.mapToScene(self.gpixmap.boundingRect()).boundingRect()

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
        image_path = self.image_list.data(current, Qt.UserRole)
        self.current_viewed_image = cv.pyrDown(cv.imread(str(image_path)))
        self.gpixmap.set_image(cv.resize(self.current_viewed_image, (0, 0), fx=0.5, fy=0.5))
        #check_endpoints(cv.resize(self.current_viewed_image, (0, 0), fx=0.5, fy=0.5), self.camera.sticks)
        #estimate_sticks_width(cv.resize(self.current_viewed_image, (0, 0), fx=0.5, fy=0.5), self.camera.sticks)
        #antar.look_for_endpoints(cv.resize(self.current_viewed_image, (0, 0), fx=0.5, fy=0.5), self.camera.sticks)
        #antar.look_for_endpoints(cv.GaussianBlur(self.current_viewed_image, (5, 5), 1.5), self.camera.sticks)
        #antar.look_for_endpoints(self.current_viewed_image, self.camera.sticks)
        #if self.stick_detection_dialog.isVisible():
        #    self.detect_sticks2(0)
        #    return
        #    width = self.stick_detection_dialog.spinWidth.value()
        #    length = max(3, self.stick_detection_dialog.spinLength.value())
        #    hog_th = self.stick_detection_dialog.spinSensitivity.value()
        #    if length % 2 == 0:
        #        length = max(3, length - 1)
        #    p0 = self.stick_detection_dialog.spinP0.value()
        #    antar.find_sticks(self.current_viewed_image, hog_th, width, length, p0)


        measurements = self.camera.get_measurement_for(image_path.name)

        # The selected photo is not processed yet, therefore set "missing measurement" value for each StickWidget
        if measurements is None:
            for sw in self.gpixmap.stick_widgets:
                sw.set_snow_height(-1)
            return

        for sw in self.gpixmap.stick_widgets:
            m_ = measurements[sw.stick.label]
            sw.set_snow_height(m_['snow_height'])

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
        self.gpixmap.left_add_button.setVisible(not self.overlay_gui.link_sticks_button_pushed())
        self.gpixmap.right_add_button.setVisible(not self.overlay_gui.link_sticks_button_pushed())
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
        cam: Optional[Camera] = None
        cam_position = "right"

        if self.camera.id == cam1.id:  # self.camera is on the left
            cam = cam2
        else:  # self.camera is on the right
            cam = cam1
            cam_position = "left"

        c_pixmap.initialise_with(cam)

        pos: QPointF = self.gpixmap.pos()
        if cam_position == "left":
            pos.setX(pos.x() - self.gpixmap.boundingRect().width())
            if self.left_link is not None:
                self.dataset.unlink_cameras(self.camera, self.left_link.camera)
            self.gpixmap.left_add_button.set_label('x')
            self.gpixmap.left_add_button.setVisible(True)
            self.gpixmap.left_add_button.set_on(True)
            self.gpixmap.left_add_button.set_tooltip("Unlink")
            self.left_link = c_pixmap
        else:
            pos.setX(pos.x() + self.gpixmap.boundingRect().width())
            if self.right_link is not None:
                self.dataset.unlink_cameras(self.camera, self.right_link.camera)
            self.gpixmap.right_add_button.set_label('x')
            self.gpixmap.right_add_button.setVisible(True)
            self.gpixmap.right_add_button.set_on(True)
            self.gpixmap.right_add_button.set_tooltip("Unlink")
            self.right_link = c_pixmap

        c_pixmap.setPos(pos)
        c_pixmap.stick_link_requested.connect(self.stick_link_manager.handle_stick_widget_link_requested)

        self._recenter_view()

        self.sync_stick_link_manager()
        self.overlay_gui.enable_link_sticks_button(True)

    def handle_cameras_unlinked(self, cam1: Camera, cam2: Camera):
        if cam1.id != self.camera.id and cam2.id != self.camera.id:
            return
        to_remove = cam1 if cam2.id == self.camera.id else cam2

        if self.left_link is not None and self.left_link.camera.id == to_remove.id:
            self.left_link.stick_link_requested.disconnect(self.stick_link_manager.handle_stick_widget_link_requested)
            self.left_link.setParentItem(None)
            self.graphics_scene.removeItem(self.left_link)
            self.left_link = None
            self.gpixmap.left_add_button.set_default_state()
            self.gpixmap.left_add_button.set_label('+')
            self.gpixmap.left_add_button.set_tooltip("Link camera")
        elif self.right_link is not None and self.right_link.camera.id == to_remove.id:
            self.right_link.stick_link_requested.disconnect(self.stick_link_manager.handle_stick_widget_link_requested)
            self.right_link.setParentItem(None)
            self.graphics_scene.removeItem(self.right_link)
            self.right_link = None
            self.gpixmap.right_add_button.set_default_state()
            self.gpixmap.right_add_button.set_label('+')
            self.gpixmap.right_add_button.set_tooltip("Link camera")
        if self.left_link is None and self.right_link is None:
            self.overlay_gui.enable_link_sticks_button(False)
        self._recenter_view()

    def handle_link_camera_button_left_clicked(self, data: Dict[str, Any]):
        self.handle_link_camera_button_clicked('left', data['button'].is_on())

    def handle_link_camera_button_right_clicked(self, data: Dict[str, Any]):
        self.handle_link_camera_button_clicked('right', data['button'].is_on())

    def handle_link_camera_button_clicked(self, button_position: str, is_pushed: bool):
        if not is_pushed:
            self.dataset.unlink_cameras(self.camera,
                                        self.left_link.camera if button_position == "left" else self.right_link.camera)
            return

        if self.link_menu_position == "right":
            self.gpixmap.right_add_button.set_default_state()
        elif self.link_menu_position == "left":
            self.gpixmap.left_add_button.set_default_state()

        self.link_menu_position = button_position
        self.adjust_link_menu_position()
        self.link_menu.setVisible(True)

    def adjust_link_menu_position(self):
        pos = self.gpixmap.left_add_button.sceneBoundingRect().center()
        if self.link_menu_position == "right":
            #self.gpixmap.left_add_button.link_cam_text.setVisible(False)
            if self.link_menu_position is not None:
                self.gpixmap.left_add_button.setVisible(True)
            pos = self.gpixmap.right_add_button.sceneBoundingRect().center()
            pos = pos - QPointF(self.link_menu.boundingRect().width(), self.link_menu.boundingRect().height() * 0.5)
            #self.gpixmap.right_add_button.hide_tooltip()
        elif self.link_menu_position == "left":
            #self.gpixmap.right_add_button.link_cam_text.setVisible(False)
            if self.link_menu_position is not None:
                self.gpixmap.right_add_button.setVisible(True)
            pos = pos - QPointF(0.0 * self.link_menu.sceneBoundingRect().width() * 0.5,
                                self.link_menu.boundingRect().height() * 0.5)
            #self.gpixmap.left_add_button.hide_tooltip()
        self.gpixmap.disable_link_button(self.link_menu_position)
        self.link_menu.setPos(pos)

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
        self.link_menu.reset_button_states()

    def handle_first_time_init_done(self):
        img_sticks_time = self.return_queue.get()
        cv.imshow('draw', img_sticks_time[0])
        cv.imshow('draw2', img_sticks_time[1])
        cv.waitKey(0)
        cv.destroyAllWindows()
        if img_sticks_time is None or True:
            self.camera.rep_image_path = self.camera.folder / self.camera.image_list[0]
            self.camera.rep_image = cv.imread(str(self.camera.rep_image_path))
            self.camera.rep_image = cv.resize(self.camera.rep_image, (0, 0), fx=0.25, fy=0.25)
            self.initialize_rest_of_gui()
            return
        self.camera.rep_image_path = img_sticks_time[1]
        self.camera.rep_image = cv.imread(str(self.camera.rep_image_path))
        self.camera.rep_image = cv.resize(self.camera.rep_image, (0, 0), fx=0.25, fy=0.25)

        self.initialize_rest_of_gui()

        lines = img_sticks_time[0]
        self.image_loading_time = img_sticks_time[2]
        #sticks: List[Stick] = self.dataset.create_new_sticks(self.camera, len(lines))
        sticks: List[Stick] = self.camera.create_new_sticks(lines)
        #for i, stick in enumerate(sticks):
        #    line = lines[i]
        #    stick.set_endpoints(*(line[0]), *(line[1]))
        #self.camera.add_sticks(sticks)
        self.camera.save()


    def handle_cam_view_rubber_band_started(self):
        for sw in self.gpixmap.stick_widgets:
            sw.setFlag(QGraphicsItem.ItemIsSelectable, True)

    def handle_cam_view_rubber_band_changed(self, rect: QRect, from_scene: QPointF, to_scene: QPointF):
        if rect.isNull():
            return
        pixmap_rect = self.gpixmap.mapFromScene(QRectF(from_scene, to_scene)).boundingRect()
        for sw in self.gpixmap.stick_widgets:
            sw.set_selected(False)
        selected = list(filter(lambda sw: pixmap_rect.contains(sw.pos()), self.gpixmap.stick_widgets))
        if len(selected) == 0:
            self.overlay_gui.enable_delete_sticks_button(False)
            return
        self.overlay_gui.enable_delete_sticks_button(True)
        for sw in selected:
            sw.set_selected(True)

    def handle_delete_sticks_clicked(self):
        for sw in list(filter(lambda sw: sw.is_selected(), self.gpixmap.stick_widgets)):
            sw.btn_delete.click_button(artificial_emit=True)
        self.overlay_gui.enable_delete_sticks_button(False)
        self.gpixmap.update()

    def handle_redetect_sticks_clicked(self):
        self.camera.remove_sticks()
        self._detect_sticks()

    #def handle_process_photos_clicked(self):
    #    self.worker_pool.apply_async(process_batch, args=(self.camera.get_batch(count=100), self.camera.folder, self.camera.sticks), callback=self.handle_worker_finished)
    #    #df = process_batch(self.camera.get_batch(count=10), self.camera.folder, self.camera.sticks)
    #    #print(f'rec {df.shape}')

    def handle_worker_finished(self, df: DataFrame):
        self.camera.insert_measurements(df)
        self.image_list.set_processed_count(self.camera.get_processed_count())
        self.photo_count_processed += df.shape[0]

        self.gpixmap.set_progress_bar_progress(self.photo_count_processed, self.photo_count_to_process)
        self.gpixmap.set_status_text(f'processed {self.photo_count_processed} / {self.photo_count_to_process}', 0)

        if self.next_batch_start < self.photo_count_to_process:
            mini_batch = min(50, self.photo_count_to_process - self.next_batch_start)
            self.worker_pool.apply_async(process_batch, args=(
                self.photo_batch[self.next_batch_start:self.next_batch_start + mini_batch], self.camera.folder, self.camera.sticks),
                                         callback=self.handle_worker_finished)
            self.next_batch_start += mini_batch

        if self.photo_count_processed >= self.photo_count_to_process:
            self.gpixmap.set_status_text(f'complete {self.photo_count_processed} / {self.photo_count_to_process}', 2000)
            self.photo_count_to_process = 0
            self.photo_batch = []
            self.next_batch_start = 0
            self.overlay_gui.enable_process_photos_button(True)
            #self.gpixmap.clear_status_progress()

    def handle_process_photos_clicked(self, count: int):
        self.photo_count_to_process = min(count, self.camera.get_photo_count() - self.camera.get_processed_count())
        self.photo_batch = self.camera.get_batch(self.photo_count_to_process)
        self.gpixmap.set_status_text(f'processed {self.photo_count_processed} / {self.photo_count_to_process}', 0)
        if self.photo_count_to_process <= 100:
            mini_batch = min(50, self.photo_count_to_process)
            self.next_batch_start = mini_batch
            self.worker_pool.apply_async(process_batch, args=(
                self.photo_batch[:mini_batch], self.camera.folder, self.camera.sticks),
                                         callback=self.handle_worker_finished)
        else:
            mini_batch = 50
            self.worker_pool.apply_async(process_batch, args=(
                self.photo_batch[:mini_batch], self.camera.folder, self.camera.sticks),
                                         callback=self.handle_worker_finished)

            self.next_batch_start = mini_batch
            self.worker_pool.apply_async(process_batch, args=(
                self.photo_batch[self.next_batch_start:self.next_batch_start + mini_batch], self.camera.folder, self.camera.sticks),
                                         callback=self.handle_worker_finished)
            self.next_batch_start += mini_batch
        #self.worker_pool.apply_async(process_batch, args=(self.camera.get_batch(count=count), self.camera.folder, self.camera.sticks), callback=self.handle_worker_finished)

    def handle_overlay_gui_clicked(self):
        if self.link_menu.isVisible():
            self.link_menu.setVisible(False)

    def handle_link_menu_close_requested(self):
        self.link_menu.reset_button_states()
        self.link_menu.setVisible(False)
        if self.link_menu_position == "left":
            self.gpixmap.left_add_button.set_default_state()
            self.gpixmap.left_add_button.setVisible(True)
        else:
            self.gpixmap.right_add_button.set_default_state()
            self.gpixmap.right_add_button.setVisible(True)

    def detect_sticks(self, _: int):
        if self.current_viewed_image is None:
            return
        antar.params = json.loads(self.stick_detection_dialog.paramsText.toPlainText())

        width = self.stick_detection_dialog.spinWidth.value()
        length = self.stick_detection_dialog.spinLength.value()
        hog_th = self.stick_detection_dialog.spinSensitivity.value()
        if length % 2 == 0:
            length = max(3, length - 1)
        p0 = self.stick_detection_dialog.spinP0.value()
        #antar.find_sticks(self.current_viewed_image, hog_th, width, length, p0)

    def detect_sticks2(self, _: int):
        if self.current_viewed_image is None:
            return
        antar.params = json.loads(self.stick_detection_dialog.paramsText.toPlainText())
        antar.segment_sticks(self.current_viewed_image, True)
        return
        start = time.time()
        gray = cv.pyrDown(cv.cvtColor(self.current_viewed_image, cv.COLOR_BGR2GRAY))
        if antar.params['f'] == 1.0:
            gray = cv.pyrUp(gray)
        #gray = cv.morphologyEx(gray, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_RECT, (1, 15)))
        hmt_index = int(antar.params['hmt_se_size'] < 0)
        hmts = antar.hmt_selems[hmt_index]
        se_choice = abs(antar.params['hmt_se_size'])
        if se_choice == 1:
            se = hmts[0]
        elif se_choice == 3:
            se = hmts[1]
        elif se_choice == 5:
            se = hmts[2]
        else:
            se = hmts[3]
        uhmt, mask = antar.uhmt(gray, se)
        #print(f'shape is {gray.shape}')
        #_, th = cv.threshold(uhmt, 10, 255.0, cv.THRESH_BINARY)
        low = antar.params['hyst_low']
        high = antar.params['hyst_high']
        #th = 255 * apply_hysteresis_threshold(uhmt, low, high).astype(np.uint8)
        #th = antar.asf(th, 9, 1, 'oco')
        th = antar.asf(mask, 9, 1, 'oco')
        if antar.params['f'] == 1.0:
            th = cv.resize(th, (0, 0), fx=0.5, fy=0.5, interpolation=cv.INTER_NEAREST)
            #th = area_opening(th, 10, 2)
            th = cv.resize(th, (0, 0), fx=2.0, fy=2.0, interpolation=cv.INTER_NEAREST)
        else:
            pass
            #th = area_opening(th, 10, 2)
        hough_th = antar.params['hough_th']
        line_len = antar.params['line_length']
        gap = antar.params['line_gap']
        lines = cv.HoughLinesP(th, 1.0, np.pi / 180.0, hough_th, None, line_len, gap)
        print(f'total is {time.time() - start} secs')
        draw = cv.pyrDown(self.current_viewed_image)
        f = 0.5 if antar.params['f'] == 1.0 else 1.0
        if lines is not None:
            for line_ in lines:
                line = line_[0]
                cv.line(draw, (int(f * line[0]), int(f * line[1])), (int(f * line[2]), int(f * line[3])), [0, 255, 0], 1)
        cv.imshow('uhmt', uhmt)
        cv.imshow('uhmt_th', 255 * th)
        cv.imshow('mask', 255 * mask)
        cv.imshow('op', gray)
        cv.imshow('lines', draw)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def update_image_list(self, start: int, end: int):
        idx_from = self.image_list.createIndex(start, 0)
        idx_to = self.image_list.createIndex(end, 2)
        self.image_list.dataChanged.emit(idx_from, idx_to)

    def handle_find_sticks_clicked(self):
        #for sw in self.stick_widgets:
        #    sw.btn_delete.click_button(True)
        self.camera.remove_sticks()
        print(self.current_viewed_image.shape)
        gray = cv.pyrDown(cv.cvtColor(self.current_viewed_image, cv.COLOR_BGR2GRAY))
        lines = antar.detect_sticks(gray)
        valid_lines = list(filter(lambda line: antar.stick_pipeline.predict(antar.extract_features_from_line(gray, line, True)), lines))
        f = 1.0
        valid_lines = list(map(lambda line: (f * line).astype(np.int32), valid_lines))
        sticks: List[Stick] = self.camera.create_new_sticks(valid_lines)
        self.camera.save()
