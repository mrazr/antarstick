import os
from typing import List, Dict

import cv2 as cv
import numpy as np
from PyQt5 import QtWidgets
from PyQt5.QtCore import (QMarginsF, QModelIndex, QPointF, QRectF, Qt,
                          pyqtSignal, QByteArray)
from PyQt5.QtCore import pyqtSlot as Slot
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QGraphicsScene
from numpy import ndarray

from camera import Camera
from camera_processing import antarstick_processing as antar
from camera_processing.antarstick_processing import (
    get_lines_from_preprocessed, get_non_snow_images,
    preprocess_phase)
from camera_processing.widgets import ui_camera_view
from camera_processing.widgets.button_menu import ButtonMenu
from camera_processing.widgets.cam_graphics_view import CamGraphicsView
from camera_processing.widgets.custom_pixmap import CustomPixmap
from camera_processing.widgets.link_camera_menu import LinkCameraMenu
from camera_processing.widgets.overlay_gui import OverlayGui
from camera_processing.widgets.stick_link_manager import StickLinkManager
from camera_processing.widgets.stick_widget import StickMode, StickWidget
from dataset import Dataset
from image_list_model import ImageListModel
from stick import Stick


class CameraViewWidget(QtWidgets.QWidget):

    sticks_changed = pyqtSignal()
    link_initiated_between = pyqtSignal([Camera, Camera, str])
    link_broken_between = pyqtSignal([Camera, Camera, str])

    def __init__(self, dataset: Dataset):
        QtWidgets.QWidget.__init__(self)

        self.ui = ui_camera_view.Ui_CameraView()
        self.ui.setupUi(self)
        self.ui.detectionSensitivitySlider.sliderReleased.connect(self._handle_slider_released)
        self.ui.detectionSensitivitySlider.valueChanged.connect(self._handle_slider_value_changed)

        self.image_list = ImageListModel()
        self.ui.image_list.setModel(self.image_list)
        self.ui.image_list.selectionModel().currentChanged.connect(self.handle_list_model_current_changed)

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
        self.gpixmap = CustomPixmap()
        self.gpixmap.setAcceptHoverEvents(False)
        self.gpixmap.setZValue(1)
        self.gpixmap.double_click_handler = self.double_click_handler

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
        self.graphics_scene.addItem(self.overlay_gui)
        self.overlay_gui.initialize()

        self.link_menu = ButtonMenu()
        self.link_menu_position: str = None
        self.graphics_scene.addItem(self.link_menu)
        self.link_menu.setZValue(100)
        self.link_menu.setVisible(False)
        self.link_menu.set_layout_direction("vertical")

    @Slot()
    def handle_find_non_snow_clicked(self):
        for entry in os.scandir(self.camera.folder):
            if entry.is_dir():
                continue
            img = cv.resize(cv.imread(entry.path), (0, 0), fx=0.5, fy=0.5)
            hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
            if antar.is_non_snow(hsv):
                self.pixmap.load(str(entry.path))
                self.gpixmap.set_image(img)
                self.cam_view.fitInView(self.gpixmap.boundingRect().toRect(), Qt.KeepAspectRatio)
                break

    def show_image(self, img: ndarray):
        raise NotImplementedError

    def initialise_with(self, camera: Camera):
        self.camera = camera
        self.stick_link_manager.camera = self.camera
        self.stick_link_manager.update_links()
        self.image_list.initialize(self.camera.folder)
        self.ui.image_list.setModel(self.image_list)
        viewport_rect = self.cam_view.viewport().rect()
        _re = self.cam_view.mapToScene(viewport_rect)
        self.graphics_scene.setSceneRect(QRectF(_re.boundingRect()))

        self.stick_widgets = self.gpixmap.initialise_with(self.camera) # returns StickWidget-s out of Stick-s in Camera if any, otherwise []

        self.gpixmap.set_show_title(True)
        x_center = self.cam_view.viewport().rect().width() / 2
        self.gpixmap.setPos(x_center - self.gpixmap.boundingRect().width() / 2, 0)
        self.cam_view.fitInView(self.gpixmap.boundingRect(), Qt.KeepAspectRatio)
        self.cam_view.centerOn(self.gpixmap)
        self.graphics_scene.update()

        self.stick_link_manager.primary_camera = self.gpixmap

        if len(self.stick_widgets) == 0:
            self._detect_sticks()
        else:
            self.connect_stick_widget_signals()

        self.initialize_link_menu()

    def handle_link_camera_button_clicked_(self, btn_position: str, button_role: str):
        if button_role.lower() == "unlink":
            #self.remove_linked_camera(btn_position, emit=True)
            self.dataset.unlink_cameras(self.camera, self.left_link.camera if btn_position == "left" else self.right_link.camera)
            return

        other_cameras = list(filter(lambda c: c.folder != self.camera.folder, self.dataset.cameras))

        self.link_menus[btn_position] = LinkCameraMenu(btn_position)

        link_menu = self.link_menus[btn_position]

        self.graphics_scene.addItem(link_menu)
        link_menu.setZValue(900)
        link_menu.initialise_with(other_cameras, self.handle_link_camera_clicked)

        pos = self.gpixmap.left_add_button.sceneBoundingRect().center()

        link_menu.position = btn_position
        if btn_position == "right":
            pos = self.gpixmap.right_add_button.sceneBoundingRect().center()
            pos = pos - QPointF(link_menu.boundingRect().width(), link_menu.boundingRect().height() * 0.5)
        else:
            pos = pos - QPointF(0.0 * link_menu.sceneBoundingRect().width() * 0.5,
                                link_menu.sceneBoundingRect().height() * 0.5)
        self.gpixmap.disable_link_button(btn_position)
        link_menu.setPos(pos)
    
    @Slot(bool)
    def link_cameras_enabled(self, value: bool):
        self.gpixmap.set_link_cameras_enabled(value)
        
    def _detect_sticks(self):
        non_snow = self.current_viewed_image
        if non_snow is None:
            non_snow = get_non_snow_images(self.camera.folder)
            if non_snow is None:
                return
            non_snow = non_snow[0]
        

        img = cv.cvtColor(non_snow, cv.COLOR_BGR2GRAY)
        img = cv.pyrDown(img)
        perc = self.ui.detectionSensitivitySlider.value() / 100.0
        height = perc * img.shape[0]
        prep, _ = preprocess_phase(img)
        lines = get_lines_from_preprocessed(prep)
        lines = list(filter(lambda l: np.linalg.norm(l[0] - l[1]) >= height, lines))

        if len(lines) == 0:
            return
        
        self.camera.sticks.clear()

        sticks: List[Stick] = self.dataset.create_new_sticks(self.camera, len(lines))
        self.camera.sticks = sticks
        for i, stick in enumerate(sticks):
            line = lines[i]
            stick.set_endpoints(*(line[0]), *(line[1]))
        self.stick_widgets = self.gpixmap.update_stick_widgets()

        self.connect_stick_widget_signals()
        
        self.sticks_changed.emit()

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
    
    def handle_link_camera_clicked_(self, camera: Camera, menu_position: str):
        link_menu = self.link_menus[menu_position]
        c_pixmap: CustomPixmap = list(filter(lambda pixmap: pixmap.camera.id == camera.id, link_menu.camera_pixmaps))[0]

        link_menu.camera_pixmaps.remove(c_pixmap)

        c_pixmap.setParentItem(None)
        self.graphics_scene.removeItem(c_pixmap)
        self.graphics_scene.removeItem(link_menu)
        link_menu.deleteLater()
        del self.link_menus[menu_position]

        self.gpixmap.enable_link_button(menu_position)

        self.add_linked_camera(camera, menu_position, emit=True)

    def handle_link_camera_clicked(self, camera: Camera, menu_position: str):
        cam1 = self.camera if menu_position == "right" else camera
        cam2 = camera if cam1 == self.camera else self.camera

        self.dataset.link_cameras(cam1, cam2)

    @Slot(int, int)
    def handle_splitter_moved(self, pos: int, index: int):
        self._recenter_view()
    
    @Slot(QModelIndex, QModelIndex)
    def handle_list_model_current_changed(self, current: QModelIndex, previous: QModelIndex):
        image_path = self.image_list.data(current, Qt.UserRole)
        self.current_viewed_image = cv.pyrDown(cv.imread(str(image_path)))
        self.gpixmap.set_image(cv.resize(self.current_viewed_image, (0, 0), fx=0.5, fy=0.5))
        self.gpixmap.update_stick_widgets()
    
    @Slot()
    def handle_edit_sticks_clicked(self):
        edit_sticks_on = self.overlay_gui.edit_sticks_button_pushed()
        self.gpixmap.set_stick_edit_mode(edit_sticks_on)
        for s in self.stick_widgets:
            s.set_edit_mode(edit_sticks_on)
        if edit_sticks_on:
            self.stick_widgets_set_mode(StickMode.EDIT)
        else:
            self.stick_widgets_set_mode(StickMode.DISPLAY)
            self.sticks_changed.emit()
        self.graphics_scene.update()
    
    @Slot(Stick)
    def handle_stick_delete_clicked(self, stick: Stick):
        self.dataset.remove_stick(stick)
        stick_widget = list(filter(lambda sw: sw.stick.id == stick.id, self.stick_widgets))[0]
        self.stick_widgets.remove(stick_widget)
        stick_widget.setParentItem(None)
        self.graphics_scene.removeItem(stick_widget)
        self.graphics_scene.update()
    
    @Slot()
    def handle_sticks_changed(self):
        if self.left_link is not None:
            self.left_link.update_stick_widgets()
        if self.right_link is not None:
            self.right_link.update_stick_widgets()
        self.sync_stick_link_manager()
        self.graphics_scene.update()
    
    def add_linked_camera(self, camera: Camera, position: str, emit: bool = False):
        c_pixmap = CustomPixmap()
        self.graphics_scene.addItem(c_pixmap)
        c_pixmap.initialise_with(camera)
        c_pixmap.setAcceptHoverEvents(False)

        c_pixmap.set_display_mode()
        c_pixmap.set_show_stick_widgets(True)

        pos: QPointF  = self.gpixmap.pos()
        if position == "left":
            pos.setX(pos.x() - self.gpixmap.boundingRect().width())
            if self.left_link is not None:
                self.remove_linked_camera(position, emit=True)
            self.gpixmap.left_add_button.set_role("UNLINK")
            self.left_link = c_pixmap
        else:
            pos.setX(pos.x() + self.gpixmap.boundingRect().width())
            if self.right_link is not None:
                self.remove_linked_camera(position, emit=True)
            self.gpixmap.right_add_button.set_role("UNLINK")
            self.right_link = c_pixmap

        c_pixmap.setPos(pos)

        self._recenter_view()
        if emit:
            self.link_initiated_between.emit(self.camera, camera, position)
        self.sync_stick_link_manager()
    
    def remove_linked_camera(self, position: str, emit: bool = False):
        if position == "left":
            self.left_link.setParentItem(None)
            self.graphics_scene.removeItem(self.left_link)
            if emit:
                self.link_broken_between.emit(self.camera, self.left_link.camera, "left")
            self.left_link = None
            self.gpixmap.left_add_button.set_role("LINK")
        elif position == "right":
            if emit:
                self.link_broken_between.emit(self.camera, self.right_link.camera, "right")
            self.right_link.setParentItem(None)
            self.graphics_scene.removeItem(self.right_link)
            self.right_link = None
            self.gpixmap.right_add_button.set_role("LINK")
        self._recenter_view()
        self.sync_stick_link_manager()

    def double_click_handler(self, x: int, y: int):
        stick = self.dataset.create_new_stick(self.camera)
        stick.set_endpoints(x, y-50, x, y+50)
        self.camera.sticks.append(stick)
        self.stick_widgets = self.gpixmap.update_stick_widgets()
        self.connect_stick_widget_signals()
        self.sticks_changed.emit()
        self.gpixmap.update()
    
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

    def connect_stick_widget_signals(self):
        for sw in self.stick_widgets:
            sw.delete_clicked.connect(self.handle_stick_delete_clicked)
            sw.link_initiated.connect(self.handle_stick_link_requested)

    def handle_link_accepted(self, stick_widget: StickWidget):
        self.stick_link_manager.accept_link_by(stick_widget)
    
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
        c_pixmap = CustomPixmap()
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

        pos = self.gpixmap.left_add_button.sceneBoundingRect().center()
        if btn_position == "right":
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
                                self.link_menu.sceneBoundingRect().height() * 0.5)
        self.link_menu_position = btn_position
        self.gpixmap.disable_link_button(btn_position)
        self.link_menu.setPos(pos)
        self.link_menu.setVisible(True)

    def initialize_link_menu(self):
        for cam in self.dataset.cameras:
            if cam.id == self.camera.id:
                continue
            self.handle_camera_added(cam)

    def handle_camera_added(self, camera: Camera):
        #img = cv.resize(cv.imread(str(camera.rep_image_path)), dsize=(0, 0), fx=0.25, fy=0.25)
        img = camera.rep_image
        barray = QByteArray(img.tobytes())
        image = QImage(barray, img.shape[1], img.shape[0], QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(image)
        btn = self.link_menu.add_button(btn_id=str(camera.folder), label=str(camera.folder.name), pixmap=pixmap)
        btn.clicked.connect(self.handle_camera_button_clicked)

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
