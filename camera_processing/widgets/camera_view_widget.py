import os
from typing import List

import cv2 as cv
import numpy as np
from numpy import ndarray
from PyQt5 import QtWidgets
from PyQt5.QtCore import QPointF, QRectF, Qt, QModelIndex
from PyQt5.QtCore import pyqtSlot as Slot
from PyQt5.QtWidgets import QGraphicsScene

from camera import Camera
from camera_processing import antarstick_processing as antar
from camera_processing.antarstick_processing import (detect_sticks_hmt,
                                                     get_non_snow_images,
                                                     preprocess_phase,
                                                     get_lines_from_preprocessed)
from camera_processing.widgets import ui_camera_view
from camera_processing.widgets.custom_pixmap import CustomPixmap
from camera_processing.widgets.link_camera_menu import LinkCameraMenu
from camera_processing.widgets.stick_widget import StickWidget
from camera_processing.widgets.cam_graphics_view import CamGraphicsView
from dataset import Dataset
from stick import Stick
from image_list_model import ImageListModel

class CameraViewWidget(QtWidgets.QWidget):

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
        self.camera: Camera = None
        self.graphics_scene = QGraphicsScene()

        self.cam_view = CamGraphicsView(self)
        self.ui.cam_view_placeholder.addWidget(self.cam_view)
        self.cam_view.setScene(self.graphics_scene)


        self.gpixmap = CustomPixmap()
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
        self.image_list.initialize(self.camera.folder)
        self.ui.image_list.setModel(self.image_list)
        viewport_rect = self.cam_view.viewport().rect()
        self.graphics_scene.setSceneRect(QRectF(viewport_rect))
        self.gpixmap.initialise_with(self.camera)
        x_center = self.cam_view.viewport().rect().width() / 2
        self.gpixmap.setPos(x_center - self.gpixmap.boundingRect().width() / 2, 0)
        self.cam_view.fitInView(self.gpixmap.boundingRect(), Qt.KeepAspectRatio)
        self.cam_view.centerOn(self.gpixmap)
        self.graphics_scene.update()

        if self.camera.stick_count() == 0:
            self._detect_sticks()
        
        self.gpixmap.update_stick_widgets()
            

    def handle_link_camera_button_clicked(self, btn_position: str):
        # TODO This if..elif could be handled better
        if btn_position == "left" and self.left_link is not None:
            self.left_link.setParentItem(None)
            self.graphics_scene.removeItem(self.left_link)
            self.left_link = None
            self.gpixmap.left_add_button.set_role("LINK")
            self._recenter_view()
            return
        elif btn_position == "right" and self.right_link is not None:
            self.right_link.setParentItem(None)
            self.graphics_scene.removeItem(self.right_link)
            self.right_link = None
            self.gpixmap.right_add_button.set_role("LINK")
            self._recenter_view()
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
        non_snow = get_non_snow_images(self.camera.folder)
        if non_snow is None:
            return
        img = cv.cvtColor(non_snow[0], cv.COLOR_BGR2GRAY)
        img = cv.pyrDown(img)
        perc = self.ui.detectionSensitivitySlider.value() / 100.0
        height = perc * img.shape[0]
        prep, _ = preprocess_phase(img)
        lines = get_lines_from_preprocessed(prep)
        lines = list(filter(lambda l: np.linalg.norm(l[0] - l[1]) >= height, lines))
        if len(lines) == 0:
            return
        
        self.camera.sticks.clear()
        sticks: List[Stick] = self.dataset.create_new_sticks(len(lines))
        for i, stick in enumerate(sticks):
            line = lines[i]
            stick.set_endpoints(*(line[0]), *(line[1]))
        self.camera.sticks = sticks
    
    @Slot()
    def _handle_slider_released(self):
        self._detect_sticks()
        self.gpixmap.update_stick_widgets()
    
    @Slot(int)
    def _handle_slider_value_changed(self, value: int):
        self.gpixmap.set_reference_line_percentage(value / 100.0)
    
    def _recenter_view(self):
        rect_to_view = self.gpixmap.sceneBoundingRect()
        
        #for pixmap in self.linked_pixmaps:
        #    rect_to_view = rect_to_view.united(pixmap.sceneBoundingRect())

        if self.left_link is not None:
            rect_to_view = rect_to_view.united(self.left_link.sceneBoundingRect())
        if self.right_link is not None:
            rect_to_view = rect_to_view.united(self.right_link.sceneBoundingRect())
        
        self.graphics_scene.setSceneRect(rect_to_view)
        self.cam_view.fitInView(rect_to_view, Qt.KeepAspectRatio)
        self.graphics_scene.update(rect_to_view)
    
    def handle_link_camera_clicked(self, camera: Camera, menu_position: str):
        link_menu = self.link_menus[menu_position]
        c_pixmap: CustomPixmap = list(filter(lambda pixmap: pixmap.camera.id == camera.id, link_menu.camera_pixmaps))[0]

        link_menu.camera_pixmaps.remove(c_pixmap)


        pos: QPointF  = self.gpixmap.pos()
        if menu_position == "left":
            pos.setX(pos.x() - self.gpixmap.boundingRect().width())
            self.gpixmap.left_add_button.set_role("UNLINK")
            self.left_link = c_pixmap
        else:
            pos.setX(pos.x() + self.gpixmap.boundingRect().width() - 1)
            self.gpixmap.right_add_button.set_role("UNLINK")
            self.right_link = c_pixmap


        c_pixmap.setParentItem(None)
        self.graphics_scene.removeItem(link_menu)
        link_menu.deleteLater()
        del self.link_menus[menu_position]

        self.gpixmap.enable_link_button(menu_position)

        c_pixmap.scale_item(1.0)
        c_pixmap.setAcceptHoverEvents(False)


        c_pixmap.set_display_mode()

        c_pixmap.setPos(pos)

        self._recenter_view()
    
    @Slot(int, int)
    def handle_splitter_moved(self, pos: int, index: int):
        self._recenter_view()
    
    @Slot(QModelIndex, QModelIndex)
    def handle_list_model_current_changed(self, current: QModelIndex, previous: QModelIndex):
        image_path = self.image_list.data(current, Qt.UserRole)
        img = cv.imread(str(image_path))
        img = cv.resize(img, (0, 0), fx=0.25, fy=0.25)
        self.gpixmap.set_image(img)