# This Python file uses the following encoding: utf-8
import os
from typing import List

import cv2 as cv
from numpy import ndarray
from PyQt5 import QtWidgets
from PyQt5.QtCore import QMarginsF, QPointF, QRectF, Qt
from PyQt5.QtCore import pyqtSlot as Slot
from PyQt5.QtWidgets import QGraphicsRectItem, QGraphicsScene

from camera import Camera
from camera_processing import antarstick_processing as antar
from camera_processing.antarstick_processing import (detect_sticks_hmt,
                                                     get_non_snow_images)
from camera_processing.widgets import ui_camera_view
from camera_processing.widgets.custom_pixmap import CustomPixmap
from camera_processing.widgets.link_camera_menu import LinkCameraMenu
from camera_processing.widgets.stick_widget import StickWidget
from dataset import Dataset
from stick import Stick
from PyQt5.QtGui import QColor, QPen


class CameraViewWidget(QtWidgets.QWidget):

    def __init__(self, dataset: Dataset):
        QtWidgets.QWidget.__init__(self)

        self.ui = ui_camera_view.Ui_CameraView()
        self.ui.setupUi(self)
        self.ui.detectionSensitivitySlider.sliderReleased.connect(self._handle_slider_released)
        self.ui.detectionSensitivitySlider.valueChanged.connect(self._handle_slider_value_changed)
        self.dataset = dataset
        self.camera: Camera = None
        self.graphics_scene = QGraphicsScene()


        self.ui.cameraView.setScene(self.graphics_scene)
        self.gpixmap = CustomPixmap()
        self.gpixmap.setAcceptHoverEvents(False)
        self.gpixmap.setZValue(1)

        self.gpixmap.right_add_button.clicked.connect(self.handle_link_camera_button_clicked)
        self.gpixmap.left_add_button.clicked.connect(self.handle_link_camera_button_clicked)

        self.graphics_scene.addItem(self.gpixmap)

        self.stick_widgets: List[StickWidget] = []
        self.detected_sticks: List[Stick] = []
        self.link_menus = dict({"right": None, "left": None})
        self.linked_pixmaps: List[CustomPixmap] = []

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
                self.ui.cameraView.fitInView(self.gpixmap.boundingRect().toRect(), Qt.KeepAspectRatio)
                break

    def show_image(self, img: ndarray):
        raise NotImplementedError

    def initialise_with(self, camera: Camera):
        self.camera = camera
        viewport_rect = self.ui.cameraView.viewport().rect()
        self.graphics_scene.setSceneRect(QRectF(viewport_rect))
        self.gpixmap.initialise_with(self.camera)
        x_center = self.ui.cameraView.viewport().rect().width() / 2
        self.gpixmap.setPos(x_center - self.gpixmap.boundingRect().width() / 2, 0)
        self.ui.cameraView.fitInView(self.gpixmap.boundingRect(), Qt.KeepAspectRatio)
        self.ui.cameraView.centerOn(self.gpixmap)
        self.graphics_scene.update()

        if self.camera.stick_count() == 0:
            self._detect_sticks()
        
        self.gpixmap.update_stick_widgets()
            

    def handle_link_camera_button_clicked(self, btn_position: str):
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
        lines = detect_sticks_hmt(img, perc)
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
        
        for pixmap in self.linked_pixmaps:
            rect_to_view = rect_to_view.united(pixmap.sceneBoundingRect())
        

        self.graphics_scene.setSceneRect(rect_to_view)
        self.ui.cameraView.fitInView(rect_to_view, Qt.KeepAspectRatio)
        self.graphics_scene.update(rect_to_view)
    
    def handle_link_camera_clicked(self, camera: Camera, menu_position: str):
        link_menu = self.link_menus[menu_position]
        c_pixmap: CustomPixmap = list(filter(lambda pixmap: pixmap.camera.id == camera.id, link_menu.camera_pixmaps))[0]

        self.linked_pixmaps.append(c_pixmap)
        link_menu.camera_pixmaps.remove(c_pixmap)


        pos: QPointF  = self.gpixmap.pos()
        if menu_position == "left":
            pos.setX(pos.x() - self.gpixmap.boundingRect().width())
            self.gpixmap.left_add_button.set_role("UNLINK")
        else:
            pos.setX(pos.x() + self.gpixmap.boundingRect().width() - 1)
            self.gpixmap.right_add_button.set_role("UNLINK")


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


        pass # TODO link `camera` with `self.camera`
