# This Python file uses the following encoding: utf-8
import os
from typing import List, Optional

import cv2 as cv
from numpy import ndarray
from PyQt5 import QtWidgets
from PyQt5.QtCore import QPointF, Qt
from PyQt5.QtCore import pyqtSlot as Slot
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QGraphicsScene

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


class CameraViewWidget(QtWidgets.QWidget):
    __font: Optional[QFont] = None

    def __init__(self, dataset: Dataset):
        QtWidgets.QWidget.__init__(self)
        if not CameraViewWidget.__font:
            CameraViewWidget.__font = QFont()
            CameraViewWidget.__font.setStyleHint(QFont().Monospace)
            CameraViewWidget.__font.setFamily("monospace")
            CameraViewWidget.__font.setPointSizeF(16)

        self.ui = ui_camera_view.Ui_CameraView()
        self.ui.setupUi(self)
        self.ui.detectionSensitivitySlider.sliderReleased.connect(self.update_stick_widgets)
        self.ui.detectionSensitivitySlider.valueChanged.connect(self.update_stick_widgets)
        self.dataset = dataset
        self.camera: Camera = None
        self.graphics_scene = QGraphicsScene()
        self.ui.cameraView.setScene(self.graphics_scene)
        self.gpixmap = CustomPixmap(CameraViewWidget.__font)

        self.gpixmap.right_add_button.clicked.connect(self.handle_link_camera_clicked)
        self.gpixmap.left_add_button.clicked.connect(self.handle_link_camera_clicked)

        self.graphics_scene.addItem(self.gpixmap)

        self.stick_widgets: List[StickWidget] = []
        self.detected_sticks: List[Stick] = []

    @Slot()
    def handle_find_non_snow_clicked(self):
        for entry in os.scandir(self.camera.folder):
            if entry.is_dir():
                continue
            img = cv.resize(cv.imread(entry.path), (0, 0), fx=0.5, fy=0.5)
            hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
            if antar.is_non_snow(hsv):
                self.pixmap.load(str(entry.path))
                #self.gpixmap.setPixmap(self.pixmap)
                self.gpixmap.set_image(img)
                self.ui.cameraView.fitInView(self.gpixmap.boundingRect().toRect(), Qt.KeepAspectRatio)
                break

    @Slot()
    def update_stick_widgets(self):
        percentage = self.ui.detectionSensitivitySlider.value() / 100.0
        self.gpixmap.set_reference_line_percentage(percentage)
        self.gpixmap.update_stick_widgets()
        if len(self.stick_widgets) > 0:
            self.gpixmap.set_show_stick_widgets(True)
        end_idx = int(1.0 * len(self.detected_sticks))
        #for sw in self.stick_widgets:
        #    self.graphics_scene.removeItem(sw)
        #self.stick_widgets.clear()
        #for i in range(end_idx):
        #    stick = self.detected_sticks[i]
        #    stick_widget = StickWidget(stick, self.gpixmap)
        #    self.stick_widgets.append(stick_widget)
        #self.gpixmap.stick_widgets = self.stick_widgets
        #self.camera.sticks.clear()
        #self.camera.sticks.extend(self.detected_sticks[:end_idx])
        #self.graphics_scene.update()

        #self.gpixmap.initialise_with(self.camera)

    def show_image(self, img: ndarray):
        pass
        #self.gpixmap.set_image(img)
        #self.ui.cameraView.fitInView(self.gpixmap.boundingRect().toRect(), Qt.KeepAspectRatio)
        ##print(self.gpixmap.boundingRect())
        #self.ui.cameraView.centerOn(self.gpixmap)
        #self.graphics_scene.update()

    def initialise_with(self, camera: Camera):
        self.camera = camera
        self.gpixmap.initialise_with(self.camera)
        self.gpixmap.setPos(1893 / 2 - self.gpixmap.boundingRect().width() / 2, 0)
        self.ui.cameraView.fitInView(self.gpixmap.boundingRect(), Qt.KeepAspectRatio)
        self.ui.cameraView.centerOn(self.gpixmap)
        self.graphics_scene.update()

        if self.camera.stick_count() == 0:
            non_snow = get_non_snow_images(self.camera.folder)
            if non_snow is None:
                return
            img = cv.cvtColor(non_snow[0], cv.COLOR_BGR2GRAY)
            img = cv.pyrDown(img)
            perc = self.ui.detectionSensitivitySlider.value() / 100.0
            lines = detect_sticks_hmt(img, perc)
            if len(lines) == 0:
                return
            
            sticks: List[Stick] = self.dataset.create_new_sticks(len(lines))
            for i, stick in enumerate(sticks):
                line = lines[i]
                stick.set_endpoints(*(line[0]), *(line[1]))
            self.camera.sticks = sticks
        
        self.gpixmap.update_stick_widgets()
            

    def handle_link_camera_clicked(self, btn_name: str):
        other_cameras = list(filter(lambda c: c.folder != self.camera.folder, self.dataset.cameras))
        link_menu = LinkCameraMenu()
        self.graphics_scene.addItem(link_menu)
        link_menu.initialise_with(other_cameras)
        pos = self.gpixmap.left_add_button.sceneBoundingRect().center()
        if btn_name == "right":
            pos = self.gpixmap.right_add_button.sceneBoundingRect().center()

        pos: QPointF = pos - QPointF(link_menu.sceneBoundingRect().width() * 0.5,
                                     link_menu.sceneBoundingRect().height() * 0.5)

        link_menu.setPos(pos)
    
    @Slot(bool)
    def link_cameras_enabled(self, value: bool):
        self.gpixmap.set_link_cameras_enabled(value)
