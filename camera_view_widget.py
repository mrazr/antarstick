# This Python file uses the following encoding: utf-8
import os
from typing import List, Optional

import cv2 as cv
from PySide2 import QtWidgets
from PySide2.QtCore import Qt, QByteArray, QPointF
from PySide2.QtCore import Slot
from PySide2.QtGui import QPixmap, QImage, QFont
from PySide2.QtWidgets import QGraphicsScene, QGraphicsView
from numpy import ndarray

import antarstick_analyzer as antar
import ui_camera_view
from camera import Camera
from custom_pixmap import CustomPixmap
from dataset import Dataset
from stick import Stick
from stick_widget import StickWidget
from link_camera_menu import LinkCameraMenu


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
        self.camera = None
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
        self.ui.cameraView.fitInView(self.gpixmap.boundingRect().toRect(), Qt.KeepAspectRatio)
        self.ui.cameraView.centerOn(self.gpixmap)
        self.graphics_scene.update()

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
