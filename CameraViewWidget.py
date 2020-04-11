# This Python file uses the following encoding: utf-8
from PySide2 import QtCore
from PySide2.QtCore import Qt, QByteArray
from PySide2 import QtWidgets
from PySide2.QtGui import QPixmap, QImage, QFont, QFontDatabase
from PySide2.QtWidgets import QGraphicsScene, QGraphicsSimpleTextItem
import ui_camera_view
from camera import Camera
from dataset import Dataset
from PySide2.QtCore import Slot
import os
import cv2 as cv
import antarstick_analyzer as antar
from CustomPixmap import CustomPixmap
from typing import List, Optional
from stick_widget import StickWidget
from stick import Stick
from numpy import ndarray


class CameraViewWidget(QtWidgets.QWidget):
    __font: Optional[QFont] = None
    def __init__(self, camera: Camera):
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
        self.camera = camera
        self.graphics_scene = QGraphicsScene()
        self.ui.cameraView.setScene(self.graphics_scene)
        self.pixmap = QPixmap(w=0, h=0)
        self.image = QImage()
        self.gpixmap = CustomPixmap(CameraViewWidget.__font)
        self.gpixmap.setPixmap(self.pixmap)
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
                self.gpixmap.setPixmap(self.pixmap)
                self.ui.cameraView.fitInView(self.gpixmap.boundingRect().toRect(), Qt.KeepAspectRatio)
                break

    @Slot()
    def update_stick_widgets(self):
        percentage = self.ui.detectionSensitivitySlider.value() / 100.0
        self.gpixmap.set_reference_line_percentage(percentage)
        end_idx = int(1.0 * len(self.detected_sticks))
        for sw in self.stick_widgets:
            self.graphics_scene.removeItem(sw)
        self.stick_widgets.clear()
        for i in range(end_idx):
            stick = self.detected_sticks[i]
            stick_widget = StickWidget(stick, self.gpixmap)
            self.stick_widgets.append(stick_widget)
        self.gpixmap.stick_widgets = self.stick_widgets
        self.camera.sticks.clear()
        self.camera.sticks.extend(self.detected_sticks[:end_idx])
        self.graphics_scene.update()

    def show_image(self, img: ndarray):
        barray = QByteArray(img.tobytes())
        image = QImage(barray, img.shape[1], img.shape[0], QImage.Format_BGR888)
        self.pixmap = QPixmap.fromImage(image)
        self.gpixmap.setPixmap(self.pixmap)
        re = self.graphics_scene.sceneRect()
        re.setWidth(1893)
        self.graphics_scene.setSceneRect(re)
        self.gpixmap.setPos(1893 / 2 - self.gpixmap.boundingRect().width() / 2, 0)
        self.ui.cameraView.fitInView(self.gpixmap.boundingRect().toRect(), Qt.KeepAspectRatio)
        self.ui.cameraView.centerOn(self.gpixmap)
        self.graphics_scene.update()
