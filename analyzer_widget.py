# This Python file uses the following encoding: utf-8
from PySide2 import QtWidgets
from dataset import Dataset
from PySide2.QtCore import Slot, QByteArray, Qt
from CameraViewWidget import CameraViewWidget
from camera import Camera
from typing import Dict, List
from stick_widget import StickWidget
import numpy as np
from os import scandir
import cv2 as cv
import antarstick_analyzer as antar
from PySide2.QtGui import QPixmap, QImage
from stick import Stick
from PySide2.QtCore import Signal


class AnalyzerWidget(QtWidgets.QTabWidget):
    """Class representing GUI widget for the analyzation part of Antarstick.
    Derives from QTabWidget. Each camera gets its own tab page.
    Just initialize this with a Dataset instance and add this widget
    wherever into a QTabWidget or wherever it is supposed to be.

    Attributes
    ----------
    dataset : Dataset
        the application-scoped Dataset instance
    camera_tab_map : Dict[int, int]
        mapping between Camera.id and QTab id
    """
    def __init__(self, dataset: Dataset):
        QtWidgets.QTabWidget.__init__(self)
        self.setTabsClosable(True)
        self.tabCloseRequested.connect(self.handle_tab_close_requested)
        self.dataset = dataset
        self.dataset.camera_added.connect(self.handle_camera_added)
        self.dataset.camera_removed.connect(self.handle_camera_removed)
        self.dataset.camera_sticks_detected.connect(self.handle_camera_sticks_detected)
        self.camera_tab_map: Dict[int, int] = dict({})

    camera_link_available = Signal(bool)

    @Slot(Camera)
    def handle_camera_added(self, camera: Camera):
        camera_widget = CameraViewWidget(camera)
        camera_widget.ui.btnFindNonSnow.clicked.connect(self.handle_detect_sticks_clicked)
        pics = self.find_non_snow_pics_in_camera(camera, 2)
        if len(pics) == 0:
            self.camera_tab_map[camera.id] = self.addTab(camera_widget, camera.get_folder_name())
            return
        img = pics[1]
        #barray = QByteArray(img.tobytes())
        #image = QImage(barray, img.shape[1], img.shape[0], QImage.Format_BGR888)
        #camera_widget.pixmap = QPixmap.fromImage(image)
        #camera_widget.gpixmap.setPixmap(camera_widget.pixmap)
        camera_widget.show_image(img)
        self.camera_link_available.connect(camera_widget.gpixmap.set_link_cameras_enabled)
        if len(self.dataset.cameras) > 1:
            print("COOL count")
            self.camera_link_available.emit(True)
        camera_widget.ui.detectionSensitivitySlider.valueChanged.emit(0)
        self.camera_tab_map[camera.id] = self.addTab(camera_widget, camera.get_folder_name())
        self.setCurrentIndex(self.camera_tab_map[camera.id])
        camera_widget.ui.cameraView.fitInView(camera_widget.gpixmap.boundingRect().toRect(), Qt.KeepAspectRatio)

    @Slot(int)
    def handle_camera_removed(self, camera_id: int):
        if self.camera_tab_map[camera_id]:
            self.removeTab(self.camera_tab_map[camera_id])
        if len(self.dataset.cameras) < 2:
            print("LOW COUNT")
            self.camera_link_available.emit(False)

    @Slot(int)
    def handle_tab_close_requested(self, tab_id: int):
        for cam_id, _tab_id in self.camera_tab_map.items():
            if _tab_id == tab_id:
                self.dataset.remove_camera(cam_id)

    @Slot(Camera)
    def handle_camera_sticks_detected(self, camera: Camera):
        camera_widget_id = self.camera_tab_map[camera.id]
        camera_widget: CameraViewWidget = self.widget(camera_widget_id)
        camera_widget.stick_widgets.clear()
        for stick in camera.sticks:
            stick_widget = StickWidget(stick)
            camera_widget.stick_widgets.append(stick_widget)

    @Slot()
    def handle_detect_sticks_clicked(self):
        camera_widget: CameraViewWidget = self.currentWidget()
        camera = camera_widget.camera
        non_snow_pics = self.find_non_snow_pics_in_camera(camera, 4)
        if len(non_snow_pics) == 0:
            return
        img = cv.cvtColor(non_snow_pics[1], cv.COLOR_BGR2GRAY)
        line_height_perc = camera_widget.ui.detectionSensitivitySlider.value() / 100.0
        lines = antar.detect_sticks_hmt(img, line_height_perc)
        if len(lines) == 0:
            return
        sticks = list(map(lambda line_: Stick(0, np.array(line_[0]), np.array(line_[1])), lines))
        sticks = sorted(sticks, key=lambda stick_: stick_.length_px, reverse=True)
        camera_widget.detected_sticks = sticks
        camera_widget.update_stick_widgets()

    def find_non_snow_pics_in_camera(self, camera: Camera, count: int) -> List[np.ndarray]:
        images = []
        for entry in scandir(camera.folder):
            if entry.is_dir():
                continue
            img = cv.pyrDown(cv.imread(entry.path))
            hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
            if antar.is_non_snow(hsv):
                images.append(cv.pyrDown(img))
                if len(images) == count:
                    break
        return images
