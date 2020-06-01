# This Python file uses the following encoding: utf-8
from os import scandir
from typing import Dict, List
import multiprocessing


import cv2 as cv
import numpy as np
from PyQt5 import QtWidgets
from PyQt5.QtCore import pyqtSignal as Signal, QRunnable, QThreadPool
from PyQt5.QtCore import pyqtSlot as Slot

import camera_processing.antarstick_processing as antar
from camera import Camera
from camera_processing.widgets.camera_view_widget import CameraViewWidget
from dataset import Dataset



class Worker(QRunnable):

    def __init__(self, cam_widget: CameraViewWidget, camera: Camera) -> None:
        QRunnable.__init__(self)
        self.cam_widget = cam_widget
        self.camera = camera

    def run(self) -> None:
        self.cam_widget.initialise_with(self.camera)


class CameraProcessingWidget(QtWidgets.QTabWidget):
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
        self._dataset = dataset
        self._dataset.camera_added.connect(self.handle_camera_added)
        self._dataset.camera_removed.connect(self.handle_camera_removed)
        self._dataset.loading_finished.connect(self.handle_dataset_loading_finished)
        self._camera_tab_map: Dict[int, int] = dict({})

    camera_link_available = Signal(bool)

    @Slot(Camera)
    def handle_camera_added(self, camera: Camera):
        camera_widget = CameraViewWidget(self._dataset)

        self.camera_link_available.connect(camera_widget.link_cameras_enabled)
        self._camera_tab_map[camera.id] = self.addTab(camera_widget, camera.get_folder_name())
        self.setCurrentIndex(self._camera_tab_map[camera.id])

        camera_widget.initialization_done.connect(self.handle_camera_widget_initialization_done)

        #process = multiprocessing.Process(target=camera_widget.initialise_with, args=(camera,))
        #process.run()
        #camera_widget.initialise_with(camera)
        #print(self.count())
        #kself.update()
        #worker = Worker(camera_widget, camera)
        #QThreadPool.globalInstance().start(worker)
        camera_widget.initialise_with(camera)
        if len(self._dataset.cameras) > 1:
            self.camera_link_available.emit(True)

    @Slot(Camera)
    def handle_camera_removed(self, camera: Camera):
        camera_widget: CameraViewWidget = None
        if self._camera_tab_map[camera.id] is not None:
            camera_widget = self.widget(self._camera_tab_map[camera.id])
            self.removeTab(self._camera_tab_map[camera.id])
            camera_widget.deleteLater()
        if len(self._dataset.cameras) < 2:
            self.camera_link_available.emit(False)

    @Slot(int)
    def handle_tab_close_requested(self, tab_id: int):
        for cam_id, _tab_id in self._camera_tab_map.items():
            if _tab_id == tab_id:
                _cam_widget: CameraViewWidget = self.widget(tab_id)
                self._dataset.remove_camera(cam_id)
                _cam_widget._destroy()
        self._camera_tab_map.clear()

        for i in range(self.count()):
            camera_widget: CameraViewWidget = self.widget(i)
            self._camera_tab_map[camera_widget.camera.id] = i

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

    @Slot()
    def handle_dataset_loading_finished(self):
        for i in range(self.count()):
            cam_widget: CameraViewWidget = self.widget(i)
            cam_widget.initialize_link_menu()

    @Slot(Camera)
    def handle_camera_widget_initialization_done(self, camera: Camera):
        if len(self._dataset.cameras) > 1:
            self.camera_link_available.emit(True)

        for cam_id, widget_id in self._camera_tab_map.items():
            if cam_id == camera.id:
                continue
            cam_widget: CameraViewWidget = self.widget(widget_id)
            cam_widget.handle_camera_added(camera)
