# This Python file uses the following encoding: utf-8
from PySide2 import QtCore
from PySide2 import QtWidgets
from dataset import Dataset
from PySide2.QtCore import Slot
from CameraViewWidget import CameraViewWidget
from camera import Camera
from typing import Dict, List

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
        self.camera_tab_map: Dict[int, int] = dict({})

    @Slot(Camera)
    def handle_camera_added(self, camera: Camera):
        camera_widget = CameraViewWidget(camera)
        self.camera_tab_map[camera.id] = self.addTab(camera_widget, camera.get_folder_name())

    @Slot(int)
    def handle_camera_removed(self, camera_id: int):
        if self.camera_tab_map[camera_id]:
            self.removeTab(self.camera_tab_map[camera_id])

    @Slot(int)
    def handle_tab_close_requested(self, tab_id: int):
        for cam_id, _tab_id in self.camera_tab_map.items():
            if _tab_id == tab_id:
                self.dataset.remove_camera(cam_id)
