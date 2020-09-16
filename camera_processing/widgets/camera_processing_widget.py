# This Python file uses the following encoding: utf-8
from multiprocessing.connection import Connection
from os import scandir, remove
from pathlib import Path
from time import time
from typing import Dict, List, Optional, Tuple
from multiprocessing import Process, Queue, Value, Pipe
import logging
from queue import Empty

import cv2 as cv
import numpy as np
from PyQt5 import QtWidgets
from PyQt5.QtCore import pyqtSignal as Signal, QRunnable, QThreadPool, QTimer
from PyQt5.QtCore import pyqtSlot as Slot
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QPushButton

import camera_processing.antarstick_processing as antar
from camera import Camera
from camera_processing.widgets.camera_view_widget import CameraViewWidget
from dataset import Dataset

logging_start = time()
logging.basicConfig(filename='process.log', level=logging.DEBUG)

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
        self.currentChanged.connect(self.handle_current_tab_changed)
        self.dataset: Dataset = dataset
        self.dataset.camera_added.connect(self.handle_camera_added)
        self.dataset.camera_removed.connect(self.handle_camera_removed)
        self.dataset.loading_finished.connect(self.handle_dataset_loading_finished)
        self._camera_tab_map: Dict[int, int] = dict({})

        #self.process: Optional[Process] = None
        self.active_camera: Queue = Queue(maxsize=10)
        self.active_camera.value = None
        self.new_cameras = Queue()
        self.remove_cameras = Queue()

        self.timer = QTimer(parent=self)
        self.timer.timeout.connect(self.handle_timeout)
        self.logging_start = time()
        self.result_queue = Queue()
        self.to_process = Queue()
        self.conn1, self.conn2 = Pipe()

        self.process = Process(target=analyze_daytime_snow, args=(self.to_process, self.result_queue, self.conn1,))

        self.pause_button = QPushButton(icon=QIcon.fromTheme('media-play'), text='Daytime && snow analysis')
        self.pause_button.setEnabled(False)
        self.pause_button.toggled.connect(self.handle_pause_button_toggled)
        self.pause_button.setCheckable(True)
        self.pause_button.setChecked(False)

        self.active_camera: Optional[Camera] = None

    camera_link_available = Signal(bool)
    camera_added = Signal(Camera)

    @Slot(Camera)
    def handle_camera_added(self, camera: Camera):
        camera_widget = CameraViewWidget(self.dataset)

        self.camera_link_available.connect(camera_widget.link_cameras_enabled)
        self._camera_tab_map[camera.id] = self.addTab(camera_widget, camera.get_folder_name())
        self.setCurrentIndex(self._camera_tab_map[camera.id])

        camera_widget.initialization_done.connect(self.handle_camera_widget_initialization_done)

        camera_widget.initialise_with(camera)

        #self.active_camera.put_nowait(camera)
        self.active_camera = camera
        self.pause_button.setEnabled(True)
        self.pause_button.setChecked(True)
        #if not self.process.is_alive():
        #    logging_start = time()
        #    self.process.start()
        #    self.pause_button.setEnabled(True)
        #    self.pause_button.setIcon(QIcon.fromTheme('media-pause'))
        #    self.fetch_next_batch()
        #    self.timer.start(1000)

    @Slot(Camera)
    def handle_camera_removed(self, camera: Camera):
        #TODO handle camera removed while the same camera is analyzed in self.process
        camera_widget: CameraViewWidget = None
        if self._camera_tab_map[camera.id] is not None:
            camera_widget = self.widget(self._camera_tab_map[camera.id])
            self.removeTab(self._camera_tab_map[camera.id])
            camera_widget.deleteLater()
        if len(self.dataset.cameras) < 2:
            self.camera_link_available.emit(False)

    @Slot(int)
    def handle_tab_close_requested(self, tab_id: int):
        for cam_id, _tab_id in self._camera_tab_map.items():
            if _tab_id == tab_id:
                _cam_widget: CameraViewWidget = self.widget(tab_id)
                _cam_widget.camera.save()
                self.dataset.remove_camera(cam_id)
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
        if len(self.dataset.cameras) > 1:
            self.camera_link_available.emit(True)

        for cam_id, widget_id in self._camera_tab_map.items():
            if cam_id == camera.id:
                continue
            cam_widget: CameraViewWidget = self.widget(widget_id)
            cam_widget.handle_camera_added(camera)

    def cleanup(self):
        if self.process is not None and self.process.is_alive():
            self.process.terminate()
        if self.dataset is None:
            return
        self.dataset.save()
        self.dataset.camera_added.disconnect(self.handle_camera_added)
        self.dataset.camera_removed.disconnect(self.handle_camera_removed)
        self.dataset.loading_finished.disconnect(self.handle_dataset_loading_finished)
        self.dataset = None
        for i in range(self.count()):
            widget = self.widget(i)
            widget.deleteLater()
        #TODO save camera states
        self.clear()
        self._camera_tab_map: Dict[int, int] = dict({})

    def set_dataset(self, dataset: Dataset):
        self.dataset = dataset
        self.dataset.camera_added.connect(self.handle_camera_added)
        self.dataset.camera_removed.connect(self.handle_camera_removed)
        self.dataset.loading_finished.connect(self.handle_dataset_loading_finished)
        self._camera_tab_map: Dict[int, int] = dict({})

    def handle_timeout(self):
        if self.conn2.poll(0.1):
            results: List[Tuple[int, List[Tuple[Path, bool, bool]]]] = self.conn2.recv()
            for result in results:
                cam_id, values = result
                camera = self.dataset.get_camera(cam_id)
                indices = []
                for img_name, daytime, snow in values:
                    indices.append(camera.set_photo_daytime_snow(img_name, daytime, snow))
                min_index = min(indices)
                max_index = max(indices)
                cam_widget: CameraViewWidget = self.widget(self._camera_tab_map[cam_id])
                cam_widget.update_image_list(min_index, max_index)

            if self.pause_button.isChecked():
                self.fetch_next_batch()
            else:
                self.process.terminate()
                self.timer.stop()

    def fetch_next_batch(self):
        photos = map(lambda cam: (cam.id, cam.get_next_photo_daytime_snow(count=6 if cam == self.active_camera else 2)),
                     self.dataset.cameras)
        photos = list(filter(lambda camid_paths: len(camid_paths[1]) > 0, photos))
        if len(photos) > 0:
            self.conn2.send((True, photos))
        else:
            self.conn2.send((False, None))
            self.timer.stop()
            self.pause_button.setEnabled(False)

    def handle_current_tab_changed(self, idx: int):
        cam_widget: CameraViewWidget = self.widget(idx)
        print(f'this is current now: {cam_widget}')
        if cam_widget is not None and isinstance(cam_widget, CameraViewWidget):
            self.active_camera = cam_widget.camera

    def handle_pause_button_toggled(self, checked: bool):
        if not checked:
            self.pause_button.setIcon(QIcon.fromTheme('media-play'))
        else:
            self.pause_button.setIcon(QIcon.fromTheme('media-pause'))
            self.logging_start = time()
            self.process = Process(target=analyze_daytime_snow, args=(self.to_process, self.result_queue, self.conn1, self.logging_start,))
            self.process.start()
            self.pause_button.setEnabled(True)
            self.pause_button.setIcon(QIcon.fromTheme('media-pause'))
            self.fetch_next_batch()
            self.timer.start(1000)


def analyze_daytime_snow(to_process: Queue, result_queue: Queue, channel: Connection, logging_start: int):
    keep_processing, items = channel.recv() # Tuple[bool, List[Tuple[int, List[Path]]]]

    while keep_processing:
        results: List[Tuple[int, List[Tuple[Path, bool, bool]]]] = []
        for item in items:  # item: Tuple[int, List[Path]]
            cam_id, img_paths = item
            res: List[Tuple[Path, bool, bool]] = []
            for img_path in img_paths:
                img = cv.imread(str(img_path))
                img = cv.resize(img, (0, 0), fx=0.25, fy=0.25, interpolation=cv.INTER_NEAREST)
                gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                daytime = not antar.is_night(img)
                snow = antar.is_snow(gray, img)
                res.append((img_path.name, daytime, snow))
            results.append((cam_id, res))

        channel.send(results)
        keep_processing, items = channel.recv()
