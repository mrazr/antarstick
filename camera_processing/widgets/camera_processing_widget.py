# This Python file uses the following encoding: utf-8
import typing
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
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtCore import pyqtSignal as Signal, QRunnable, QThreadPool, QTimer, pyqtSignal, Qt
from PyQt5.QtCore import pyqtSlot as Slot
from PyQt5.QtGui import QIcon, QColor, QBrush, QPalette
from PyQt5.QtWidgets import QPushButton, QProxyStyle, QStyle, QWidget, QStyleOption, QStyleOptionTab, QTabBar, \
    QProgressDialog, QMessageBox

import camera_processing.stick_detection as antar
from camera import Camera
from camera_processing.widgets.camera_view_widget import CameraViewWidget
from dataset import Dataset, CameraSynchronization


#logging_start = time()
#logging.basicConfig(filename='process.log', level=logging.DEBUG)


class TabProxyStyle(QProxyStyle):
    def __init__(self, base_style_name: str):
        QProxyStyle.__init__(self, base_style_name)
        self.tab_color_map: Dict[str, QBrush] = {}
        self.next_hue = -30
        self.should_offset = True

    def drawControl(self, element: QStyle.ControlElement, option: QStyleOption, painter: QtGui.QPainter,
                    widget: typing.Optional[QWidget] = ...) -> None:
        if element == QStyle.CE_TabBarTab:
            if tab := QStyleOptionTab(option):
                if tab.text in self.tab_color_map:
                    opt = QStyleOptionTab(tab)
                    opt.palette.setBrush(QPalette.Background, self.tab_color_map[tab.text])
                    return super().drawControl(element, opt, painter, widget)

        super().drawControl(element, option, painter, widget)

    def add_new_tab_group(self, tabs: List[str]):
        if self.should_offset:
            self.next_hue += 30
            self.should_offset = False
        else:
            self.should_offset = True
            self.next_hue = (self.next_hue + 180) % 360
        color = QColor.fromHsvF(self.next_hue / 360.0, 1.0, 1.0, 1.0)
        brush = QBrush(color)
        for tab in tabs:
            self.tab_color_map[tab] = brush

    def set_tab_groups(self, groups: List[List[str]]):
        self.next_hue = -30
        self.should_offset = True
        for group in groups:
            self.add_new_tab_group(group)


class Worker(QRunnable):

    def __init__(self, cam_widget: CameraViewWidget, camera: Camera) -> None:
        QRunnable.__init__(self)
        self.cam_widget = cam_widget
        self.camera = camera

    def run(self) -> None:
        self.cam_widget.initialise_with(self.camera)


LINK_MARKERS = ['🔴', '🟠', '🟡', '🟢', '🔵', '🟣', '🟥', '🟧', '🟨', '🟩', '🟦', '🟪']


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
    camera_loaded = pyqtSignal()
    processing_started = pyqtSignal([str, int])
    processing_updated = pyqtSignal([int])
    processing_stopped = pyqtSignal([int])
    stick_verification_needed = pyqtSignal(str)
    no_cameras_open = pyqtSignal()

    def __init__(self):
        QtWidgets.QTabWidget.__init__(self)
        self.tab_style = TabProxyStyle('')
        self.tabBar().setStyle(self.tab_style)
        self.setTabsClosable(True)
        self.tabCloseRequested.connect(self.handle_tab_close_requested)
        self.currentChanged.connect(self.handle_current_tab_changed)
        self.dataset: Optional[Dataset] = None
        self._camera_tab_map: Dict[int, int] = dict({})

        self.assigned_markers: Dict[Camera, Dict[Camera, int]] = {}
        self.marker_ids: List[int] = [len(LINK_MARKERS) - i - 1 for i in range(len(LINK_MARKERS))]
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

        #self.process = #Process(target=analyze_daytime_snow, args=(self.to_process, self.result_queue, self.conn1,))

        self.pause_button = QPushButton(icon=QIcon.fromTheme('media-playback-start'), text='Daytime && snow analysis')
        self.pause_button.setEnabled(False)
        self.pause_button.toggled.connect(self.handle_pause_button_toggled)
        self.pause_button.setCheckable(True)
        self.pause_button.setChecked(False)

        self.active_camera: Optional[Camera] = None
        self.closing = False
        self.cameras_for_linking: typing.Set[Camera] = set()

    camera_link_available = Signal(bool)
    camera_added = Signal(Camera)

    @Slot(Camera)
    def handle_camera_added(self, camera: Camera):
        camera_widget = CameraViewWidget(self.dataset)

        self.camera_link_available.connect(camera_widget.link_cameras_enabled)
        camera_widget.available_for_linking.connect(self.handle_available_for_linking)
        self._camera_tab_map[camera.id] = self.addTab(camera_widget, camera.get_folder_name())
        self.setCurrentIndex(self._camera_tab_map[camera.id])
        self.setTabToolTip(self.currentIndex(), str(camera.folder))

        self.camera_loaded.emit()
        camera_widget.initialization_done.connect(self.handle_camera_widget_initialization_done)
        camera_widget.processing_started.connect(self.handle_camera_processing_started)
        camera_widget.processing_updated.connect(self.handle_camera_processing_updated)
        camera_widget.processing_stopped.connect(self.handle_camera_processing_stopped)
        camera_widget.stick_verification_needed.connect(self.handle_stick_verification_needed)

        camera_widget.initialise_with(camera)
        camera_widget.initialize_rest_of_gui()

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
        if self._camera_tab_map[camera.id] is not None:
            camera_widget = self.widget(self._camera_tab_map[camera.id])
            self.removeTab(self._camera_tab_map[camera.id])
            camera_widget.deleteLater()
        #if len(self.dataset.cameras) < 2:
        #    self.camera_link_available.emit(False)
        self.cameras_for_linking.remove(camera)
        if self.count() == 0:
            self.no_cameras_open.emit()

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

    @Slot()
    def handle_dataset_loading_finished(self):
        for i in range(self.count()):
            cam_widget: CameraViewWidget = self.widget(i)
            cam_widget.initialize_link_menu()

    @Slot(Camera)
    def handle_camera_widget_initialization_done(self, camera: Camera):
        pass
        #if len(self.dataset.cameras) > 1:
        #    self.camera_link_available.emit(True)

        #for cam_id, widget_id in self._camera_tab_map.items():
        #    if cam_id == camera.id:
        #        continue
        #    cam_widget: CameraViewWidget = self.widget(widget_id)
        #    cam_widget.handle_camera_added(camera)

    def cleanup(self):
        #if self.process is not None and self.process.is_alive():
        #    self.process.terminate()
        self.closing = True
        if self.dataset is None:
            return
        self.dataset.save()
        self.dataset.camera_added.disconnect(self.handle_camera_added)
        self.dataset.camera_removed.disconnect(self.handle_camera_removed)
        self.dataset.loading_finished.disconnect(self.handle_dataset_loading_finished)
        self.dataset.cameras_linked.disconnect(self.handle_cameras_linked)
        self.dataset.cameras_unlinked.disconnect(self.handle_cameras_unlinked)
        self.dataset = None
        for i in range(self.count()):
            widget: CameraViewWidget = self.widget(i)
            widget.dispose()
            widget.deleteLater()
        #TODO save camera states
        self.clear()
        self._camera_tab_map: Dict[int, int] = dict({})

    def set_dataset(self, dataset: Dataset):
        self.cleanup()
        self.closing = False
        self.dataset = dataset
        self.dataset.camera_added.connect(self.handle_camera_added)
        self.dataset.camera_removed.connect(self.handle_camera_removed)
        self.dataset.loading_finished.connect(self.handle_dataset_loading_finished)
        self.dataset.cameras_linked.connect(self.handle_cameras_linked)
        self.dataset.cameras_unlinked.connect(self.handle_cameras_unlinked)
        self.dataset.synchronization_finished.connect(self.handle_synchronization_finished)
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
        if cam_widget is not None and isinstance(cam_widget, CameraViewWidget):
            self.active_camera = cam_widget.camera
            if not self.closing:
                cam_widget.recenter_view()

    def handle_pause_button_toggled(self, checked: bool):
        if not checked:
            self.pause_button.setIcon(QIcon.fromTheme('media-playback-start'))
        else:
            self.pause_button.setIcon(QIcon.fromTheme('media-playback-pause'))
            self.logging_start = time()
            #self.process = Process(target=analyze_daytime_snow, args=(self.to_process, self.result_queue, self.conn1, self.logging_start,))
            #self.process.start()
            self.pause_button.setEnabled(True)
            self.pause_button.setIcon(QIcon.fromTheme('media-playback-pause'))
            self.fetch_next_batch()
            self.timer.start(1000)

    def handle_cameras_linked(self, cam1: Camera, cam2: Camera, sync: CameraSynchronization):
        separator = '|'
        cam1_tab = self._camera_tab_map[cam1.id]
        cam1_tab_text = self.tabText(cam1_tab)
        if separator not in cam1_tab_text:
            cam1_tab_text += separator
        marker_id = self.marker_ids.pop()
        cam1_tab_text += LINK_MARKERS[marker_id]
        self.setTabText(cam1_tab, cam1_tab_text)
        cam2_tab = self._camera_tab_map[cam2.id]
        cam2_tab_text = self.tabText(cam2_tab)
        if separator not in cam2_tab_text:
            cam2_tab_text += separator
        cam2_tab_text += LINK_MARKERS[marker_id]
        self.setTabText(cam2_tab, cam2_tab_text)

        cam_markers = self.assigned_markers.setdefault(cam1, {})
        cam_markers[cam2] = marker_id
        self.tabBar().update()

    def handle_cameras_unlinked(self, cam1: Camera, cam2: Camera):
        separator = '|'
        cam1_tab = self._camera_tab_map[cam1.id]
        cam1_tab_text = self.tabText(cam1_tab)

        cam_marker_id = self.assigned_markers.get(cam1, {}).get(cam2, -1)
        if cam_marker_id < 0:
            cam_marker_id = self.assigned_markers[cam2][cam1]

        cam1_tab_text = cam1_tab_text.replace(LINK_MARKERS[cam_marker_id], '')
        if cam1_tab_text.find(separator) == len(cam1_tab_text) - 1:
            cam1_tab_text = cam1_tab_text[:-1]
        self.setTabText(cam1_tab, cam1_tab_text)
        cam2_tab = self._camera_tab_map[cam2.id]
        cam2_tab_text = self.tabText(cam2_tab)
        cam2_tab_text = cam2_tab_text.replace(LINK_MARKERS[cam_marker_id], '')
        try:
            if cam2_tab_text.index(separator) == len(cam2_tab_text) - 1:
                cam2_tab_text = cam2_tab_text[:-1]
        except ValueError:
            pass
        self.marker_ids.append(cam_marker_id)
        self.setTabText(cam2_tab, cam2_tab_text)

        self.tabBar().update()

    def handle_camera_processing_started(self, widget: CameraViewWidget, unprocessed_count: int):
        cam = widget.camera
        self.processing_started.emit(str(cam.folder.name), unprocessed_count)

    def handle_camera_processing_updated(self, processed: int, total: int, job_count: int, processing_stopped: bool):
        self.processing_updated.emit(processed)

    def handle_camera_processing_stopped(self, widget: CameraViewWidget):
        cam = widget.camera
        self.processing_stopped.emit(cam.get_processed_count())

    def handle_stick_verification_needed(self, widget: CameraViewWidget):
        cam = widget.camera
        self.stick_verification_needed.emit(str(cam.folder.name))

    def handle_synchronization_finished(self, sync: CameraSynchronization):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)

        left_img = sync.left_camera.measurements.loc[sync.left_timestamp]['image_name']
        right_img = sync.right_camera.measurements.loc[sync.right_timestamp]['image_name']

        msg.setWindowTitle('Synchronization complete')
        msg.setTextFormat(Qt.RichText)
        msg.setText(f'Cameras {sync.left_camera.folder.name} and {sync.right_camera.folder.name} have been synchronized.')
        msg.setInformativeText(f'The synchronization point is<br><b>{sync.left_camera.folder.name}: {str(sync.left_timestamp)}({left_img})</b>\n\t<->\n<b>{sync.right_camera.folder.name}: {str(sync.right_timestamp)}({right_img})</b>.<br>'
                               f'If you wish to adjust the synchronization, you can do so by manually defining the synchronization point by clicking on <b>Synchronize</b> under the secondary camera.')
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec()

    def handle_available_for_linking(self, cam_widget: CameraViewWidget):
        camera = cam_widget.camera
        self.cameras_for_linking.add(camera)

        for cam_id, widget_id in self._camera_tab_map.items():
            if cam_id == camera.id:
                continue
            cam_widget: CameraViewWidget = self.widget(widget_id)
            cam_widget.handle_camera_added(camera)


