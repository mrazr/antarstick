# This Python file uses the following encoding: utf-8
import typing
from typing import Dict, List, Optional

from PyQt5 import QtWidgets
from PyQt5.QtCore import pyqtSignal as Signal, pyqtSignal, Qt
from PyQt5.QtCore import pyqtSlot as Slot
from PyQt5.QtWidgets import QMessageBox

from camera import Camera
from camera_processing.widgets.camera_view_widget import CameraViewWidget
from dataset import Dataset, CameraSynchronization

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
    camera_link_available = Signal(bool)
    camera_added = Signal(Camera)

    def __init__(self):
        QtWidgets.QTabWidget.__init__(self)
        self.setTabsClosable(True)
        self.tabCloseRequested.connect(self.handle_tab_close_requested)
        self.currentChanged.connect(self.handle_current_tab_changed)
        self.dataset: Optional[Dataset] = None
        self._camera_tab_map: Dict[int, int] = dict({})

        self.assigned_markers: Dict[Camera, Dict[Camera, int]] = {}
        self.marker_ids: List[int] = [len(LINK_MARKERS) - i - 1 for i in range(len(LINK_MARKERS))]

        self.closing = False
        self.cameras_for_linking: typing.Set[Camera] = set()

    @Slot(Camera)
    def handle_camera_added(self, camera: Camera):
        camera_widget = CameraViewWidget(self.dataset)

        self.camera_link_available.connect(camera_widget.link_cameras_enabled)
        camera_widget.available_for_linking.connect(self.handle_available_for_linking)
        self._camera_tab_map[camera.id] = self.addTab(camera_widget, camera.get_folder_name())
        self.setCurrentIndex(self._camera_tab_map[camera.id])
        self.setTabToolTip(self.currentIndex(), str(camera.folder))

        self.camera_loaded.emit()
        camera_widget.processing_started.connect(self.handle_camera_processing_started)
        camera_widget.processing_updated.connect(self.handle_camera_processing_updated)
        camera_widget.processing_stopped.connect(self.handle_camera_processing_stopped)
        camera_widget.stick_verification_needed.connect(self.handle_stick_verification_needed)

        camera_widget.initialise_with(camera)
        camera_widget.initialize_rest_of_gui()

    @Slot(Camera)
    def handle_camera_removed(self, camera: Camera):
        #TODO handle camera removed while the same camera is analyzed in self.process
        if self._camera_tab_map[camera.id] is not None:
            camera_widget = self.widget(self._camera_tab_map[camera.id])
            self.removeTab(self._camera_tab_map[camera.id])
            camera_widget.deleteLater()
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

    def cleanup(self):
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

    def handle_current_tab_changed(self, idx: int):
        cam_widget: CameraViewWidget = self.widget(idx)
        if cam_widget is not None and isinstance(cam_widget, CameraViewWidget):
            if not self.closing:
                cam_widget.recenter_view()

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

    @staticmethod
    def handle_synchronization_finished(sync: CameraSynchronization):
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


