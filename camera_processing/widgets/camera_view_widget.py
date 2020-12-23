import json
import time
from enum import IntEnum
from multiprocessing import Pool
import multiprocessing
from multiprocessing.synchronize import Lock
from pathlib import Path
from queue import Queue
from typing import List, Dict, Optional, Any, Tuple
import itertools

import cv2 as cv
import numpy as np
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtCore import (QMarginsF, QModelIndex, QPointF, QRectF, Qt,
                          pyqtSignal, QByteArray, QRect, QItemSelection, QPoint)
from PyQt5.QtCore import pyqtSlot as Slot, QTimer, QMutex
from PyQt5.QtGui import QImage, QPixmap, QPainter, QBrush, QColor, QFont, QPen
from PyQt5.QtWidgets import QGraphicsScene, QGraphicsItem, QSizePolicy, QAbstractScrollArea, QMessageBox
import matplotlib.pyplot as plt
import skimage.filters as fil
import pandas as pd
import exifread

import camera_processing.antarstick_processing
import camera_processing.stick_detection as antar
import camera_processing.antarstick_processing as snow
from camera import Camera, PD_IMAGE_STATE, PhotoState
from camera_processing import antarstick_processing
from camera_processing.widgets import ui_camera_view
from camera_processing.widgets.button import Button, ButtonColor
from camera_processing.widgets.button_menu import ButtonMenu
from camera_processing.widgets.cam_graphics_view import CamGraphicsView
from camera_processing.widgets.camera_view import CameraView
from camera_processing.widgets.overlay_gui import OverlayGui
from camera_processing.widgets.split_view import SplitView
from camera_processing.widgets.stick_length_input import TextInputWidget
from camera_processing.widgets.stick_link_manager import StickLinkManager, CameraToCameraStickLinkingStrategy, MovedSticksLinkingStrategy
from camera_processing.widgets.stick_widget import StickMode, StickWidget
from dataset import Dataset, CameraSynchronization
from image_list_model import ImageListModel
from stick import Stick
from stick_detection_dialog import StickDetectionDialog
from thumbnail_storage import ThumbnailDelegate


class CameraSide(IntEnum):
    Hidden = 0,
    Left = 1,
    Right = 2,


class SideCameraState(IntEnum):
    Vacant = 0,
    Adding = 1,
    Shown = 2,
    Hidden = 3,
    Unavailable = 4,  # no other camera available for linking
    Synchronizing = 5,


class CameraViewWidget(QtWidgets.QWidget):

    sticks_changed = pyqtSignal()
    initialization_done = pyqtSignal(Camera)
    processing_started = pyqtSignal(['PyQt_PyObject', int])
    processing_updated = pyqtSignal([int, int, int, bool])
    processing_stopped = pyqtSignal('PyQt_PyObject')
    stick_verification_needed = pyqtSignal('PyQt_PyObject')
    timestamps_extracted = pyqtSignal('PyQt_PyObject')

    BTN_LEFT_ADD = 'btn_left_add'
    BTN_LEFT_SHOW = 'btn_left_show'
    BTN_RIGHT_ADD = 'btn_right_add'
    BTN_RIGHT_SHOW = 'btn_right_show'

    def __init__(self, dataset: Dataset):
        QtWidgets.QWidget.__init__(self)

        self.ui = ui_camera_view.Ui_CameraView()
        self.ui.setupUi(self)

        self.image_list = ImageListModel()
        thumbnail_delegate = ThumbnailDelegate(self.image_list.thumbnails)
        #self.ui.image_list.setHorizontalHeader(None)
        #self.ui.image_list.setVerticalHeader(None)
        self.ui.image_list.setSpacing(10)
        self.ui.image_list.setUniformItemSizes(False)
        self.ui.image_list.setModel(self.image_list)
        self.ui.image_list.verticalScrollBar().sliderPressed.connect(self.image_list.handle_slider_pressed)
        self.ui.image_list.verticalScrollBar().sliderReleased.connect(self.handle_image_list_slider_released)
        #self.ui.image_list.verticalHeader().setDefaultSectionSize(self.image_list.thumb.height())
        self.ui.image_list.selectionModel().currentChanged.connect(self.handle_list_model_current_changed)
        self.ui.image_list.selectionModel().selectionChanged.connect(self.handle_image_selection_changed)
        self.ui.image_list.setEnabled(False)
        self.ui.image_list.setItemDelegate(thumbnail_delegate)
        self.ui.image_list.setSizePolicy(QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Preferred))

        self.dataset = dataset
        self.dataset.cameras_linked.connect(self.handle_cameras_linked)
        self.dataset.cameras_unlinked.connect(self.handle_cameras_unlinked)
        self.dataset.camera_removed.connect(self.handle_camera_removed)
        self.dataset.synchronization_finished.connect(self.handle_synchronization_finished)
        self.camera: Optional[Camera] = None
        self.graphics_scene = QGraphicsScene()

        self.stick_link_manager = StickLinkManager()
        self.stick_link_manager_strat = CameraToCameraStickLinkingStrategy(self.dataset, self.camera, self.stick_link_manager)
        self.graphics_scene.addItem(self.stick_link_manager)
        self.stick_link_manager.setZValue(2)
        self.stick_link_manager.setVisible(False)

        self.split_view: Optional[SplitView] = None
        self.viewed = 'camera'

        self.link_manager2 = StickLinkManager()
        self.graphics_scene.addItem(self.link_manager2)
        self.link_manager2.setZValue(2)
        self.link_manager2.setVisible(False)

        self.moved_sticks_linking = MovedSticksLinkingStrategy()
        self.moved_sticks_linking.set_view(self.link_manager2)

        self.graphics_view = CamGraphicsView(self.stick_link_manager_strat, self.moved_sticks_linking, self)
        self.graphics_view.setSizePolicy(QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding))

        self.ui.graphicsViewLayout.addWidget(self.graphics_view)

        self.graphics_view.setScene(self.graphics_scene)
        self.graphics_view.rubberBandChanged.connect(self.handle_cam_view_rubber_band_changed)
        self.graphics_view.rubber_band_started.connect(self.handle_cam_view_rubber_band_started)

        self.current_viewed_image: Optional[np.ndarray] = None
        self.scaling = 2.0

        self.camera_view = CameraView(self.scaling)
        self.camera_view.stick_context_menu.connect(self.handle_stick_widget_context_menu)
        self.camera_view.setAcceptHoverEvents(False)
        self.camera_view.setZValue(3)
        self.camera_view.control_widget.show_sync_button(False)
        self.camera_view.next_photo_clicked.connect(self.handle_next_photo_clicked)
        self.camera_view.previous_photo_clicked.connect(self.handle_previous_photo_clicked)

        self.graphics_scene.addItem(self.camera_view)

        self.stick_widgets: List[StickWidget] = []
        self.detected_sticks: List[Stick] = []
        self.link_menus = dict({'right': None, 'left': None})
        self.left_link: Optional[CameraView] = None
        self.right_link: Optional[CameraView] = None

        self.overlay_gui = OverlayGui(self.graphics_view)
        self.overlay_gui.reset_view_requested.connect(self.recenter_view)
        self.overlay_gui.edit_sticks_clicked.connect(self.handle_edit_sticks_clicked)
        self.overlay_gui.link_sticks_clicked.connect(self.handle_link_sticks_clicked)
        self.overlay_gui.delete_sticks_clicked.connect(self.handle_delete_sticks_clicked)
        self.overlay_gui.process_photos_clicked.connect(self.handle_process_photos_clicked_mp)
        self.overlay_gui.clicked.connect(self.handle_overlay_gui_clicked)
        self.overlay_gui.find_sticks_clicked.connect(self.handle_find_sticks_clicked)
        self.overlay_gui.mes.connect(self.handle_mes)
        self.overlay_gui.measure_all.connect(self.handle_measure_all_clicked)
        self.detect_thin_sticks = False
        #self.overlay_gui.stick_length_input.input_entered.connect(self.handle_stick_length_entered)
        self.overlay_gui.sticks_length_clicked.connect(self.handle_stick_length_clicked)
        self.overlay_gui.confirm_sticks_clicked.connect(self.handle_confirm_sticks_clicked)
        self.overlay_gui.set_stick_label_clicked.connect(self.handle_set_stick_label_clicked)
        self.overlay_gui.set_stick_length_clicked.connect(self.handle_set_stick_length_clicked)
        #self.overlay_gui.save_measurements.connect(self.handle_save_measurements)
        self.overlay_gui.process_stop_clicked.connect(self.handle_process_stop_clicked)
        self.overlay_gui.show_measurements.connect(self.handle_show_measurements)
        self.overlay_gui.exclude_photos.connect(self.handle_exclude_photos_clicked)
        self.overlay_gui.include_photos.connect(self.handle_include_photos_clicked)
        self.overlay_gui.measurement_mode_toggle.connect(self.handle_measurement_mode_toggled)
        self.processing_updated.connect(self.overlay_gui.handle_process_count_changed)
        self.processing_stopped.connect(self.overlay_gui.handle_processing_stopped)
        #self.overlay_gui.set_stick_label.connect(self.handle_set_stick_label_clicked)

        self.graphics_scene.addItem(self.overlay_gui)
        #self.overlay_gui.initialize()

        self.link_menu = ButtonMenu(self.scaling, self.overlay_gui)
        self.link_menu.show_close_button(True)
        self.link_menu_position: CameraSide = CameraSide.Hidden
        self.link_menu.setZValue(100)
        self.link_menu.setVisible(False)
        self.link_menu.set_layout_direction("vertical")
        self.link_menu.close_requested.connect(self.handle_link_menu_close_requested)
        self.link_menu.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)

        self.links_available = False

        self.return_queue = Queue()
        self.result_timer = QTimer()
        self.result_timer.setInterval(1000)
        self.result_timer.timeout.connect(self.process_result_queue)
        self.worker_pool = Pool(processes=16)
        self.image_loading_time: float = -1.0

        self.photo_count_to_process: int = 0
        self.photo_count_processed: int = 0
        self.next_batch_start: int = 0
        self.photo_batch: List[str] = []
        self.timer: QTimer = QTimer()
        self.fix_timer: QTimer = QTimer()

        self.stick_detection_dialog = StickDetectionDialog()
        self.stick_detection_dialog.spinLength.valueChanged.connect(self.detect_sticks)
        self.stick_detection_dialog.spinWidth.valueChanged.connect(self.detect_sticks)
        self.stick_detection_dialog.spinP0.valueChanged.connect(self.detect_sticks)
        self.stick_detection_dialog.buttonBox.clicked.connect(lambda _: cv.destroyAllWindows())
        self.stick_detection_dialog.spinSensitivity.valueChanged.connect(self.detect_sticks)
        self.overlay_gui.redetect_sticks_clicked.connect(self.handle_redetect_sticks_clicked_)
        self.stick_detection_dialog.btnApply.clicked.connect(self.update_detection_params)

        self.left_add_button = Button(self.BTN_LEFT_ADD, '+', parent=self.camera_view)
        self.left_side_camera_state = SideCameraState.Unavailable

        self.right_add_button = Button(self.BTN_RIGHT_ADD, '+', parent=self.camera_view)
        self.right_side_camera_state = SideCameraState.Unavailable

        self.left_show_button = Button(self.BTN_LEFT_SHOW, 'Show', parent=self.camera_view)
        self.right_show_button = Button(self.BTN_RIGHT_SHOW, 'Show', parent=self.camera_view)

        self._setup_buttons()
        self.set_side_camera_state(SideCameraState.Unavailable, CameraSide.Left)
        self.set_side_camera_state(SideCameraState.Unavailable, CameraSide.Right)

        self.rect_to_view = QRectF()
        self.stick_widget_in_context: Optional[StickWidget] = None
        self.confirmation_queue: Queue = Queue(16)
        self.queue_lock = QMutex()
        self.current_sticks: List[Stick] = []
        self.running_jobs: int = 0
        self.paused_jobs: int = 0
        self.job_counter_lock = QMutex()
        self.processing_should_continue: bool = False
        self.single_proc = False
        self.skip = False
        self.rem_photos = []
        self.s_sticks = []
        self.overlay_gui.use_single_proc.connect(self.handle_use_single)
        self.sync: Dict[CameraSide, CameraSynchronization] = {}
        self.timestamps_extracted.connect(self.handle_timestamps_extracted)

    def _setup_buttons(self):
        self.left_add_button.set_label('Link camera', direction='vertical')
        self.left_add_button.set_is_check_button(True)  # , ['green', 'red'])
        self.left_add_button.set_base_color([ButtonColor.GREEN, ButtonColor.RED])
        self.left_add_button.setZValue(3)
        self.left_add_button.setFlag(QGraphicsItem.ItemIgnoresTransformations, False)
        self.left_add_button.set_height(18)
        self.left_add_button.clicked.connect(self.handle_link_camera_button_left_clicked)
        self.left_add_button.hovered_.connect(self.handle_side_button_hovered)

        self.left_show_button.set_label('Hide', direction='vertical')
        self.left_show_button.set_is_check_button(True)
        self.left_show_button.set_base_color([ButtonColor.RED, ButtonColor.GREEN])
        self.left_show_button.setZValue(3)
        self.left_show_button.setFlag(QGraphicsItem.ItemIgnoresTransformations, False)
        self.left_show_button.set_height(18)
        self.left_show_button.clicked.connect(self.handle_side_show_button_clicked)
        self.left_show_button.hovered_.connect(self.handle_side_button_hovered)

        self.right_add_button.set_label('Link camera', direction='vertical')
        self.right_add_button.set_is_check_button(True)  # , ['green', 'red'])
        self.right_add_button.set_base_color([ButtonColor.GREEN, ButtonColor.RED])
        self.right_add_button.setZValue(3)
        self.right_add_button.setFlag(QGraphicsItem.ItemIgnoresTransformations, False)
        self.right_add_button.set_height(18)
        self.right_add_button.clicked.connect(self.handle_link_camera_button_right_clicked)
        self.right_add_button.hovered_.connect(self.handle_side_button_hovered)

        self.right_show_button.set_label('Hide', direction='vertical')
        self.right_show_button.set_is_check_button(True)
        self.right_show_button.set_base_color([ButtonColor.RED, ButtonColor.GREEN])
        self.right_show_button.setZValue(3)
        self.right_show_button.setFlag(QGraphicsItem.ItemIgnoresTransformations, False)
        self.right_show_button.set_height(18)
        self.right_show_button.clicked.connect(self.handle_side_show_button_clicked)
        self.right_show_button.hovered_.connect(self.handle_side_button_hovered)

    def handle_redetect_sticks_clicked_(self):
        param_json = json.dumps(antar.params)
        self.stick_detection_dialog.paramsText.setPlainText(param_json)
        self.stick_detection_dialog.show()

    def initialise_with(self, camera: Camera):
        self.camera = camera
        self.camera.sticks_added.connect(lambda: self.camera.timestamps_available and
                                                 self.overlay_gui.enable_confirm_sticks_button(True))
        self.camera.sticks_removed.connect(self.handle_stick_removed)
        self.camera.stick_removed.connect(self.handle_stick_removed)
        self.camera.non_increasingness.connect(self.handle_camera_temporal_non_increasigness_found)
        if self.camera.rep_image is None:
            half = (int(round(0.5 * self.camera.standard_image_size[0])),
                    int(round(0.5 * self.camera.standard_image_size[1])))
            self.camera.rep_image = cv.resize(cv.imread(str(self.camera.folder / self.camera.rep_image_path)),
                                              dsize=half)
        self.stick_link_manager_strat.camera = self.camera
        self.stick_link_manager_strat.update_links()
        self.image_list.initialize(self.camera, self.camera.get_processed_count())
        self.ui.image_list.setEnabled(True)
        self.ui.image_list.setModel(self.image_list)
        self.ui.image_list.setSpacing(0)
        self.ui.image_list.updateGeometry()
        select_index = self.image_list.index(0, 0)
        self.ui.image_list.setCurrentIndex(select_index)
        self.overlay_gui.stick_length_input.set_value(str(self.camera.default_stick_length_cm))
        #self.initialize_rest_of_gui()
        if not self.camera.timestamps_available:
            self.camera_view.show_status_message("Extracting timestamps...")
            self.worker_pool.apply_async(self.extract_timestamps, args=(self.camera.image_list, self.camera.folder),
                                         callback=self.timestamps_extracted.emit)

    def initialize_rest_of_gui(self):
        self.overlay_gui.initialize()
        viewport_rect = self.graphics_view.viewport().rect()
        _re = self.graphics_view.mapToScene(viewport_rect)
        self.graphics_scene.setSceneRect(QRectF(_re.boundingRect()))

        self.camera_view.initialise_with(self.camera)
        if not self.camera.timestamps_available:
            self.camera_view.show_status_message("Extracting timestamps...")

        #self.camera_view.set_show_title(False)
        self.camera_view.setPos(QPointF(0.5 * self.camera_view.boundingRect().width(), 0))
        self.left_add_button.set_button_height(self.camera_view.boundingRect().height())
        #self.left_add_button.setPos(self.camera_view.get_top_left() -
        #                            QPointF(self.left_add_button.boundingRect().width(), 0))
        self.left_add_button.setPos(QPointF(-self.left_add_button.boundingRect().width(), 0.0))

        self.right_add_button.set_button_height(self.camera_view.boundingRect().height())
        #self.right_add_button.setPos(self.camera_view.get_top_right())
        self.right_add_button.setPos(QPointF(self.camera_view.boundingRect().width(), 0.0))

        self.left_show_button.set_button_height(int(0.5 * self.camera_view.boundingRect().height()))
        #self.left_show_button.setPos(self.left_add_button.pos() +
        #                             QPointF(0, 0.5 * self.camera_view.boundingRect().height()))
        self.left_show_button.setPos(QPointF(-self.left_add_button.boundingRect().width(),
                                             0.5 * self.camera_view.boundingRect().height()))

        self.right_show_button.set_button_height(int(0.5 * self.camera_view.boundingRect().height()))
        #self.right_show_button.setPos(self.right_add_button.pos() +
        #                              QPointF(0, 0.5 * self.camera_view.boundingRect().height()))
        self.right_show_button.setPos(QPointF(self.camera_view.boundingRect().width(),
                                             0.5 * self.camera_view.boundingRect().height()))

        self.stick_link_manager_strat.primary_camera = self.camera_view

        self.ui.image_list.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)
        self.ui.image_list.setMinimumWidth(self.ui.image_list.sizeHintForColumn(0) +
                                         self.ui.image_list.verticalScrollBar().width() + 20)

        self.initialize_link_menu()
        self.overlay_gui.show_loading_screen(False)
        self.overlay_gui.process_photos_with_jobs_clicked.connect(self.handle_process_photos_clicked_mp)
        self.overlay_gui.handle_cam_view_changed()
        self.initialization_done.emit(self.camera)
        self.recenter_view()
        self.graphics_view.view_changed.emit()
        self.fix_timer.setSingleShot(True)
        self.fix_timer.timeout.connect(lambda: self.recenter_view())
        self.fix_timer.setInterval(5)
        self.fix_timer.start()

        if len(self.camera.sticks) == 0 or not self.camera.timestamps_available:
            self.overlay_gui.enable_confirm_sticks_button(False)

    @Slot(bool)
    def link_cameras_enabled(self, enabled: bool):
        if enabled:
            if self.left_side_camera_state == SideCameraState.Unavailable:
                self.set_side_camera_state(SideCameraState.Vacant, CameraSide.Left)
            if self.right_side_camera_state == SideCameraState.Unavailable:
                self.set_side_camera_state(SideCameraState.Vacant, CameraSide.Right)
        else:
            self.set_side_camera_state(SideCameraState.Unavailable, CameraSide.Left)
            self.set_side_camera_state(SideCameraState.Unavailable, CameraSide.Right)
        self.links_available = enabled

    def recenter_view(self):
        if self.viewed == 'camera':
            self.rect_to_view = self.camera_view.sceneBoundingRect().united(self.left_add_button.sceneBoundingRect())
            self.rect_to_view = self.rect_to_view.united(self.right_add_button.sceneBoundingRect())

            if self.left_link is not None and self.left_side_camera_state == SideCameraState.Shown:
                self.rect_to_view = self.rect_to_view.united(self.left_link.sceneBoundingRect())
            if self.right_link is not None and self.right_side_camera_state == SideCameraState.Shown:
                self.rect_to_view = self.rect_to_view.united(self.right_link.sceneBoundingRect())
        else:
            self.rect_to_view = self.split_view.sceneBoundingRect()

        self.graphics_scene.setSceneRect(self.rect_to_view.marginsAdded(QMarginsF(50, 50, 50, 50)))

        self.graphics_view.fitInView(self.rect_to_view, Qt.KeepAspectRatio)
        self.graphics_scene.update(self.rect_to_view)
        self.link_menu.set_layout_direction('vertical')

    @Slot(QModelIndex, QModelIndex)
    def handle_list_model_current_changed(self, current: QModelIndex, previous: QModelIndex):
        image_path = self.image_list.data(current, Qt.UserRole)
        half = (int(round(0.5 * self.camera.standard_image_size[0])),
                int(round(0.5 * self.camera.standard_image_size[1])))
        img = cv.imread(str(image_path))
        if img.shape == self.camera.standard_image_size:
            self.current_viewed_image = cv.pyrDown(img)
        else:
            self.current_viewed_image = cv.resize(img, dsize=half, interpolation=cv.INTER_LINEAR)
        self.camera_view.set_image(self.current_viewed_image, image_path.name)
        self.current_sticks = self.camera.get_sticks_in_image(image_path.name)

        if self.left_side_camera_state == SideCameraState.Shown:
            if self.left_link.control_widget.mode != 'sync':
                sync = self.sync[CameraSide.Left]
                left_cam = sync.left_camera
                reciprocal_image = sync.get_reciprocal_image_by_name(self.camera, image_path.name)
                if reciprocal_image is not None:
                    reciprocal_image = reciprocal_image[0]
                    rec_img = cv.imread(str(left_cam.folder / reciprocal_image))
                    rec_img = self.standardized_image(rec_img)
                    self.left_link.set_image(rec_img, reciprocal_image)
                    left_sticks = left_cam.get_sticks_in_image(reciprocal_image)
                    for sw in self.left_link.stick_widgets:
                        for st in left_sticks:
                            if st.label == sw.stick.label:
                                sw.set_stick(st)
                else:
                    self.left_link.set_image(None)

        if self.right_side_camera_state == SideCameraState.Shown:
            if self.right_link.control_widget.mode != 'sync':
                sync = self.sync[CameraSide.Right]
                right_cam = sync.right_camera
                reciprocal_image = sync.get_reciprocal_image_by_name(self.camera, image_path.name)
                if reciprocal_image is not None:
                    reciprocal_image = reciprocal_image[0]
                    rec_img = cv.imread(str(right_cam.folder / reciprocal_image))
                    rec_img = self.standardized_image(rec_img)
                    self.right_link.set_image(rec_img, reciprocal_image)
                    right_sticks = right_cam.get_sticks_in_image(reciprocal_image)
                    for sw in self.right_link.stick_widgets:
                        for st in right_sticks:
                            if st.label == sw.stick.label:
                                sw.set_stick(st)
                else:
                    self.right_link.set_image(None)

        for sw in self.camera_view.stick_widgets:
            for st in self.current_sticks:
                if st.label == sw.stick.label:
                    sw.set_stick(st)
        #TODO synchronize linked camera views

    def handle_image_selection_changed(self, sel: QItemSelection, des: QItemSelection):
        indexes = self.ui.image_list.selectedIndexes()
        if len(indexes) > 3: # greater than 3 because I'm using TableView with 3 columns, so this is equivalent of one selected row
            first_idx = indexes[0]
            last_idx = indexes[-1]
            tags = set(self.camera.measurements.iloc[first_idx.row():last_idx.row(), PD_IMAGE_STATE].to_list())
            if PhotoState.Skipped in tags:
                self.overlay_gui.show_include_button()
            else:
                self.overlay_gui.hide_include_button()
            if PhotoState.Unprocessed in tags:
                self.overlay_gui.show_exclude_button()
            else:
                self.overlay_gui.hide_exclude_button()
        else:
            self.overlay_gui.hide_exclude_include_menu()

    def standardized_image(self, img: np.ndarray, camera: Optional[Camera] = None) -> np.ndarray:
        if camera is None:
            camera = self.camera
        half = (int(round(0.5 * camera.standard_image_size[0])),
                int(round(0.5 * camera.standard_image_size[1])))
        if img.shape == camera.standard_image_size:
            return cv.pyrDown(img)
        return cv.resize(img, dsize=half, interpolation=cv.INTER_LINEAR)

    @Slot()
    def handle_edit_sticks_clicked(self):
        edit_sticks_on = self.overlay_gui.edit_sticks_button_pushed()
        if self.viewed == 'camera':
            if edit_sticks_on:
                self.overlay_gui.uncheck_confirm_sticks_button()
                self.camera_view.set_stick_widgets_mode(StickMode.EditDelete)
            else:
                self.camera_view.set_stick_widgets_mode(StickMode.Display)
                if len(self.camera_view.sticks_without_width) > 0:
                    gray = cv.cvtColor(self.current_viewed_image, cv.COLOR_BGR2GRAY)
                    dx, dy = cv.Sobel(gray, cv.CV_32F, 1, 0), cv.Sobel(gray, cv.CV_32F, 0, 1)
                    mag = cv.magnitude(dx, dy)
                    for stick in self.camera_view.sticks_without_width:
                        edge_offsets, _ = antar.line_edge_offsets(stick.line(), mag, 25)
                        stick.width = int(edge_offsets[0] + edge_offsets[1])
                        print(f'stick {stick.label} is {stick.width} px wide')
                    self.camera_view.sticks_without_width.clear()
        else:
            if edit_sticks_on:
                self.link_manager2.hide_links()
                self.split_view.set_stick_widgets_mode(StickMode.Edit)
            else:
                self.link_manager2.show_links()
                self.split_view.set_stick_widgets_mode(StickMode.Display)
        self.graphics_scene.update()
    
    @Slot()
    def handle_sticks_changed(self):
        if self.left_link is not None:
            self.left_link.update_stick_widgets()
        if self.right_link is not None:
            self.right_link.update_stick_widgets()
        self.sync_stick_link_manager()
        self.graphics_scene.update()
    
    @Slot('PyQt_PyObject')
    def handle_stick_link_requested(self, stick_widget: StickWidget):
        for sw in self.stick_widgets:
            sw.set_available_for_linking(False)

        if self.left_link is not None:
            for sw in self.left_link.stick_widgets:
                sw.set_available_for_linking(True)
        
        if self.right_link is not None:
            for sw in self.right_link.stick_widgets:
                sw.set_available_for_linking(True)

        self.stick_link_manager.handle_stick_widget_link_requested(stick_widget)
        self.graphics_scene.update()

    def stick_widgets_set_mode(self, mode: StickMode):
        for sw in self.stick_widgets:
            sw.set_mode(mode)
    
    def handle_link_sticks_clicked(self):
        self.left_add_button.set_disabled(self.overlay_gui.link_sticks_button_pushed())
        self.right_add_button.set_disabled(self.overlay_gui.link_sticks_button_pushed())

        if self.left_side_camera_state == SideCameraState.Hidden:
            self.left_show_button.click_button(artificial_emit=True)

        if self.right_side_camera_state == SideCameraState.Hidden:
            self.right_show_button.click_button(artificial_emit=True)

        if self.overlay_gui.link_sticks_button_pushed():
            self.stick_link_manager.set_rect(self.rect_to_view)
            self.stick_link_manager.setZValue(1)
            self.stick_link_manager_strat.start()
        else:
            self.stick_link_manager_strat.stop()

    def sync_stick_link_manager(self):
        cams = []
        if self.left_link is not None:
            cams.append(self.left_link)
        if self.right_link is not None:
            cams.append(self.right_link)

        self.stick_link_manager_strat.set_secondary_cameras(cams)
    
    def _destroy(self):
        self.graphics_scene.removeItem(self.stick_link_manager)

    def handle_cameras_linked(self, cam1: Camera, cam2: Camera, sync: CameraSynchronization):
        if cam1.id != self.camera.id and cam2.id != self.camera.id:
            return
        other_camera_view = CameraView(self.scaling)
        other_camera_view.stick_context_menu.connect(self.handle_stick_widget_context_menu)
        self.graphics_scene.addItem(other_camera_view)
        other_camera_view.setAcceptHoverEvents(False)

        other_camera_view.synchronize_clicked.connect(self.handle_synchronize_clicked)
        other_camera_view.previous_photo_clicked.connect(self.handle_previous_photo_clicked)
        other_camera_view.next_photo_clicked.connect(self.handle_next_photo_clicked)
        other_camera_view.sync_cancel_clicked.connect(self.handle_sync_cancel_clicked)
        other_camera_view.sync_confirm_clicked.connect(self.handle_sync_confirm_clicked)
        other_camera_view.first_photo_clicked.connect(self.handle_first_photo_clicked)

        other_camera_view.set_display_mode()
        other_camera_view.set_show_stick_widgets(True)
        cam_position = CameraSide.Right

        if self.camera.id == cam1.id:  # self.camera is on the left
            cam = cam2
        else:  # self.camera is on the right
            cam = cam1
            cam_position = CameraSide.Left

        self.link_menu.hide_button(str(cam.folder))
        if self.link_menu.visible_buttons_count() == 0:
            if self.left_side_camera_state == SideCameraState.Vacant:
                self.set_side_camera_state(SideCameraState.Unavailable, CameraSide.Left)
            if self.right_side_camera_state == SideCameraState.Vacant:
                self.set_side_camera_state(SideCameraState.Unavailable, CameraSide.Right)
        other_camera_view.initialise_with(cam)

        pos: QPointF = self.camera_view.pos()
        if cam_position == CameraSide.Left:
            pos = self.left_add_button.scenePos() - QPointF(self.camera_view.boundingRect().width(), 0)
            if self.left_link is not None:
                self.dataset.unlink_cameras(self.camera, self.left_link.camera)
            self.left_link = other_camera_view
            self.left_link.setZValue(3)
            self.set_side_camera_state(SideCameraState.Shown, CameraSide.Left)
            self.sync[CameraSide.Left] = sync
        else:
            pos = self.right_add_button.scenePos() + QPointF(self.right_add_button.boundingRect().width(), 0)
            if self.right_link is not None:
                self.dataset.unlink_cameras(self.camera, self.right_link.camera)
            self.right_link = other_camera_view
            self.right_link.setZValue(3)
            self.set_side_camera_state(SideCameraState.Shown, CameraSide.Right)
            self.sync[CameraSide.Right] = sync

        other_camera_view.setPos(pos)
        other_camera_view.stick_link_requested.connect(self.stick_link_manager_strat.handle_stick_widget_link_requested)

        self.recenter_view()

        self.sync_stick_link_manager()
        self.overlay_gui.enable_link_sticks_button(True)

    def handle_cameras_unlinked(self, cam1: Camera, cam2: Camera):
        if cam1.id != self.camera.id and cam2.id != self.camera.id:
            return
        to_remove = cam1 if cam2.id == self.camera.id else cam2
        self.link_menu.show_button(str(to_remove.folder))

        if self.left_link is not None and self.left_link.camera.id == to_remove.id:
            self.left_link.stick_link_requested.disconnect(self.stick_link_manager_strat.handle_stick_widget_link_requested)
            self.left_link.stick_context_menu.disconnect(self.handle_stick_widget_context_menu)
            self.left_link.setParentItem(None)
            self.graphics_scene.removeItem(self.left_link)
            self.left_link = None
            self.set_side_camera_state(SideCameraState.Vacant, CameraSide.Left)
        elif self.right_link is not None and self.right_link.camera.id == to_remove.id:
            self.right_link.stick_link_requested.disconnect(self.stick_link_manager_strat.handle_stick_widget_link_requested)
            self.right_link.stick_context_menu.disconnect(self.handle_stick_widget_context_menu)
            self.right_link.setParentItem(None)
            self.graphics_scene.removeItem(self.right_link)
            self.right_link = None
            self.set_side_camera_state(SideCameraState.Vacant, CameraSide.Right)
        if self.left_link is None and self.right_link is None:
            self.overlay_gui.enable_link_sticks_button(False)

        if self.link_menu.visible_buttons_count() > 0:
            if self.left_side_camera_state == SideCameraState.Unavailable:
                self.set_side_camera_state(SideCameraState.Vacant, CameraSide.Left)
            if self.right_side_camera_state == SideCameraState.Unavailable:
                self.set_side_camera_state(SideCameraState.Vacant, CameraSide.Right)

        self.recenter_view()

    def handle_link_camera_button_left_clicked(self, data: Dict[str, Any]):
        self.handle_link_camera_button_clicked(CameraSide.Left, data['button'].is_on())

    def handle_link_camera_button_right_clicked(self, data: Dict[str, Any]):
        self.handle_link_camera_button_clicked(CameraSide.Right, data['button'].is_on())

    def handle_link_camera_button_clicked(self, button_position: CameraSide, is_pushed: bool):
        if not is_pushed:
            self.dataset.unlink_cameras(self.camera,
                                        self.left_link.camera if button_position == CameraSide.Left
                                        else self.right_link.camera)
            return
        self.link_menu.center_buttons()
        if self.link_menu_position != CameraSide.Hidden:
            self.set_side_camera_state(SideCameraState.Vacant, self.link_menu_position)

        self.link_menu_position = button_position
        self.set_side_camera_state(SideCameraState.Adding, self.link_menu_position)
        self.adjust_link_menu_position()
        self.link_menu.setVisible(True)

    def adjust_link_menu_position(self):
        self.link_menu.setPos(self.graphics_view.size().width() * 0.5 - self.link_menu.boundingRect().width() * 0.5,
                              self.graphics_view.size().height() * 0.5 - self.link_menu.boundingRect().height() * 0.5)

    def initialize_link_menu(self):
        for cam in self.dataset.cameras:
            if cam.id == self.camera.id:
                continue
            self.handle_camera_added(cam)

    def handle_camera_added(self, camera: Camera):
        img = camera.rep_image
        pixmap = None
        if img is None:
            pixmap = QPixmap(170, 128)
            painter = QPainter()
            pixmap.fill(QColor(0, 125, 125, 100))
            painter.begin(pixmap)
            painter.setRenderHint(QPainter.TextAntialiasing)
            font = QFont(Button.font)
            font.setPixelSize(14)
            painter.setFont(font)
            brush = QBrush(QColor(255, 255, 255, 100))
            pen = QPen()
            pen.setWidth(1.5)
            pen.setBrush(brush)
            painter.setPen(pen)
            painter.drawText(pixmap.rect(), Qt.AlignCenter, "initializing")
            painter.end()
        else:
            barray = QByteArray(img.tobytes())
            image = QImage(barray, img.shape[1], img.shape[0], QImage.Format_BGR888)
            pixmap = QPixmap.fromImage(image)
        btn = self.link_menu.get_button(btn_id=str(camera.folder))
        if btn is None:
            btn = self.link_menu.add_button(btn_id=str(camera.folder), label=str(camera.folder.name), pixmap=pixmap)
            btn.clicked.connect(self.handle_camera_button_clicked)
        else:
            btn.set_pixmap(pixmap)
            self.link_menu.set_layout_direction(self.link_menu.layout_direction)
        self.adjust_link_menu_position()

    def handle_camera_removed(self, camera: Camera):
        if self.camera.id == camera.id:
            return
        button = self.link_menu.get_button(str(camera.folder))
        self.link_menu.remove_button(str(camera.folder))
        button.setParentItem(None)
        self.graphics_scene.removeItem(button)
        button.deleteLater()

    def handle_camera_button_clicked(self, btn_dict: Dict[str, str]):
        btn_id = btn_dict["btn_id"]
        camera = self.dataset.get_camera(btn_id)

        if self.link_menu_position == CameraSide.Right:
            self.dataset.link_cameras(self.camera, camera)
        else:
            self.dataset.link_cameras(camera, self.camera)

        self.link_menu.setVisible(False)
        self.link_menu_position = CameraSide.Hidden
        self.link_menu.reset_button_states()

    def handle_cam_view_rubber_band_started(self):
        for sw in self.camera_view.stick_widgets:
            sw.setFlag(QGraphicsItem.ItemIsSelectable, True)

    def handle_cam_view_rubber_band_changed(self, rect: QRect, from_scene: QPointF, to_scene: QPointF):
        if rect.isNull():
            return
        pixmap_rect = self.camera_view.mapFromScene(QRectF(from_scene, to_scene)).boundingRect()
        for sw in self.camera_view.stick_widgets:
            sw.set_selected(False)
        selected = list(filter(lambda sw: pixmap_rect.contains(sw.pos()), self.camera_view.stick_widgets))
        if len(selected) == 0:
            self.overlay_gui.enable_delete_sticks_button(False)
            return
        self.overlay_gui.enable_delete_sticks_button(True)
        for sw in selected:
            sw.set_selected(True)

    def handle_delete_sticks_clicked(self):
        for sw in list(filter(lambda sw: sw.is_selected(), self.camera_view.stick_widgets)):
            sw.btn_delete.click_button(artificial_emit=True)
        self.overlay_gui.enable_delete_sticks_button(False)
        self.camera_view.update()

    def handle_overlay_gui_clicked(self):
        if self.link_menu.isVisible():
            self.link_menu.setVisible(False)

    def handle_link_menu_close_requested(self):
        self.link_menu.reset_button_states()
        self.link_menu.setVisible(False)
        self.set_side_camera_state(SideCameraState.Vacant, self.link_menu_position)

    def detect_sticks(self, _: int):
        if self.current_viewed_image is None:
            return
        antar.params = json.loads(self.stick_detection_dialog.paramsText.toPlainText())

        width = self.stick_detection_dialog.spinWidth.value()
        length = self.stick_detection_dialog.spinLength.value()
        hog_th = self.stick_detection_dialog.spinSensitivity.value()
        if length % 2 == 0:
            length = max(3, length - 1)
        p0 = self.stick_detection_dialog.spinP0.value()
        #antar.find_sticks(self.current_viewed_image, hog_th, width, length, p0)

    def update_detection_params(self, _: int):
        antar.params = json.loads(self.stick_detection_dialog.paramsText.toPlainText())

    def update_image_list(self, start: int, end: int):
        idx_from = self.image_list.createIndex(start, 0)
        idx_to = self.image_list.createIndex(end, 2)
        self.image_list.dataChanged.emit(idx_from, idx_to)

    def handle_find_sticks_clicked(self):
        self.camera.remove_sticks()
        gray = cv.cvtColor(self.current_viewed_image, cv.COLOR_BGR2GRAY)
        bgr = self.current_viewed_image
        f = 2
        lines = antar.find_sticks(gray, bgr, equalize=False)
        valid_lines = list(map(lambda line_valid: (np.round(1.0 * line_valid[0]).astype(np.int32), line_valid[2]), filter(lambda line_valid: line_valid[1], lines)))
        sticks: List[Stick] = self.camera.create_new_sticks(valid_lines,
                                                            self.image_list.data(self.ui.image_list.selectionModel().
                                                                                 selectedRows(0)[0], Qt.DisplayRole))
        self.camera.save()

    def handle_stick_length_entered(self):
        self.camera.default_stick_length_cm = int(self.overlay_gui.stick_length_input.get_value())
        self.camera.save()

    def handle_stick_length_clicked(self, btn_info):
        self.stick_widget_in_context = None
        if btn_info['checked']:
            self.overlay_gui.sticks_length_input.set_getter_setter_parser_validator(self.camera.get_default_stick_length,
                                                                                    self.camera.set_default_stick_length,
                                                                                    int,
                                                                                    lambda l: l > 0)
            self.overlay_gui.show_sticks_length_input()
        else:
            self.overlay_gui.hide_sticks_length_input()

    def handle_confirm_sticks_clicked(self, btn_info):
        if self.overlay_gui.edit_sticks_button_pushed():
            self.overlay_gui.toggle_edit_sticks_button()
        self.overlay_gui.enable_measurement_mode_button(btn_info['checked'])
        if btn_info['checked']:
            self.camera.initialize_measurements(False)

    def handle_process_photos_clicked_mp(self, batch_count: int):
        self.processing_should_continue = True
        photos, count = self.camera.get_batch(batch_count, 0)
        #photos = [list(itertools.dropwhile(lambda img_name: img_name != "IMAG5000.JPG", photos))]
        self.processing_started.emit(self, count)
        self.job_counter_lock.lock() # this should not be needed, better safe than sorry
        self.running_jobs = batch_count
        self.job_counter_lock.unlock()
        for i in range(batch_count):
            #self.worker_pool.apply_async(antar.analyze_photos, args=(photos[i], self.camera.folder, self.camera.sticks,
            #                                                         self.camera.stick_to_stick_vectors),
            #                             callback=self.handle_worker_returned)
            self.worker_pool.apply_async(snow.analyze_photos_with_stick_tracking, args=(photos[i], self.camera.folder, self.camera.sticks, self.camera.standard_image_size),
                                         callback=self.handle_worker_returned)
            #self.worker_pool.apply_async(snow.analyze_photos_ip, args=(
            #    photos[i], self.camera.folder, self.camera.sticks, self.camera.standard_image_size),
            #                             callback=self.handle_worker_returned)
        self.result_timer.setSingleShot(False)
        self.result_timer.start()
        self.timer.setInterval(1000)
        self.timer.setSingleShot(False)
        self.timer.timeout.connect(self.process_confirmation_queue)
        self.timer.start()
        self.overlay_gui.hide_process_photos()
        self.overlay_gui.show_process_stop()

    def handle_process_photos_clicked_sp(self):
        # image = self.image_list.data(self.ui.image_list.selectionModel().selectedRows(0)[0])
        photos, count = self.camera.get_batch(1, 0)#[30:]
        photos = photos[0]
        #photos = list(itertools.dropwhile(lambda img_name: img_name != "IMAG1303.JPG", photos))
        sticks = self.camera.sticks
        # if not self.skip:
        #    photos = list(itertools.dropwhile(lambda name: name != "IMAG2930.JPG", photos))
        #    sticks = self.camera.sticks
        # else:
        #    photos = self.rem_photos[1:]
        #    sticks = self.s_sticks
        #result = antar.analyze_photos(photos, self.camera.folder, sticks, self.camera.stick_to_stick_vectors)
        antarstick_processing.analyze_photos_with_stick_tracking(photos, self.camera.folder, self.camera.sticks, self.camera.standard_image_size) #, update_step=4000)
        #snow_processing.analyze_photos_ip(photos, self.camera.folder, self.camera.sticks, self.camera.standard_image_size) #, update_step=4000)
        return
        self.rem_photos = result.remaining_photos
        self.result_timer.setSingleShot(False)
        self.result_timer.start()
        if not self.skip:
            self.timer.setInterval(1000)
            self.timer.setSingleShot(False)
            self.timer.timeout.connect(self.process_confirmation_queue)
            self.timer.start()
        self.return_queue.put_nowait(result)
        # self.camera.insert_measurements2(result.measurements)
        # if result.reason == antar.Reason.SticksMoved:
        #    #self.camera.insert_measurements2({result.current_img:  {'sticks': result.sticks_to_confirm,
        #    #                                                        'image_quality': 0.0}})
        return

    def handle_stick_removed(self):
        if self.camera.stick_count() == 0:
            self.overlay_gui.enable_confirm_sticks_button(False)

    def dispose(self):
        for sw in self.camera_view.stick_widgets:
            sw.prepare_for_deleting()
        self.worker_pool.terminate()
        self.graphics_scene.clear()
        self.graphics_scene.deleteLater()

    def set_side_camera_state(self, state: SideCameraState, side: CameraSide):
        if side == CameraSide.Left:
            add_button = self.left_add_button
            show_button = self.left_show_button
            link = self.left_link
            self.left_side_camera_state = state
        else:
            add_button = self.right_add_button
            show_button = self.right_show_button
            self.right_side_camera_state = state
            link = self.right_link
        if state == SideCameraState.Vacant:
            add_button.set_disabled(False)
            add_button.set_default_state()
            add_button.set_label('Link camera', direction='vertical')
            add_button.setVisible(True)
            add_button.set_button_height(self.camera_view.boundingRect().height())
            show_button.setVisible(False)
        elif state == SideCameraState.Shown:
            add_button.set_disabled(False)
            add_button.setVisible(True)
            add_button.set_button_height(int(0.5 * self.camera_view.boundingRect().height()))
            add_button.set_label('Remove', direction='vertical')
            add_button.set_on(True)
            show_button.setVisible(True)
            show_button.set_label('Hide', direction='vertical')
            self.stick_link_manager_strat.show_links_from_camera(link.camera)
            link.setVisible(True)
            self.recenter_view()
        elif state == SideCameraState.Hidden:
            add_button.setVisible(True)
            add_button.set_button_height(int(0.5 * self.camera_view.boundingRect().height()))
            show_button.setVisible(True)
            show_button.set_label('Show', direction='vertical')
            self.stick_link_manager_strat.hide_links_from_camera(link.camera)
            link.setVisible(False)
            self.recenter_view()
        elif state == SideCameraState.Adding:
            add_button.set_disabled(True)
        elif state == SideCameraState.Synchronizing:
            add_button.set_disabled(True)
            show_button.set_disabled(True)
            link.show_overlay_message(f'synchronizing with {self.camera.folder.name}')
        else:
            add_button.setVisible(False)
            show_button.setVisible(False)
        self.update()

    def handle_side_show_button_clicked(self, btn_state: Dict[str, Any]):
        if btn_state['btn_id'] == 'btn_left_show':
            state = self.left_side_camera_state
            side = CameraSide.Left
        else:
            state = self.right_side_camera_state
            side = CameraSide.Right
        state = SideCameraState.Hidden if state == SideCameraState.Shown else SideCameraState.Shown
        self.set_side_camera_state(state, side)

    def show_hide_side_camera(self, side: CameraSide, state: SideCameraState):
        link = self.left_link if side == CameraSide.Left else self.right_link
        link.setVisible(state == SideCameraState.Shown)
        self.recenter_view()

    def handle_side_button_hovered(self, btn: Dict[str, Any]):
        button = btn['button']
        if button.label.text()[0] == 'S':
            return
        if btn['btn_id'] in [self.BTN_LEFT_ADD, self.BTN_LEFT_SHOW]:
            link = self.left_link
        else:
            link = self.right_link
        if link is None:
            return
        if btn['hovered']:
            if btn['btn_id'] in [self.BTN_LEFT_ADD, self.BTN_RIGHT_ADD]:
                color = QColor(250, 50, 0, 100)
            else:
                color = self.palette().base().color()
                color.setAlpha(128)
            link.highlight(color)
        else:
            link.highlight(None)

    def handle_stick_widget_context_menu(self, sw: StickWidget, view: CameraView):
        self.stick_widget_in_context = sw
        scene_pos = view.mapToScene(sw.pos())
        screen_pos = self.graphics_view.mapFromScene(scene_pos)
        self.overlay_gui.stick_widget_menu.center_buttons()
        self.overlay_gui.show_stick_context_menu_at(screen_pos)

    def handle_set_stick_label_clicked(self, btn_info):
        if self.stick_widget_in_context is None:  # should not happen
            return
        if not btn_info['checked']:
            self.overlay_gui.hide_stick_label_input()
            return
        sw = self.stick_widget_in_context
        self.overlay_gui.stick_label_input.set_getter_setter_parser_validator(sw.get_stick_label, sw.set_stick_label,
                                                                              str,
                                                                              lambda s: self.camera.is_label_available(self.stick_widget_in_context.stick, s))
        self.overlay_gui.show_stick_label_input()

    def handle_set_stick_length_clicked(self, btn_info):
        if self.stick_widget_in_context is None:  # should not happen
            return
        if not btn_info['checked']:
            self.overlay_gui.hide_stick_length_input()
            return
        sw = self.stick_widget_in_context
        self.overlay_gui.stick_length_input.set_getter_setter_parser_validator(sw.get_stick_length_cm,
                                                                               sw.set_stick_length_cm,
                                                                               int,
                                                                               lambda l: l > 0)
        self.overlay_gui.stick_length_input.set_label(f'{sw.get_stick_label()} length:')
        self.overlay_gui.show_stick_length_input()

    def handle_moved_sticks_confirmed(self):
        links = self.link_manager2.stick_links_list
        for link in links:
            stick1 = link.stick1.stick
            stick2 = link.stick2.stick
            stick2.id = stick1.id
            stick2.local_id = stick1.local_id
            stick2.label = stick1.label
            stick2.camera_id = stick1.camera_id
        sticks = list(map(lambda sw: sw.stick, self.split_view.target_widgets))
        res = self.split_view.measurement
        self.graphics_scene.removeItem(self.split_view)
        self.split_view.deleteLater()
        self.split_view = None
        self._set_view_mode('camera')
        self.moved_sticks_linking.reset()
        self.camera.insert_measurements2({res.remaining_photos[0]: {'sticks': sticks, 'image_quality': 1.0,
                                                                    'state': -1,
                                                                    'is_day': True}})
        self.job_counter_lock.lock()
        if self.single_proc:
            self.skip = True
            self.s_sticks = sticks
            self.queue_lock.unlock()
            return
        if self.processing_should_continue:
            stick_to_stick = {}
            for s1 in sticks:
                for s2 in sticks:
                    if s1 == s2:
                        continue
                    s1_to = stick_to_stick.setdefault(s1, {})
                    s1_to[s2] = s1.bottom - s2.bottom
            self.worker_pool.apply_async(antar.analyze_photos, args=(res.remaining_photos[1:], self.camera.folder, sticks,
                                                                     stick_to_stick), callback=self.handle_worker_returned)
            self.paused_jobs -= 1
            self.running_jobs += 1
        else:
            self.paused_jobs -= 1

        if self.running_jobs + self.paused_jobs > 0:
            self.processing_updated.emit(self.camera.processed_photos_count, self.camera.get_photo_count(),
                                         self.running_jobs + self.paused_jobs, not self.processing_should_continue)
        else:
            self.processing_stopped.emit(self)

        self.job_counter_lock.unlock()
        self.queue_lock.unlock()

    def handle_moved_sticks_skipped(self, skip_batch: bool):
        sticks = list(map(lambda sw: sw.stick, self.split_view.source_widgets))
        res: camera_processing.antarstick_processing.Measurement = self.split_view.measurement
        self.graphics_scene.removeItem(self.split_view)
        self.split_view.deleteLater()
        self.camera.skipped_image(res.current_img)
        if skip_batch:
            for img in res.remaining_photos:
                self.camera.skipped_image(img)
        self.split_view = None
        self._set_view_mode('camera')
        self.job_counter_lock.lock()
        if self.processing_should_continue and not skip_batch:
            #stick_to_stick = {}
            #for s1 in sticks:
            #    for s2 in sticks:
            #        s1_to = stick_to_stick.setdefault(s1, {})
            #        s1_to[s2] = s1.bottom - s2.bottom
            self.worker_pool.apply_async(antar.analyze_photos, args=(res.remaining_photos[1:], self.camera.folder, sticks,
                                                                     self.camera.stick_to_stick_vectors), callback=self.handle_worker_returned)
            self.paused_jobs -= 1
            self.running_jobs += 1
        else:
            self.paused_jobs -= 1

        if self.running_jobs + self.paused_jobs > 0:
            self.processing_updated.emit(self.camera.processed_photos_count, self.camera.get_photo_count(),
                                         self.running_jobs + self.paused_jobs, not self.processing_should_continue)
        else:
            self.processing_stopped.emit(self)

        self.job_counter_lock.unlock()
        self.moved_sticks_linking.reset()
        self.queue_lock.unlock()

    def handle_worker_returned(self, result: camera_processing.antarstick_processing.Measurement):
        self.return_queue.put_nowait(result)

    def process_confirmation_queue(self):
        if not self.queue_lock.tryLock(0):
            return
        if self.confirmation_queue.empty():
            self.queue_lock.unlock()
            return
        result = self.confirmation_queue.get_nowait()
        #print(f'need to confirm img {result.current_img}')
        result.camera = self.camera
        self.split_view = SplitView(result)
        self.graphics_scene.addItem(self.split_view)
        self.split_view.initialise()
        self.split_view.confirmed.connect(self.handle_moved_sticks_confirmed)
        self.split_view.skipped.connect(self.handle_moved_sticks_skipped)
        self.split_view.setPos(self.camera_view.pos())
        self.moved_sticks_linking.set_split_view(self.split_view)
        self.link_manager2.set_rect(self.rect_to_view)
        self.moved_sticks_linking.start()
        self._set_view_mode('split')
        self.stick_verification_needed.emit(self)

    def _set_view_mode(self, mode: str):
        self.viewed = mode
        if self.viewed == 'camera':
            self.camera_view.setVisible(True)
            if self.split_view is not None:
                self.split_view.setVisible(False)
        else:
            self.camera_view.setVisible(False)
            self.split_view.setVisible(True)
        self.recenter_view()

    def handle_mes(self):
        gray = cv.cvtColor(self.current_viewed_image, cv.COLOR_BGR2GRAY)
        hsv = cv.cvtColor(self.current_viewed_image, cv.COLOR_BGR2HSV)
        for sw in self.camera_view.stick_widgets:
            antar.estimate_snow_height2(sw.stick, cv.pyrDown(self.current_viewed_image[:, :, 2]))
            #antar.estimate_snow_height(sw.stick, self.current_viewed_image)
            #antar.estimate_height(sw.stick, gray) #self.current_viewed_image)
            sw.set_snow_height(sw.stick.snow_height_px)
        cv.destroyAllWindows()

    def handle_save_measurements(self):
        self.camera.save_measurements()

    def process_result_queue(self):
        if self.return_queue.empty():
            return
        result: camera_processing.antarstick_processing.Measurement = self.return_queue.get_nowait()
        self.camera.insert_measurements2(result.measurements)
        if len(result.measurements) > 0:
            processed = sorted(result.measurements.keys())
            self.image_list.update_items(processed[0], processed[-1])
        self.job_counter_lock.lock()
        if result.reason == camera_processing.antarstick_processing.Reason.FinishedQueue:
            self.running_jobs -= 1
        else:
            if result.reason == camera_processing.antarstick_processing.Reason.Update:
                #print(f'updated at {self.camera.get_processed_count()} / {self.camera.get_photo_count()}')
                if self.processing_should_continue:
                    stick_to_stick = {}
                    for s1 in result.last_valid_sticks:
                        for s2 in result.last_valid_sticks:
                            if s1 == s2:
                                continue
                            s1_to = stick_to_stick.setdefault(s1, {})
                            s1_to[s2] = s1.bottom - s2.bottom
                    #self.worker_pool.apply_async(antar.analyze_photos, args=(result.remaining_photos,
                    #                                                         self.camera.folder,
                    #                                                         result.last_valid_sticks,
                    #                                                         stick_to_stick),
                    #                             callback=self.handle_worker_returned)
                    #self.worker_pool.apply_async(snow.analyze_photos_ip, args=(result.remaining_photos,
                    #                                                         self.camera.folder,
                    #                                                         self.camera.sticks,
                    #                                                         self.camera.standard_image_size),
                    #                             callback=self.handle_worker_returned)
                    self.worker_pool.apply_async(snow.analyze_photos_with_stick_tracking, args=(result.remaining_photos,
                                                                                                self.camera.folder,
                                                                                                self.camera.sticks,
                                                                                                self.camera.standard_image_size),
                                                 callback=self.handle_worker_returned)
                else:
                    self.running_jobs -= 1
            else:
                self.running_jobs -= 1
                self.paused_jobs += 1
                self.confirmation_queue.put_nowait(result)

        if self.running_jobs + self.paused_jobs > 0:
            self.processing_updated.emit(self.camera.processed_photos_count, self.camera.get_photo_count(),
                                         self.running_jobs + self.paused_jobs, not self.processing_should_continue)
        else:
            self.processing_stopped.emit(self)
        self.job_counter_lock.unlock()

    def handle_process_stop_clicked(self):
        self.job_counter_lock.lock()
        self.processing_should_continue = False
        self.processing_updated.emit(self.camera.processed_photos_count, self.camera.get_photo_count(),
                                     self.running_jobs + self.paused_jobs, True)
        self.job_counter_lock.unlock()

    def handle_show_measurements(self, btn):
        for sw in self.camera_view.stick_widgets:
            sw.set_show_measurements(btn['checked'])

    def handle_use_single(self, btn):
        self.single_proc = btn['checked']
        if self.single_proc:
            self.overlay_gui.process_photos_with_jobs_clicked.disconnect(self.handle_process_photos_clicked_mp)
            self.overlay_gui.process_photos_with_jobs_clicked.connect(self.handle_process_photos_clicked_sp)
        else:
            self.overlay_gui.process_photos_with_jobs_clicked.disconnect(self.handle_process_photos_clicked_sp)
            self.overlay_gui.process_photos_with_jobs_clicked.connect(self.handle_process_photos_clicked_mp)

    def __draw_length_hist(self):
        lengths = list(map(lambda s: np.round(s.length_px).astype(np.int32), self.camera.sticks))
        lengths.sort()
        lengths = np.array(lengths)
        print(f'average length is {np.mean(lengths)}')
        print(f'median is {np.median(lengths)}')
        print(f'std is {np.std(lengths)}')
        range = np.max(lengths) - np.min(lengths)
        bins = int(range / 10.0)
        th = fil.threshold_triangle(lengths, bins)

        diff = np.abs(np.diff(lengths))
        big_diff = min(np.argmax(diff), len(lengths) - 1)
        print(f'threshold is {th}')
        plt.hist(lengths, bins=bins)
        plt.savefig("/home/radoslav/hist.png")
        plt.clf()

    def handle_measure_all_clicked(self):
        js = self.dataset.get_json_data()
        return
        sticks = list(map(lambda s: s.copy(), self.camera.sticks))
        for img in self.camera.image_list:
            measurement = self.camera.get_measurements(img, output_sticks=sticks)
            sticks = measurement['sticks']
            image = cv.pyrDown(cv.imread(str(self.camera.folder / img)))
            for stick in sticks:
                antar.estimate_snow_height2(stick, image[:, :, 2])
            self.camera.insert_measurements2({img: {'sticks': sticks, 'image_quality': measurement['image_quality']}})

    @staticmethod
    def extract_timestamps(images: List[str], folder: Path) -> pd.Series:
        timestamps = []
        for img in images:
            with open(folder / img, 'rb') as f:
                tags = exifread.process_file(f, details=False)
            timestamps.append(pd.to_datetime(tags['EXIF DateTimeOriginal'].values, format='%Y:%m:%d %H:%M:%S'))
        return pd.Series(data=timestamps)

    def handle_timestamps_extracted(self, timestamps: pd.Series):
        self.camera.insert_timestamps(timestamps)
        self.camera_view.show_status_message(None)
        if len(self.camera.sticks) > 0:
            self.overlay_gui.enable_confirm_sticks_button(True)

    def handle_synchronize_clicked(self, cam_view: CameraView):
        cam_view.control_widget.set_mode('sync')
        if self.right_link != cam_view and self.right_link is not None:
            self.right_link.control_widget.disable_widget()
        elif self.left_link != cam_view and self.left_link is not None:
            self.left_link.control_widget.disable_widget()
        if self.left_link == cam_view:
            self.left_add_button.set_disabled(True)
            self.left_show_button.set_disabled(True)
        else:
            self.right_add_button.set_disabled(True)
            self.right_show_button.set_disabled(True)


    def handle_previous_photo_clicked(self, cam_view: CameraView):
        if cam_view.control_widget.mode == 'sync':
            curr_photo = cam_view.current_image_name
            prev_photo_id = cam_view.camera.image_names_ids[curr_photo] - 1
            if prev_photo_id >= len(cam_view.camera.image_list):
                return
            prev_photo = cam_view.camera.image_list[prev_photo_id]
            self.display_image(prev_photo, cam_view)
        else:
            if cam_view == self.right_link:
                image = self.sync[CameraSide.Right].get_reciprocal_image_by_name(cam_view.camera,
                                                                                 cam_view.current_image_name)
            elif cam_view == self.left_link:
                image = self.sync[CameraSide.Left].get_reciprocal_image_by_name(cam_view.camera,
                                                                                cam_view.current_image_name)
            else:
                image = (self.camera_view.current_image_name, None)
            if image is None:
                self.display_image(cam_view.current_image_name, cam_view)
            else:
                image_id = self.camera.image_names_ids[image[0]] - 1
                if image_id >= 0:
                    index = self.image_list.index(image_id, 0)
                    self.ui.image_list.setCurrentIndex(index)

    def handle_next_photo_clicked(self, cam_view: CameraView):
        if cam_view.control_widget.mode == 'sync':
            curr_photo = cam_view.current_image_name
            next_photo_id = cam_view.camera.image_names_ids[curr_photo] + 1
            if next_photo_id >= len(cam_view.camera.image_list):
                return
            next_photo = cam_view.camera.image_list[next_photo_id]
            self.display_image(next_photo, cam_view)
        else:
            if cam_view == self.right_link:
                image = self.sync[CameraSide.Right].get_reciprocal_image_by_name(cam_view.camera,
                                                                                 cam_view.current_image_name)
            elif cam_view == self.left_link:
                image = self.sync[CameraSide.Left].get_reciprocal_image_by_name(cam_view.camera,
                                                                                cam_view.current_image_name)
            else:
                image = (self.camera_view.current_image_name, None)
            if image is None:
                self.display_image(cam_view.current_image_name, cam_view)
            else:
                image_id = self.camera.image_names_ids[image[0]] + 1
                if image_id < len(self.camera.image_list):
                    index = self.image_list.index(image_id, 0)
                    self.ui.image_list.setCurrentIndex(index)

    def display_image(self, img_name: str, cam_view: CameraView):
        img = cv.imread(str(cam_view.camera.folder / img_name))
        img = self.standardized_image(img, cam_view.camera)
        cam_view.set_image(img, img_name)

    def handle_sync_confirm_clicked(self, cam_view: CameraView):
        img_id = self.camera.image_names_ids[self.camera_view.current_image_name]
        this_dt = self.camera.measurements.iloc[img_id]['date_time']
        if cam_view == self.left_link:
            other_id = self.left_link.camera.image_names_ids[self.left_link.current_image_name]
            other_dt = self.left_link.camera.measurements.iloc[other_id]['date_time']
            #print(f'trying to sync {str(this_dt)} to {str(other_dt)}')
            #self.set_side_camera_state(SideCameraState.Synchronizing, LinkMenuPosition.LEFT)
            self.sync[CameraSide.Left].synchronize(other_dt, this_dt)
            sync = self.sync[CameraSide.Left]
        else:
            other_id = self.right_link.camera.image_names_ids[self.right_link.current_image_name]
            other_dt = self.right_link.camera.measurements.iloc[other_id]['date_time']
            #print(f'trying to sync {str(this_dt)} to {str(other_dt)}')
            #self.set_side_camera_state(SideCameraState.Synchronizing, LinkMenuPosition.RIGHT)
            self.sync[CameraSide.Right].synchronize(this_dt, other_dt)
            sync = self.sync[CameraSide.Right]
        cam_view.control_widget.set_mode('view')

        self.left_add_button.set_disabled(False)
        self.left_show_button.set_disabled(False)
        self.right_add_button.set_disabled(False)
        self.right_show_button.set_disabled(False)
        #self.handle_synchronization_finished(sync)

    def handle_sync_cancel_clicked(self, cam_view: CameraView):
        if cam_view == self.left_link:
            rec_image_name = self.sync[CameraSide.Left].get_reciprocal_image_by_name(self.camera,
                                                                                     self.camera_view.current_image_name)
        else:
            rec_image_name = self.sync[CameraSide.Right].get_reciprocal_image_by_name(self.camera,
                                                                                      self.camera_view.current_image_name)
        if rec_image_name is not None:
            rec_image_name = rec_image_name[0]
            self.display_image(rec_image_name, cam_view)
        if self.right_link is not None:
            self.right_link.control_widget.enable_widget()
            self.right_link.control_widget.set_mode('view')
        if self.left_link is not None:
            self.left_link.control_widget.enable_widget()
            self.left_link.control_widget.set_mode('view')
        self.left_add_button.set_disabled(False)
        self.left_show_button.set_disabled(False)
        self.right_add_button.set_disabled(False)
        self.right_show_button.set_disabled(False)

    def handle_camera_temporal_non_increasigness_found(self, point: Tuple[str, pd.Timestamp, str, pd.Timestamp, pd.Timestamp]):
        last_correct_img = point[0]
        last_correct_date = point[1]
        problem_img = point[2]
        problem_date = point[3]
        proposed_date = point[4]
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle(f'Camera {self.camera.folder.name} - Temporal inconsistency')
        msg.setText('The sequence of images in the camera is not temporally monotonically increasing.')
        msg.setInformativeText(f'The image {last_correct_img} has timestamp {str(last_correct_date)} while {problem_img} has timestamp {str(problem_date)}\nDo you want to set the timestamp for the image {problem_img} to {str(proposed_date)} and adjust all subsequent images accordingly?')
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        ret = msg.exec()
        if ret == QMessageBox.Yes:
            self.camera.repair_timestamps()

    def handle_first_photo_clicked(self, cam_view: CameraView):
        if cam_view == self.camera_view:
            index = self.image_list.index(0, 0)
            self.ui.image_list.setCurrentIndex(index)
        else:
            if cam_view.control_widget.mode == 'sync':
                self.display_image(cam_view.camera.image_list[0], cam_view)

    def handle_synchronization_finished(self, sync: CameraSynchronization):
        if self.camera != sync.left_camera and self.camera != sync.right_camera:
            return
        timestamp = sync.left_timestamp if self.camera == sync.left_camera else sync.right_timestamp
        img_name = self.camera.measurements.loc[timestamp, 'image_name']
        img_id = self.camera.image_names_ids[img_name]
        index = self.image_list.index(img_id, 0)
        self.ui.image_list.setCurrentIndex(index)

    def handle_exclude_photos_clicked(self):
        sel = self.ui.image_list.selectionModel().selection()
        for index in sel.indexes():
            self.camera.measurements.iat[index.row(), PD_IMAGE_STATE] = PhotoState.Skipped
        first = self.camera.image_list[sel.indexes()[0].row()]
        last = self.camera.image_list[sel.indexes()[-1].row()]
        self.image_list.update_items(first, last)
        self.overlay_gui.hide_exclude_button()
        self.overlay_gui.show_include_button()

    def handle_include_photos_clicked(self):
        sel = self.ui.image_list.selectionModel().selection()
        for index in sel.indexes():
            self.camera.measurements.iat[index.row(), PD_IMAGE_STATE] = PhotoState.Unprocessed
        first = self.camera.image_list[sel.indexes()[0].row()]
        last = self.camera.image_list[sel.indexes()[-1].row()]
        self.image_list.update_items(first, last)
        self.overlay_gui.show_exclude_button()
        self.overlay_gui.hide_include_button()

    def handle_measurement_mode_toggled(self, shown: bool):
        mode = StickMode.Measurement if shown else StickMode.Display
        for sw in self.camera_view.stick_widgets:
            sw.set_mode(mode)

    def handle_image_list_slider_released(self):
        first_idx = self.ui.image_list.indexAt(QPoint(0, 0))
        last_idx = self.ui.image_list.indexAt(self.ui.image_list.viewport().rect().bottomLeft())
        self.image_list.handle_slider_released(first_idx, last_idx)
