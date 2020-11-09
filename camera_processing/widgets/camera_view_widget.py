import json
import time
from enum import IntEnum
from multiprocessing import Pool
from queue import Queue
from typing import List, Dict, Optional, Any

import cv2 as cv
import numpy as np
from PyQt5 import QtWidgets
from PyQt5.QtCore import (QMarginsF, QModelIndex, QPointF, QRectF, Qt,
                          pyqtSignal, QByteArray, QRect)
from PyQt5.QtCore import pyqtSlot as Slot, QTimer
from PyQt5.QtGui import QImage, QPixmap, QPainter, QBrush, QColor, QFont, QPen
from PyQt5.QtWidgets import QGraphicsScene, QGraphicsItem, QSizePolicy, QAbstractScrollArea
from pandas import DataFrame

import camera_processing.antarstick_processing as antar
from camera import Camera
from camera_processing.antarstick_processing import process_batch
from camera_processing.widgets import ui_camera_view
from camera_processing.widgets.button import Button, ButtonColor
from camera_processing.widgets.button_menu import ButtonMenu
from camera_processing.widgets.cam_graphics_view import CamGraphicsView
from camera_processing.widgets.camera_view import CameraView
from camera_processing.widgets.overlay_gui import OverlayGui
from camera_processing.widgets.stick_link_manager import StickLinkManager
from camera_processing.widgets.stick_widget import StickMode, StickWidget
from dataset import Dataset
from image_list_model import ImageListModel
from stick import Stick
from stick_detection_dialog import StickDetectionDialog


class LinkMenuPosition(IntEnum):
    HIDDEN = 0,
    LEFT = 1,
    RIGHT = 2,


class SideCameraState(IntEnum):
    Vacant = 0,
    Adding = 1,
    Shown = 2,
    Hidden = 3,
    Unavailable = 4,  # no other camera available for linking


class CameraViewWidget(QtWidgets.QWidget):

    sticks_changed = pyqtSignal()
    initialization_done = pyqtSignal(Camera)

    BTN_LEFT_ADD = 'btn_left_add'
    BTN_LEFT_SHOW = 'btn_left_show'
    BTN_RIGHT_ADD = 'btn_right_add'
    BTN_RIGHT_SHOW = 'btn_right_show'

    def __init__(self, dataset: Dataset):
        QtWidgets.QWidget.__init__(self)

        self.ui = ui_camera_view.Ui_CameraView()
        self.ui.setupUi(self)

        self.image_list = ImageListModel()
        self.ui.image_list.setHorizontalHeader(None)
        self.ui.image_list.setVerticalHeader(None)
        self.ui.image_list.setModel(self.image_list)
        self.ui.image_list.selectionModel().currentChanged.connect(self.handle_list_model_current_changed)
        self.ui.image_list.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)
        self.ui.image_list.setEnabled(False)

        self.dataset = dataset
        self.dataset.cameras_linked.connect(self.handle_cameras_linked)
        self.dataset.cameras_unlinked.connect(self.handle_cameras_unlinked)
        self.dataset.camera_removed.connect(self.handle_camera_removed)
        self.camera: Optional[Camera] = None
        self.graphics_scene = QGraphicsScene()

        self.stick_link_manager = StickLinkManager(self.dataset, self.camera)
        self.graphics_scene.addItem(self.stick_link_manager)
        self.stick_link_manager.setZValue(2)
        self.stick_link_manager.setVisible(False)

        self.graphics_view = CamGraphicsView(self.stick_link_manager, self)
        self.graphics_view.setSizePolicy(QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding))

        self.ui.graphicsViewLayout.addWidget(self.graphics_view)

        self.graphics_view.setScene(self.graphics_scene)
        self.graphics_view.rubberBandChanged.connect(self.handle_cam_view_rubber_band_changed)
        self.graphics_view.rubber_band_started.connect(self.handle_cam_view_rubber_band_started)

        self.current_viewed_image: Optional[np.ndarray] = None
        self.scaling = 2.0
        self.camera_view = CameraView(self.dataset, self.scaling)
        self.camera_view.setAcceptHoverEvents(False)
        self.camera_view.setZValue(1)


        self.graphics_scene.addItem(self.camera_view)

        self.stick_widgets: List[StickWidget] = []
        self.detected_sticks: List[Stick] = []
        self.link_menus = dict({'right': None, 'left': None})
        self.left_link: Optional[CameraView] = None
        self.right_link: Optional[CameraView] = None

        self.overlay_gui = OverlayGui(self.graphics_view)
        self.overlay_gui.reset_view_requested.connect(self._recenter_view)
        self.overlay_gui.edit_sticks_clicked.connect(self.handle_edit_sticks_clicked)
        self.overlay_gui.link_sticks_clicked.connect(self.handle_link_sticks_clicked)
        self.overlay_gui.delete_sticks_clicked.connect(self.handle_delete_sticks_clicked)
        self.overlay_gui.process_photos_clicked.connect(self.handle_process_photos_clicked2)
        self.overlay_gui.clicked.connect(self.handle_overlay_gui_clicked)
        self.overlay_gui.find_sticks_clicked.connect(self.handle_find_sticks_clicked)
        self.detect_thin_sticks = False
        self.overlay_gui.stick_length_input.input_entered.connect(self.handle_stick_length_entered)
        self.overlay_gui.sticks_length_clicked.connect(self.handle_stick_length_clicked)
        self.overlay_gui.confirm_sticks_clicked.connect(self.handle_confirm_sticks_clicked)

        self.graphics_scene.addItem(self.overlay_gui)
        self.overlay_gui.initialize()

        self.link_menu = ButtonMenu(self.scaling, self.overlay_gui)
        self.link_menu.show_close_button(True)
        self.link_menu_position: LinkMenuPosition = LinkMenuPosition.HIDDEN
        self.link_menu.setZValue(100)
        self.link_menu.setVisible(False)
        self.link_menu.set_layout_direction("vertical")
        self.link_menu.close_requested.connect(self.handle_link_menu_close_requested)
        self.link_menu.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)

        self.links_available = False

        self.return_queue = Queue()
        self.worker_pool = Pool(processes=2)
        self.image_loading_time: float = -1.0

        self.photo_count_to_process: int = 0
        self.photo_count_processed: int = 0
        self.next_batch_start: int = 0
        self.photo_batch: List[str] = []
        self.timer: QTimer = QTimer()

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
        self.set_side_camera_state(SideCameraState.Unavailable, LinkMenuPosition.LEFT)
        self.set_side_camera_state(SideCameraState.Unavailable, LinkMenuPosition.RIGHT)

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
        self.camera.sticks_added.connect(lambda: self.overlay_gui.enable_confirm_sticks_button(True))
        self.camera.sticks_removed.connect(self.handle_stick_removed)
        self.camera.stick_removed.connect(self.handle_stick_removed)
        if self.camera.rep_image is None:
            self.camera.rep_image = cv.resize(cv.imread(str(self.camera.folder / self.camera.rep_image_path)), (0, 0), fx=0.5, fy=0.5)
        self.stick_link_manager.camera = self.camera
        self.stick_link_manager.update_links()
        self.image_list.initialize(self.camera, self.camera.get_processed_count())
        self.ui.image_list.setEnabled(True)
        self.ui.image_list.setModel(self.image_list)
        self.ui.image_list.resizeColumnToContents(0)
        self.ui.image_list.resizeColumnToContents(1)
        self.ui.image_list.resizeColumnToContents(2)
        self.initialize_rest_of_gui()
        select_index = self.image_list.index(0, 0)
        self.ui.image_list.setCurrentIndex(select_index)
        self.overlay_gui.stick_length_input.set_length(self.camera.default_stick_length_cm)
        if len(self.camera.sticks) == 0 and False:
            self.handle_find_sticks_clicked()

    def initialize_rest_of_gui(self):
        viewport_rect = self.graphics_view.viewport().rect()
        _re = self.graphics_view.mapToScene(viewport_rect)
        self.graphics_scene.setSceneRect(QRectF(_re.boundingRect()))

        self.camera_view.initialise_with(self.camera)

        rect = self.camera_view.stick_length_input.boundingRect()
        self.camera_view.stick_length_input.setParentItem(self.overlay_gui)
        self.camera_view.stick_length_input.setPos(viewport_rect.width() * 0.5 - rect.width() * 0.5,
                                                   viewport_rect.height() * 0.5 - rect.height() * 0.5)

        self.camera_view.set_show_title(False)
        self.left_add_button.set_button_height(self.camera_view.boundingRect().height())
        self.left_add_button.setPos(self.camera_view.get_top_left() -
                                    QPointF(self.left_add_button.boundingRect().width(), 0))

        self.right_add_button.set_button_height(self.camera_view.boundingRect().height())
        self.right_add_button.setPos(self.camera_view.get_top_right())

        self.left_show_button.set_button_height(int(0.5 * self.camera_view.boundingRect().height()))
        self.left_show_button.setPos(self.left_add_button.pos() +
                                     QPointF(0, 0.5 * self.camera_view.boundingRect().height()))

        self.right_show_button.set_button_height(int(0.5 * self.camera_view.boundingRect().height()))
        self.right_show_button.setPos(self.right_add_button.pos() +
                                      QPointF(0, 0.5 * self.camera_view.boundingRect().height()))

        self.stick_link_manager.primary_camera = self.camera_view

        self.initialize_link_menu()
        self.overlay_gui.show_loading_screen(False)
        self.overlay_gui.process_photos_count_clicked.connect(self.handle_process_photos_clicked)
        self.overlay_gui.initialize_process_photos_popup(self.camera.get_photo_count(), self.image_loading_time)
        self.overlay_gui.handle_cam_view_changed()
        self.initialization_done.emit(self.camera)
        self._recenter_view()
        self.graphics_view.view_changed.emit()

    @Slot(bool)
    def link_cameras_enabled(self, enabled: bool):
        if enabled:
            if self.left_side_camera_state == SideCameraState.Unavailable:
                self.set_side_camera_state(SideCameraState.Vacant, LinkMenuPosition.LEFT)
            if self.right_side_camera_state == SideCameraState.Unavailable:
                self.set_side_camera_state(SideCameraState.Vacant, LinkMenuPosition.RIGHT)
        else:
            self.set_side_camera_state(SideCameraState.Unavailable, LinkMenuPosition.LEFT)
            self.set_side_camera_state(SideCameraState.Unavailable, LinkMenuPosition.RIGHT)
        self.links_available = enabled

    def _recenter_view(self):
        rect_to_view = self.camera_view.sceneBoundingRect().united(self.left_add_button.sceneBoundingRect())
        rect_to_view = rect_to_view.united(self.right_add_button.sceneBoundingRect())

        if self.left_link is not None and self.left_side_camera_state == SideCameraState.Shown:
            rect_to_view = rect_to_view.united(self.left_link.sceneBoundingRect())
        if self.right_link is not None and self.right_side_camera_state == SideCameraState.Shown:
            rect_to_view = rect_to_view.united(self.right_link.sceneBoundingRect())

        self.graphics_scene.setSceneRect(rect_to_view.marginsAdded(QMarginsF(50, 50, 50, 50)))

        self.graphics_view.fitInView(rect_to_view, Qt.KeepAspectRatio)
        self.graphics_scene.update(rect_to_view)
        self.link_menu.set_layout_direction('vertical')

    @Slot(QModelIndex, QModelIndex)
    def handle_list_model_current_changed(self, current: QModelIndex, previous: QModelIndex):
        image_path = self.image_list.data(current, Qt.UserRole)
        self.current_viewed_image = cv.pyrDown(cv.imread(str(image_path)))
        self.camera_view.set_image(self.current_viewed_image)
        sticks = self.camera.get_sticks_in_image(image_path.name)

        #measurements = self.camera.get_measurement_for(image_path.name)
        measurements = self.camera.get_sticks_in_image(image_path.name)

        # The selected photo is not processed yet, therefore set "missing measurement" value for each StickWidget
        if measurements is None:
            for sw in self.camera_view.stick_widgets:
                sw.set_snow_height(-1)
            return

        for sw in self.camera_view.stick_widgets:
            #m_ = measurements[sw.stick.label]
            #sw.set_snow_height(m_['snow_height'])
            for st in measurements:
                if st.label == sw.stick.label:
                    sw.set_snow_height(st.snow_height_px)

    @Slot()
    def handle_edit_sticks_clicked(self):
        edit_sticks_on = self.overlay_gui.edit_sticks_button_pushed()
        if edit_sticks_on:
            self.camera_view.set_stick_widgets_mode(StickMode.EDIT)
        else:
            self.camera_view.set_stick_widgets_mode(StickMode.DISPLAY)
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
        self.left_add_button.setVisible(not self.overlay_gui.link_sticks_button_pushed())
        self.right_add_button.setVisible(not self.overlay_gui.link_sticks_button_pushed())
        if self.overlay_gui.link_sticks_button_pushed():
            self.stick_link_manager.start()
        else:
            self.stick_link_manager.stop()

    def sync_stick_link_manager(self):
        cams = []
        if self.left_link is not None:
            cams.append(self.left_link)
        if self.right_link is not None:
            cams.append(self.right_link)

        self.stick_link_manager.set_secondary_cameras(cams)
    
    def _destroy(self):
        self.graphics_scene.removeItem(self.stick_link_manager)

    def handle_cameras_linked(self, cam1: Camera, cam2: Camera):
        if cam1.id != self.camera.id and cam2.id != self.camera.id:
            return
        other_camera_view = CameraView(self.dataset, self.scaling)
        self.graphics_scene.addItem(other_camera_view)
        other_camera_view.setAcceptHoverEvents(False)

        other_camera_view.set_display_mode()
        other_camera_view.set_show_stick_widgets(True)
        cam_position = 'right'

        if self.camera.id == cam1.id:  # self.camera is on the left
            cam = cam2
        else:  # self.camera is on the right
            cam = cam1
            cam_position = 'left'

        self.link_menu.hide_button(str(cam.folder))
        if self.link_menu.visible_buttons_count() == 0:
            if self.left_side_camera_state == SideCameraState.Vacant:
                self.set_side_camera_state(SideCameraState.Unavailable, LinkMenuPosition.LEFT)
            if self.right_side_camera_state == SideCameraState.Vacant:
                self.set_side_camera_state(SideCameraState.Unavailable, LinkMenuPosition.RIGHT)
        other_camera_view.initialise_with(cam)

        pos: QPointF = self.camera_view.pos()
        if cam_position == 'left':
            pos = self.left_add_button.scenePos() - QPointF(self.camera_view.boundingRect().width(), 0)
            if self.left_link is not None:
                self.dataset.unlink_cameras(self.camera, self.left_link.camera)
            self.set_side_camera_state(SideCameraState.Shown, LinkMenuPosition.LEFT)
            self.left_link = other_camera_view
        else:
            pos = self.right_add_button.scenePos() + QPointF(self.right_add_button.boundingRect().width(), 0)
            if self.right_link is not None:
                self.dataset.unlink_cameras(self.camera, self.right_link.camera)
            self.set_side_camera_state(SideCameraState.Shown, LinkMenuPosition.RIGHT)
            self.right_link = other_camera_view

        other_camera_view.setPos(pos)
        other_camera_view.stick_link_requested.connect(self.stick_link_manager.handle_stick_widget_link_requested)

        self._recenter_view()

        self.sync_stick_link_manager()
        self.overlay_gui.enable_link_sticks_button(True)

    def handle_cameras_unlinked(self, cam1: Camera, cam2: Camera):
        if cam1.id != self.camera.id and cam2.id != self.camera.id:
            return
        to_remove = cam1 if cam2.id == self.camera.id else cam2
        self.link_menu.show_button(str(to_remove.folder))

        if self.left_link is not None and self.left_link.camera.id == to_remove.id:
            self.left_link.stick_link_requested.disconnect(self.stick_link_manager.handle_stick_widget_link_requested)
            self.left_link.setParentItem(None)
            self.graphics_scene.removeItem(self.left_link)
            self.left_link = None
            self.set_side_camera_state(SideCameraState.Vacant, LinkMenuPosition.LEFT)
            #self.left_add_button.set_default_state()
            #self.left_add_button.set_label('Link camera', direction='vertical')
            #self.left_add_button.set_tooltip('Link camera')
        elif self.right_link is not None and self.right_link.camera.id == to_remove.id:
            self.right_link.stick_link_requested.disconnect(self.stick_link_manager.handle_stick_widget_link_requested)
            self.right_link.setParentItem(None)
            self.graphics_scene.removeItem(self.right_link)
            self.right_link = None
            self.set_side_camera_state(SideCameraState.Vacant, LinkMenuPosition.RIGHT)
        if self.left_link is None and self.right_link is None:
            self.overlay_gui.enable_link_sticks_button(False)

        if self.link_menu.visible_buttons_count() > 0:
            if self.left_side_camera_state == SideCameraState.Unavailable:
                self.set_side_camera_state(SideCameraState.Vacant, LinkMenuPosition.LEFT)
            if self.right_side_camera_state == SideCameraState.Unavailable:
                self.set_side_camera_state(SideCameraState.Vacant, LinkMenuPosition.RIGHT)

        self._recenter_view()

    def handle_link_camera_button_left_clicked(self, data: Dict[str, Any]):
        self.handle_link_camera_button_clicked(LinkMenuPosition.LEFT, data['button'].is_on())

    def handle_link_camera_button_right_clicked(self, data: Dict[str, Any]):
        self.handle_link_camera_button_clicked(LinkMenuPosition.RIGHT, data['button'].is_on())

    def handle_link_camera_button_clicked(self, button_position: LinkMenuPosition, is_pushed: bool):
        if not is_pushed:
            self.dataset.unlink_cameras(self.camera,
                                        self.left_link.camera if button_position == LinkMenuPosition.LEFT
                                        else self.right_link.camera)
            return
        self.link_menu.center_buttons()
        #if self.link_menu_position == LinkMenuPosition.RIGHT:
        #    self.right_add_button.set_default_state()
        #elif self.link_menu_position == LinkMenuPosition.LEFT:
        #    self.left_add_button.set_default_state()
        if self.link_menu_position != LinkMenuPosition.HIDDEN:
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

        if self.link_menu_position == LinkMenuPosition.RIGHT:
            self.dataset.link_cameras(self.camera, camera)
        else:
            self.dataset.link_cameras(camera, self.camera)

        self.link_menu.setVisible(False)
        self.link_menu_position = LinkMenuPosition.HIDDEN
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

    def handle_process_photos_clicked__(self):
        self.worker_pool.apply_async(process_batch, args=(self.camera.get_batch(count=100), self.camera.folder, self.camera.sticks), callback=self.handle_worker_finished)
        #df = process_batch(self.camera.get_batch(count=10), self.camera.folder, self.camera.sticks)

    def handle_worker_finished(self, df: DataFrame):
        self.camera.insert_measurements(df)
        self.image_list.set_processed_count(self.camera.get_processed_count())
        self.photo_count_processed += df.shape[0]

        self.camera_view.set_progress_bar_progress(self.photo_count_processed, self.photo_count_to_process)
        self.camera_view.set_status_text(f'processed {self.photo_count_processed} / {self.photo_count_to_process}', 0)

        if self.next_batch_start < self.photo_count_to_process:
            mini_batch = min(50, self.photo_count_to_process - self.next_batch_start)
            self.worker_pool.apply_async(process_batch, args=(
                self.photo_batch[self.next_batch_start:self.next_batch_start + mini_batch], self.camera.folder, self.camera.sticks),
                                         callback=self.handle_worker_finished)
            self.next_batch_start += mini_batch

        if self.photo_count_processed >= self.photo_count_to_process:
            self.camera_view.set_status_text(f'complete {self.photo_count_processed} / {self.photo_count_to_process}', 2000)
            self.photo_count_to_process = 0
            self.photo_batch = []
            self.next_batch_start = 0
            self.overlay_gui.enable_process_photos_button(True)
            #self.gpixmap.clear_status_progress()

    def handle_process_photos_clicked(self, count: int = -1):
        return
        self.photo_count_to_process = min(count, self.camera.get_photo_count() - self.camera.get_processed_count())
        self.photo_batch = self.camera.get_batch(self.photo_count_to_process)
        self.camera_view.set_status_text(f'processed {self.photo_count_processed} / {self.photo_count_to_process}', 0)
        if self.photo_count_to_process <= 100:
            mini_batch = min(50, self.photo_count_to_process)
            self.next_batch_start = mini_batch
            self.worker_pool.apply_async(process_batch, args=(
                self.photo_batch[:mini_batch], self.camera.folder, self.camera.sticks),
                                         callback=self.handle_worker_finished)
        else:
            mini_batch = 50
            self.worker_pool.apply_async(process_batch, args=(
                self.photo_batch[:mini_batch], self.camera.folder, self.camera.sticks),
                                         callback=self.handle_worker_finished)

            self.next_batch_start = mini_batch
            self.worker_pool.apply_async(process_batch, args=(
                self.photo_batch[self.next_batch_start:self.next_batch_start + mini_batch], self.camera.folder, self.camera.sticks),
                                         callback=self.handle_worker_finished)
            self.next_batch_start += mini_batch
        #self.worker_pool.apply_async(process_batch, args=(self.camera.get_batch(count=count), self.camera.folder, self.camera.sticks), callback=self.handle_worker_finished)

    def handle_overlay_gui_clicked(self):
        if self.link_menu.isVisible():
            self.link_menu.setVisible(False)

    def handle_link_menu_close_requested(self):
        self.link_menu.reset_button_states()
        self.link_menu.setVisible(False)
        #if self.link_menu_position == LinkMenuPosition.LEFT:
        #    #self.left_add_button.set_disabled(False)
        #    #self.left_add_button.set_default_state()
        #    #self.left_add_button.setVisible(True)
        #else:
        #    self.right_add_button.set_disabled(False)
        #    self.right_add_button.set_default_state()
        #    #self.right_add_button.setVisible(True)
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
        #start = time.time()
        self.camera.remove_sticks()
        gray = cv.cvtColor(self.current_viewed_image, cv.COLOR_BGR2GRAY)
        bgr = self.current_viewed_image
        f = 2
        lines = antar.find_sticks(gray, bgr, equalize=True)
        start_valid = time.time()
        valid_lines = list(map(lambda line_valid: ((1.0 * line_valid[0]).astype(np.int32), line_valid[2]), filter(lambda line_valid: line_valid[1], lines)))
        sticks: List[Stick] = self.camera.create_new_sticks(valid_lines,
                                                            self.image_list.data(self.ui.image_list.selectionModel().
                                                                                 selectedRows(0)[0], Qt.DisplayRole))
        camera_start = time.time()
        self.camera.save()
        #print(f'handle took {time.time() - start} secs')

    def handle_stick_length_entered(self):
        self.camera.default_stick_length_cm = self.overlay_gui.stick_length_input.get_length()
        self.camera.save()

    def handle_stick_length_clicked(self):
        self.overlay_gui.stick_length_input.set_length(self.camera.default_stick_length_cm)

    def handle_confirm_sticks_clicked(self):
        self.camera.initialize_measurements(False)

    def handle_process_photos_clicked2(self):
        image = self.image_list.data(self.ui.image_list.selectionModel().selectedRows(0)[0])
        photos = [image]#self.camera.get_batch(100)
        if False:
            results = antar.process_photos_([image], self.camera.folder, self.camera.sticks)
            result = results[image]
            for sw in self.camera_view.stick_widgets:
                if result[sw.stick]:
                    sw.border_positive()
                    self.camera.stick_changed.emit(sw.stick)
                else:
                    sw.border_negative()
            return
        image_sticks = antar.process_photos(photos, self.camera.folder, self.camera.sticks)
        self.camera.insert_measurements2(image_sticks)
        measurements = self.camera.get_sticks_in_image(image)

        # The selected photo is not processed yet, therefore set "missing measurement" value for each StickWidget
        if measurements is None:
            for sw in self.camera_view.stick_widgets:
                sw.set_snow_height(-1)
            return

        for sw in self.camera_view.stick_widgets:
            #m_ = measurements[sw.stick.label]
            #sw.set_snow_height(m_['snow_height'])
            for st in measurements:
                if st.label == sw.stick.label:
                    sw.set_snow_height(st.snow_height_px)

    def handle_stick_removed(self):
        if self.camera.stick_count() == 0:
            self.overlay_gui.enable_confirm_sticks_button(False)

    def dispose(self):
        self.graphics_scene.clear()
        self.graphics_scene.deleteLater()

    def set_side_camera_state(self, state: SideCameraState, side: LinkMenuPosition):
        if side == LinkMenuPosition.LEFT:
            add_button = self.left_add_button
            show_button = self.left_show_button
            self.left_side_camera_state = state
        else:
            add_button = self.right_add_button
            show_button = self.right_show_button
            self.right_side_camera_state = state
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
        elif state == SideCameraState.Hidden:
            add_button.setVisible(True)
            add_button.set_button_height(int(0.5 * self.camera_view.boundingRect().height()))
            show_button.setVisible(True)
            show_button.set_label('Show', direction='vertical')
        elif state == SideCameraState.Adding:
            add_button.set_disabled(True)
        else:
            add_button.setVisible(False)
            show_button.setVisible(False)
        self.update()

    def handle_side_show_button_clicked(self, btn_state: Dict[str, Any]):
        if btn_state['btn_id'] == 'btn_left_show':
            state = self.left_side_camera_state
            side = LinkMenuPosition.LEFT
        else:
            state = self.right_side_camera_state
            side = LinkMenuPosition.RIGHT
        state = SideCameraState.Hidden if state == SideCameraState.Shown else SideCameraState.Shown
        self.set_side_camera_state(state, side)
        self.show_hide_side_camera(side, state)

    def show_hide_side_camera(self, side: LinkMenuPosition, state: SideCameraState):
        link = self.left_link if side == LinkMenuPosition.LEFT else self.right_link
        link.setVisible(state == SideCameraState.Shown)
        self._recenter_view()

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
