import os
import sys
from pathlib import Path
from typing import Dict, Any, List

from PyQt5.Qt import QBrush, QColor, QPen
from PyQt5.QtCore import QPoint, QRectF, Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QPainter, QPixmap
from PyQt5.QtWidgets import QGraphicsItem, QGraphicsObject

from camera_processing.widgets.button import ButtonColor
from camera_processing.widgets.button_menu import ButtonMenu
from camera_processing.widgets.cam_graphics_view import CamGraphicsView
from camera_processing.widgets.stick_length_input import TextInputWidget


class OverlayGui(QGraphicsObject):

    reset_view_requested = pyqtSignal()
    edit_sticks_clicked = pyqtSignal()
    link_sticks_clicked = pyqtSignal()
    delete_sticks_clicked = pyqtSignal()
    process_photos_clicked = pyqtSignal('PyQt_PyObject')
    process_photos_with_jobs_clicked = pyqtSignal(int)
    clicked = pyqtSignal()
    find_sticks_clicked = pyqtSignal()
    detect_thin_sticks_set = pyqtSignal('PyQt_PyObject')
    sticks_length_clicked = pyqtSignal('PyQt_PyObject')
    confirm_sticks_clicked = pyqtSignal('PyQt_PyObject')
    set_stick_label_clicked = pyqtSignal('PyQt_PyObject')
    set_stick_length_clicked = pyqtSignal('PyQt_PyObject')
    process_stop_clicked = pyqtSignal()
    save_measurements = pyqtSignal()
    show_measurements = pyqtSignal('PyQt_PyObject')
    exclude_photos_no_snow = pyqtSignal('PyQt_PyObject')
    exclude_photos_bad_quality = pyqtSignal('PyQt_PyObject')
    include_photos = pyqtSignal()
    vertical_offset_clicked = pyqtSignal()
    process_from_this_and_next = pyqtSignal()
    measurement_mode_toggle = pyqtSignal(bool)
    reset_measurements_clicked = pyqtSignal()
    low_quality_clicked = pyqtSignal()

    def __init__(self, view: CamGraphicsView, parent: QGraphicsItem = None):
        QGraphicsObject.__init__(self, parent)

        self.view = view
        self.view.view_changed.connect(self.handle_cam_view_changed)
        self.setZValue(43)
        self.setAcceptHoverEvents(False)
        self.setAcceptedMouseButtons(Qt.AllButtons)
        self.top_menu = ButtonMenu(1.0, self)

        self.exclude_include_menu = ButtonMenu(1.0, self)
        self.exclude_include_menu.setVisible(False)

        self.process_menu = ButtonMenu(1.0, self)
        self.process_menu.setVisible(False)

        self.mouse_zoom_pic = QPixmap()
        self.mouse_pan_pic = QPixmap()
        self.icons_rect = QRectF(0, 0, 0, 0)
        self.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
        self.process_photo_popup = ButtonMenu(1.0, self)
        self.process_photo_popup.set_layout_direction('vertical')
        self.sticks_length_input = TextInputWidget(mode='number', label='Sticks length(cm):', parent=self)
        self.sticks_length_input.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
        self.sticks_length_input.setVisible(False)
        self.sticks_length_input.adjust_layout()
        self.sticks_length_input.setZValue(2)

        self.stick_length_input = TextInputWidget(mode='number', label='Sticks length(cmi):', parent=self)
        self.stick_length_input.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
        self.stick_length_input.setVisible(False)
        self.stick_length_input.adjust_layout()
        self.stick_length_input.setZValue(2)

        self.stick_label_input = TextInputWidget(mode='text', label='Label:', parent=self)
        self.stick_label_input.setVisible(False)
        self.stick_label_input.adjust_layout()
        self.stick_label_input.setZValue(2)

        self.stick_widget_menu = ButtonMenu(1.0, self)
        self.stick_widget_menu.show_close_button(True)
        self.stick_widget_menu.setVisible(False)

        self.submenus: List[ButtonMenu] = [self.process_menu, self.exclude_include_menu]

    def initialize(self):
        path = Path(sys.argv[0]).parent / "camera_processing/gui_resources/"

        self.mouse_zoom_pic = QPixmap(str(path / "mouse_zoom.png"))
        self.mouse_zoom_pic = self.mouse_zoom_pic.scaledToWidth(80, Qt.SmoothTransformation)

        self.mouse_pan_pic = QPixmap(str(path / "mouse_pan.png"))
        self.mouse_pan_pic = self.mouse_pan_pic.scaledToWidth(80, Qt.SmoothTransformation)

        self.top_menu.add_button('find_sticks', 'Find sticks', call_back=self.find_sticks_clicked)
        self.top_menu.add_button("edit_sticks", "Edit sticks", is_checkable=True,
                                 call_back=self.edit_sticks_clicked.emit)
        button = self.top_menu.add_button('sticks_length', 'Sticks length', is_checkable=True,
                                 call_back=self.sticks_length_clicked.emit)
        self.sticks_length_input.input_cancelled.connect(lambda: button.click_button(True))
        self.sticks_length_input.input_entered.connect(lambda: button.click_button(True))
        self.top_menu.add_button("link_sticks", "Link sticks", is_checkable=True,
                                 call_back=self.link_sticks_clicked.emit)
        self.enable_link_sticks_button(False)
        self.top_menu.add_button("reset_view", "Reset view", call_back=self.reset_view_requested.emit)
        self.top_menu.add_button("delete_sticks", "Delete selected sticks", call_back=self.delete_sticks_clicked.emit, base_color=ButtonColor.RED)
        self.top_menu.hide_button("delete_sticks")
        self.top_menu.add_button("photo_analysis", "Photo analysis", is_checkable=True,
                                 call_back=self.handle_photo_analysis_clicked)
        self.process_menu.add_button('confirm_sticks', 'Confirm sticks', call_back=self.handle_confirm_sticks_clicked,
                                     is_checkable=True)
        nighttime_btn = self.process_menu.add_button("process_nighttime", "Process nighttime photos: Yes",
                                                     is_checkable=True, call_back=self.handle_process_nighttime_clicked)
        nighttime_btn.set_on(True)
        process_btn = self.process_menu.add_button("process_photos", "Process photos", is_checkable=True,
                                                   call_back=self.handle_process_photos_clicked)
        process_btn.set_disabled(True)
        btn = self.process_menu.add_button("reset_measurements", "Reset measurements", ButtonColor.RED,
                                     call_back=self.reset_measurements_clicked.emit)
        btn.set_disabled(True)
        self.process_menu.add_button("process_stop", "Stop processing", ButtonColor.RED,
                                 call_back=self.process_stop_clicked.emit)
        self.process_menu.hide_button("process_stop")
        btn = self.top_menu.add_button("show_measurements", "Show measurements", call_back=self.show_measurements.emit,
                                 is_checkable=True)
        btn.set_base_color([ButtonColor.GRAY, ButtonColor.GREEN])

        self.top_menu.add_button("measurement_mode", "Measurement mode",  is_checkable=True,
                                 call_back=self.toggle_measurement_mode)

        self.enable_measurement_mode_button(False)

        self.exclude_include_menu.add_button("exclude_photos_no_snow", "Mark as 'no snow'", ButtonColor.RED,
                                             call_back=self.exclude_photos_no_snow.emit)
        self.exclude_include_menu.add_button("exclude_photos_bad", "Mark as 'bad quality'", ButtonColor.RED,
                                             call_back=self.exclude_photos_bad_quality.emit)
        self.exclude_include_menu.add_button("include_photos", "Mark as 'snowy'", ButtonColor.GREEN,
                                             call_back=self.include_photos.emit)
        self.exclude_include_menu.center_buttons()

        self.top_menu.set_height(12)
        self.top_menu.center_buttons()

        self.top_menu.setPos(QPoint(0, 0))

        self.icons_rect = QRectF(5, 5, self.mouse_pan_pic.width() * 1.3,
                                       3 * self.mouse_pan_pic.height())

        btn = self.stick_widget_menu.add_button('set_stick_length', 'Set length',
                                                call_back=self.set_stick_length_clicked.emit, is_checkable=True)
        self.stick_length_input.input_cancelled.connect(self.hide_stick_length_input)
        self.stick_length_input.input_entered.connect(self.hide_stick_length_input)

        btn = self.stick_widget_menu.add_button('set_stick_label', 'Change label',
                                                call_back=self.set_stick_label_clicked.emit, is_checkable=True)
        self.stick_label_input.input_cancelled.connect(self.hide_stick_label_input)
        self.stick_label_input.input_entered.connect(self.hide_stick_label_input)
        self.stick_widget_menu.set_layout_direction('vertical')

        self.initialize_process_photos_popup()

    @pyqtSlot()
    def handle_cam_view_changed(self):
        self.prepareGeometryChange()
        self.setPos(self.view.mapToScene(QPoint(0, 0)))
        self.top_menu.setPos(self.boundingRect().width() / 2.0 - self.top_menu.boundingRect().width() / 2.0, 2)
        self.exclude_include_menu.setPos(
            self.boundingRect().width() / 2.0 - self.exclude_include_menu.boundingRect().width() / 2.0,
            self.top_menu.pos().y() + self.top_menu.boundingRect().height())
        self.sticks_length_input.setPos(self.view.size().width() * 0.5, self.view.size().height() * 0.5)
        self.stick_length_input.setPos(self.view.size().width() * 0.5, self.view.size().height() * 0.5)
        self.stick_label_input.setPos(self.view.size().width() * 0.5, self.view.size().height() * 0.5)

    def boundingRect(self):
        return QRectF(self.view.viewport().rect())
    
    def paint(self, painter: QPainter, options, widget=None):
        painter.save()

        painter.setWorldMatrixEnabled(False)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor("#aa555555")))
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.TextAntialiasing, True)

        painter.setPen(QPen(QColor("#e5c15f")))
        font = painter.font()
        font.setPointSize(10)

        painter.setWorldMatrixEnabled(True)
        painter.restore()

    def edit_sticks_button_pushed(self) -> bool:
        return self.top_menu.is_button_checked("edit_sticks")
    
    def link_sticks_button_pushed(self) -> bool:
        return self.top_menu.is_button_checked("link_sticks")

    def enable_delete_sticks_button(self, val: bool):
        self.top_menu.show_button("delete_sticks") if val else self.top_menu.hide_button("delete_sticks")
        self.handle_cam_view_changed()

    def initialize_process_photos_popup(self):
        self.process_photo_popup.set_width(100)
        for i in range(os.cpu_count()):
            self.process_photo_popup.add_button(str(i+1), f'Assign {i+1} cores',
                                                call_back=self.handle_process_jobs_count_clicked)

        self.process_photo_popup.show_close_button(True)
        self.process_photo_popup.set_layout_direction('vertical')
        self.process_photo_popup.center_buttons()
        self.process_photo_popup.close_button.clicked.connect(self.handle_process_photos_count_cancel_clicked)
        self.process_photo_popup.setVisible(False)
        self.process_photo_popup.update()
        self.update(self.boundingRect())

    def handle_process_photos_count_cancel_clicked(self):
        self.process_photo_popup.setVisible(False)
        self.process_menu.get_button('process_photos').click_button(True)
        self.process_photo_popup.reset_button_states()

    def handle_process_jobs_count_clicked(self, btn_data: Dict[str, Any]):
        self.process_photo_popup.setVisible(False)
        self.process_photo_popup.reset_button_states()
        btn = self.process_menu.get_button('process_photos')
        btn.click_button(True)
        self.process_menu.get_button("confirm_sticks").set_disabled(True)
        self.process_menu.get_button("process_nighttime").set_disabled(True)
        self.process_photos_with_jobs_clicked.emit(int(btn_data['btn_id']))

    def handle_process_photos_clicked(self):
        self.process_photo_popup.setPos(self.boundingRect().width() / 2.0 - self.process_photo_popup.boundingRect().width() / 2.0,
                                        self.boundingRect().height() / 2.0 - self.process_photo_popup.boundingRect().height() / 2.0)
        is_down = self.process_menu.get_button('process_photos').is_on()
        self.process_photo_popup.setVisible(is_down)

    def enable_process_photos_button(self, enable: bool):
        self.process_menu.get_button('process_photos').set_disabled(not enable)

    def enable_link_sticks_button(self, enable: bool):
        self.top_menu.get_button('link_sticks').set_disabled(not enable)

    def enable_confirm_sticks_button(self, enable: bool):
        self.process_menu.get_button('confirm_sticks').set_disabled(not enable)

    def handle_confirm_sticks_clicked(self, btn_info: Dict[str, Any]):
        self.confirm_sticks_clicked.emit(btn_info)
        btn = self.process_menu.get_button('process_photos')
        if btn is not None:
            self.process_menu.get_button('process_photos').set_disabled(not btn_info['checked'])

    def show_stick_context_menu_at(self, pos: QPoint):
        self.stick_widget_menu.setPos(pos)
        self.stick_widget_menu.setVisible(True)

    def hide_stick_context_menu(self):
        self.hide_stick_label_input()
        self.hide_stick_label_input()
        self.stick_widget_menu.setVisible(False)

    def show_stick_label_input(self):
        self.hide_sticks_length_input()
        self.hide_stick_length_input()
        self.stick_label_input.set_focus()
        self.stick_label_input.setVisible(True)

    def hide_stick_label_input(self):
        self.stick_widget_menu.get_button('set_stick_label').set_default_state()
        self.stick_label_input.setVisible(False)

    def show_stick_length_input(self):
        self.hide_stick_label_input()
        self.hide_sticks_length_input()
        self.stick_length_input.set_focus()
        self.stick_length_input.setVisible(True)

    def hide_stick_length_input(self):
        self.stick_widget_menu.get_button('set_stick_length').set_default_state()
        self.stick_length_input.setVisible(False)

    def show_sticks_length_input(self):
        self.hide_stick_context_menu()
        self.sticks_length_input.set_focus()
        self.sticks_length_input.setVisible(True)

    def hide_sticks_length_input(self):
        self.sticks_length_input.setVisible(False)

    def hide_process_photos(self):
        self.process_menu.hide_button("process_photos")

    def show_process_photos(self):
        self.process_menu.show_button("process_photos")

    def hide_process_stop(self):
        self.process_menu.hide_button("process_stop")

    def show_process_stop(self):
        self.process_menu.show_button("process_stop")

    def handle_process_count_changed(self, _: int, _i: int, job_count: int, processing_stopped: bool):
        btn = self.process_menu.get_button("process_stop")
        if processing_stopped:
            if job_count > 0:
                btn.set_disabled(True)
                btn.set_label(f'Waiting on {job_count} job(s)')
                btn.fit_to_contents()
                self.process_menu.center_buttons()
            else:
                self.handle_processing_stopped()
        self.update()

    def handle_processing_stopped(self):
        btn = self.process_menu.get_button("process_stop")
        self.hide_process_stop()
        if btn is None:
            return
        btn.set_disabled(False)
        btn.set_label("Stop processing")
        self.show_process_photos()
        self.process_menu.center_buttons()
        self.process_menu.get_button("confirm_sticks").set_disabled(False)
        self.process_menu.get_button("process_nighttime").set_disabled(False)
        self.update()

    def toggle_edit_sticks_button(self):
        self.top_menu.get_button("edit_sticks").click_button(artificial_emit=True)

    def uncheck_confirm_sticks_button(self):
        btn = self.process_menu.get_button("confirm_sticks")
        if btn is not None:
            if btn.is_on():
                btn.click_button(artificial_emit=True)

    def check_confirm_sticks_button(self):
        btn = self.process_menu.get_button("confirm_sticks")
        if btn is not None:
            if not btn.is_on():
                btn.click_button(artificial_emit=True)

    def show_exclude_button(self):
        if not self.exclude_include_menu.isVisible():
            self.exclude_include_menu.setVisible(True)
        self.exclude_include_menu.show_button("exclude_photos_no_snow")
        self.exclude_include_menu.show_button("exclude_photos_bad_quality")
        self.exclude_include_menu.setPos(
            self.boundingRect().width() / 2.0 - self.exclude_include_menu.boundingRect().width() / 2.0,
            self.top_menu.pos().y() + self.top_menu.boundingRect().height())

    def hide_exclude_button(self):
        self.exclude_include_menu.hide_button("exclude_photos_no_snow")
        self.exclude_include_menu.hide_button("exclude_photos_bad_quality")

    def show_include_button(self):
        if not self.exclude_include_menu.isVisible():
            self.exclude_include_menu.setVisible(True)
        self.exclude_include_menu.show_button("include_photos")
        self.exclude_include_menu.setPos(
            self.boundingRect().width() / 2.0 - self.exclude_include_menu.boundingRect().width() / 2.0,
            self.top_menu.pos().y() + self.top_menu.boundingRect().height())

    def hide_include_button(self):
        self.exclude_include_menu.hide_button("include_photos")

    def hide_exclude_include_menu(self):
        self.exclude_include_menu.hide_button("exclude_photos_no_snow")
        self.exclude_include_menu.hide_button("exclude_photos_bad_quality")
        self.exclude_include_menu.hide_button("include_photos")
        self.exclude_include_menu.setVisible(False)

    def toggle_measurement_mode(self):
        self.measurement_mode_toggle.emit(self.top_menu.get_button("measurement_mode").is_on())

    def enable_measurement_mode_button(self, enable: bool):
        self.top_menu.get_button("measurement_mode").set_disabled(not enable)

    def handle_photo_analysis_clicked(self):
        is_on = self.top_menu.get_button("photo_analysis").is_on()
        if is_on:
            for menu in self.submenus:
                menu.setVisible(False)
            self.process_menu.setPos(
                self.boundingRect().width() / 2.0 - self.process_menu.boundingRect().width() / 2.0,
                self.top_menu.pos().y() + self.top_menu.boundingRect().height())
        self.process_menu.setVisible(is_on)

    def handle_process_nighttime_clicked(self, btn_info: Dict[str, Any]):
        is_on = btn_info['checked']
        btn = self.process_menu.get_button("process_nighttime")
        btn.set_label(f'Process nighttime photos: {"Yes" if is_on else "No"}')
        self.process_menu.center_buttons()

    def enable_reset_measurements(self, enable: bool):
        self.process_menu.get_button("reset_measurements").set_disabled(not enable)
