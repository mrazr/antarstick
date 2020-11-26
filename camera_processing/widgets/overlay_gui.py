import sys
from pathlib import Path
from typing import Dict, Any
import os

from PyQt5.Qt import QBrush, QColor, QPen
from PyQt5.QtCore import QPoint, QPointF, QRectF, Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QPainter, QPixmap, QFontMetrics
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
    redetect_sticks_clicked = pyqtSignal()
    process_photos_clicked = pyqtSignal('PyQt_PyObject')
    process_photos_with_jobs_clicked = pyqtSignal(int)
    clicked = pyqtSignal()
    find_sticks_clicked = pyqtSignal()
    detect_thin_sticks_set = pyqtSignal('PyQt_PyObject')
    sticks_length_clicked = pyqtSignal('PyQt_PyObject')
    confirm_sticks_clicked = pyqtSignal()
    set_stick_label_clicked = pyqtSignal('PyQt_PyObject')
    set_stick_length_clicked = pyqtSignal('PyQt_PyObject')
    process_stop_clicked = pyqtSignal()
    mes = pyqtSignal()
    save_measurements = pyqtSignal()
    use_single_proc = pyqtSignal('PyQt_PyObject')

    def __init__(self, view: CamGraphicsView, parent: QGraphicsItem = None):
        QGraphicsObject.__init__(self, parent)

        self.view = view
        self.view.view_changed.connect(self.handle_cam_view_changed)
        self.setZValue(43)
        self.setAcceptHoverEvents(False)
        self.setAcceptedMouseButtons(Qt.AllButtons)
        self.top_menu = ButtonMenu(1.0, self)
        self.find_sticks_menu = ButtonMenu(1.0, self.top_menu)
        self.find_sticks_menu.setVisible(False)
        self.mouse_zoom_pic = QPixmap()
        self.mouse_pan_pic = QPixmap()
        self.icons_rect = QRectF(0, 0, 0, 0)
        self.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
        #self.top_menu.setVisible(False)
        self.loading_screen_shown = True
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
        #self.stick_label_input.input_entered.connect(self.set_stick_label.emit)
        #self.stick_label_input.input_cancelled.connect(self.set_stick_label.emit)

        self.stick_widget_menu = ButtonMenu(1.0, self)
        self.stick_widget_menu.show_close_button(True)
        self.stick_widget_menu.setVisible(False)

    def initialize(self):
        path = Path(sys.argv[0]).parent / "camera_processing/gui_resources/"

        self.mouse_zoom_pic = QPixmap(str(path / "mouse_zoom.png"))
        self.mouse_zoom_pic = self.mouse_zoom_pic.scaledToWidth(80, Qt.SmoothTransformation)

        self.mouse_pan_pic = QPixmap(str(path / "mouse_pan.png"))
        self.mouse_pan_pic = self.mouse_pan_pic.scaledToWidth(80, Qt.SmoothTransformation)

        self.top_menu.add_button('find_sticks', 'Find sticks', call_back=self.find_sticks_clicked)
        # self.top_menu.add_button("find_sticks", "Find sticks",
        #                          call_back=lambda: self.find_sticks_menu.setVisible(not self.find_sticks_menu.isVisible()),
        #                          is_checkable=True)
        self.top_menu.add_button('confirm_sticks', 'Confirm sticks', call_back=self.handle_confirm_sticks_clicked,
                                 is_checkable=True)
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
        process_btn = self.top_menu.add_button("process_photos", "Process photos", is_checkable=True,
                                               call_back=self.handle_process_photos_clicked)
        process_btn.set_disabled(True)
        self.top_menu.add_button("process_stop", "Stop processing", ButtonColor.RED,
                                 call_back=self.process_stop_clicked.emit)
        self.top_menu.hide_button("process_stop")
        self.top_menu.add_button("measure_snow", "Measure", call_back=self.mes.emit)
        self.top_menu.add_button("save_measurements", "Save measurements", call_back=self.save_measurements.emit)
        self.top_menu.add_button("use_single_proc", "Use single process", call_back=self.use_single_proc.emit,
                                 is_checkable=True)
        self.top_menu.set_height(12)
        self.top_menu.center_buttons()

        self.top_menu.setPos(QPoint(0, 0))

        self.icons_rect = QRectF(5, 5, self.mouse_pan_pic.width() * 1.3,
                                       3 * self.mouse_pan_pic.height())

        self.find_sticks_menu.add_button('detect_thin_sticks', 'Thin sticks',
                                         call_back=self.detect_thin_sticks_set.emit,
                                         is_checkable=True)
        self.find_sticks_menu.add_button('find_sticks', 'Find', call_back=self.find_sticks_clicked.emit)
        self.find_sticks_menu.setPos(QPoint(0, 40))
        self.find_sticks_menu.set_height(12)

        btn = self.stick_widget_menu.add_button('set_stick_length', 'Set length',
                                                call_back=self.set_stick_length_clicked.emit, is_checkable=True)
        self.stick_length_input.input_cancelled.connect(self.hide_stick_length_input)
        self.stick_length_input.input_entered.connect(self.hide_stick_length_input)

        btn = self.stick_widget_menu.add_button('set_stick_label', 'Change label',
                                                call_back=self.set_stick_label_clicked.emit, is_checkable=True)
        self.stick_label_input.input_cancelled.connect(self.hide_stick_label_input)
        self.stick_label_input.input_entered.connect(self.hide_stick_label_input)
        #self.top_menu.set_height(12)
        self.stick_widget_menu.set_layout_direction('vertical')
        #self.stick_widget_menu.center_buttons()
        self.initialize_process_photos_popup()

    @pyqtSlot()
    def handle_cam_view_changed(self):
        self.prepareGeometryChange()
        self.setPos(self.view.mapToScene(QPoint(0, 0)))
        self.top_menu.setPos(self.boundingRect().width() / 2.0 - self.top_menu.boundingRect().width() / 2.0, 2)
        #self.process_photo_popup.setPos(self.boundingRect().width() / 2.0 - self.process_photo_popup.boundingRect().width() / 2.0,
        #                                self.boundingRect().height() / 2.0 - self.process_photo_popup.boundingRect().height() / 2.0)
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

        if self.loading_screen_shown:
            painter.fillRect(0, 0, self.boundingRect().width(), self.boundingRect().height(), QBrush(QColor(255, 255, 255, 255)))
            fm = QFontMetrics(font)
            font.setPixelSize(int(self.boundingRect().width() / 15))
            painter.setFont(font)
            painter.drawText(self.boundingRect(), Qt.AlignCenter, 'initializing')
        painter.setWorldMatrixEnabled(True)
        painter.restore()

    def edit_sticks_button_pushed(self) -> bool:
        return self.top_menu.is_button_checked("edit_sticks")
    
    def link_sticks_button_pushed(self) -> bool:
        return self.top_menu.is_button_checked("link_sticks")

    def show_loading_screen(self, show: bool):
        self.loading_screen_shown = show
        self.top_menu.setVisible(not self.loading_screen_shown)
        self.update()

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
        self.top_menu.get_button('process_photos').click_button(True)
        self.process_photo_popup.reset_button_states()

    def handle_process_jobs_count_clicked(self, btn_data: Dict[str, Any]):
        self.process_photo_popup.setVisible(False)
        self.process_photo_popup.reset_button_states()
        btn = self.top_menu.get_button('process_photos')
        btn.click_button(True)
        self.process_photos_with_jobs_clicked.emit(int(btn_data['btn_id']))

    def handle_process_photos_clicked(self):
        self.process_photo_popup.setPos(self.boundingRect().width() / 2.0 - self.process_photo_popup.boundingRect().width() / 2.0,
                                        self.boundingRect().height() / 2.0 - self.process_photo_popup.boundingRect().height() / 2.0)
        is_down = self.top_menu.get_button('process_photos').is_on()
        self.process_photo_popup.setVisible(is_down)

    def enable_process_photos_button(self, enable: bool):
        self.top_menu.get_button('process_photos').set_disabled(not enable)

    def enable_link_sticks_button(self, enable: bool):
        self.top_menu.get_button('link_sticks').set_disabled(not enable)

    def enable_confirm_sticks_button(self, enable: bool):
        self.top_menu.get_button('confirm_sticks').set_disabled(not enable)

    #def handle_sticks_length_clicked(self, btn_state: Dict[str, Any]):
    #    self.sticks_length_input.setVisible(btn_state['checked'])
    #    if btn_state['checked']:
    #        self.sticks_length_input.set_focus()

    #def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
    #    event.ignore()

    #def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent) -> None:
    #    self.clicked.emit()
    #    event.ignore()

    def handle_confirm_sticks_clicked(self, btn_info: Dict[str, Any]):
        if btn_info['checked']:
            self.confirm_sticks_clicked.emit()
        self.top_menu.get_button('process_photos').set_disabled(not btn_info['checked'])

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
        self.top_menu.hide_button("process_photos")

    def show_process_photos(self):
        self.top_menu.show_button("process_photos")

    def hide_process_stop(self):
        self.top_menu.hide_button("process_stop")

    def show_process_stop(self):
        self.top_menu.show_button("process_stop")

    def handle_process_count_changed(self, _: int, _i: int, job_count: int, processing_stopped: bool):
        btn = self.top_menu.get_button("process_stop")
        if processing_stopped:
            if job_count > 0:
                btn.set_disabled(True)
                btn.set_label(f'Waiting on {job_count} job(s)')
                btn.fit_to_contents()
                self.top_menu.center_buttons()
            else:
                self.handle_processing_stopped()
        self.update()

    def handle_processing_stopped(self):
        btn = self.top_menu.get_button("process_stop")
        if btn is None:
            return
        self.hide_process_stop()
        btn.set_disabled(False)
        btn.set_label("Stop processing")
        self.show_process_photos()
        self.top_menu.center_buttons()
        self.update()
