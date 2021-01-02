from pathlib import Path
from typing import Callable, List, Optional, Dict

import PyQt5
from PyQt5.QtCore import QByteArray, QLine, QMarginsF, QPoint, pyqtSignal, QPointF, Qt, QTimer, QTimerEvent, QThread, \
    QThreadPool, QRunnable, pyqtProperty, QPropertyAnimation, QEasingCurve, QAbstractAnimation, QRectF
from PyQt5.QtGui import QBrush, QColor, QFont, QImage, QPainter, QPen, QPixmap, QStaticText, QKeyEvent
from PyQt5.QtWidgets import (QGraphicsItem, QGraphicsPixmapItem,
                             QGraphicsRectItem, QGraphicsSceneHoverEvent,
                             QGraphicsSceneMouseEvent, QGraphicsSimpleTextItem,
                             QWidget, QGraphicsObject, QStyleOptionGraphicsItem, QGraphicsBlurEffect)
import numpy as np
from PyQt5.uic.properties import QtGui

from camera import Camera
from camera_processing.widgets.stick_length_input import TextInputWidget
from camera_processing.widgets.stick_widget import StickWidget, StickMode
from camera_processing.widgets.button import Button, ButtonColor, ButtonMode
from dataset import Dataset
from camera_processing.antarstick_processing import get_stick_area
from stick import Stick


class Timer(QRunnable):

    def __init__(self, func, duration_ms: int):
        QRunnable.__init__(self)
        self.func = func
        self.duration_ms = duration_ms

    def run(self) -> None:
        QThread.sleep(self.duration_ms)
        self.func()


class ControlWidget(QGraphicsObject):

    def __init__(self, parent: Optional[QGraphicsItem] = None):
        QGraphicsObject.__init__(self, parent)
        self.mode = 'view' # 'sync'
        self.widget_width: int = 0
        self.title_btn = Button('btn_title', '', parent=self)
        self.title_btn.setFlag(QGraphicsItem.ItemIgnoresTransformations, False)
        self.title_btn.setZValue(5)
        self.title_btn.setVisible(True)
        self.title_btn.set_mode(ButtonMode.Label)

        self.first_photo_btn = Button('btn_first_photo', 'First photo', parent=self)
        self.first_photo_btn.setFlag(QGraphicsItem.ItemIgnoresTransformations, False)
        self.prev_photo_btn = Button('btn_prev_photo', '<', parent=self)
        self.prev_photo_btn.setFlag(QGraphicsItem.ItemIgnoresTransformations, False)
        self.enter_photo_num_btn = Button('btn_enter_photo_num', 'Manual', parent=self)
        self.enter_photo_num_btn.setFlag(QGraphicsItem.ItemIgnoresTransformations, False)
        self.enter_photo_num_btn.set_is_check_button(True)
        self.enter_photo_num_btn.setVisible(False)

        self.accept_btn = Button('btn_accept_sync', 'Confirm', parent=self)
        self.accept_btn.set_base_color([ButtonColor.GREEN])
        self.cancel_btn = Button('btn_cancel_sync', 'Cancel', parent=self)
        self.cancel_btn.set_base_color([ButtonColor.RED])

        self.next_photo_btn = Button('btn_next_photo', '>', parent=self)
        self.next_photo_btn.setFlag(QGraphicsItem.ItemIgnoresTransformations, False)
        self.synchronize_btn = Button('btn_synchronize', 'Synchronize', parent=self)
        self.synchronize_btn.setFlag(QGraphicsItem.ItemIgnoresTransformations, False)

        self.sync_mode_btns = [self.prev_photo_btn, self.accept_btn, self.first_photo_btn, self.cancel_btn,
                               self.next_photo_btn]
        self.view_mode_btns = [self.prev_photo_btn, self.title_btn, self.synchronize_btn, self.next_photo_btn]

        for btn in self.sync_mode_btns:
            btn.setFlag(QGraphicsItem.ItemIgnoresTransformations, False)
        for btn in self.view_mode_btns:
            btn.setFlag(QGraphicsItem.ItemIgnoresTransformations, False)

        self.set_font_height(12)

        self._layout()

    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem, widget: Optional[QWidget] = ...) -> None:
        pass

    def boundingRect(self) -> QRectF:
        return QRectF(self.first_photo_btn.boundingRect().topLeft(),
                      self.synchronize_btn.pos() + self.synchronize_btn.boundingRect().bottomRight())

    def _layout(self):
        if self.mode == 'sync':
            btns = self.sync_mode_btns
        else:
            btns = self.view_mode_btns

        for i, btn in enumerate(btns):
            if i == 0:
                btn.setPos(0, 0)
                continue
            prev_btn = btns[i-1]
            btn.setPos(prev_btn.pos().x() + prev_btn.boundingRect().width(), 0)

    def set_widget_height(self, height: int):
        for btn in self.sync_mode_btns:
            btn.set_button_height(height)
        for btn in self.view_mode_btns:
            btn.set_button_height(height)
        self._layout()

    def set_font_height(self, height: int):
        for btn in self.sync_mode_btns:
            btn.set_height(height)
            btn.fit_to_contents()
        for btn in self.view_mode_btns:
            btn.set_height(height)
            btn.fit_to_contents()
        self._layout()

    def set_widget_width(self, width: int):
        self.widget_width = width
        btns = self.sync_mode_btns if self.mode == 'sync' else self.view_mode_btns
        units_per_character = width / sum(map(lambda btn: len(btn.label.text()), btns))

        for btn in btns:
            btn.set_width(int(round(len(btn.label.text()) * units_per_character)))

        self._layout()

    def set_mode(self, mode: str):
        self.mode = mode
        if self.mode == 'sync':
            for btn in self.view_mode_btns:
                btn.setVisible(False)
            for btn in self.sync_mode_btns:
                btn.setVisible(True)
        else:
            for btn in self.sync_mode_btns:
                btn.setVisible(False)
            for btn in self.view_mode_btns:
                btn.setVisible(True)
        self.set_widget_width(self.widget_width)

    def disable_widget(self):
        for btn in self.sync_mode_btns:
            btn.set_disabled(True)
        for btn in self.view_mode_btns:
            btn.set_disabled(True)

    def enable_widget(self):
        for btn in self.sync_mode_btns:
            btn.set_disabled(False)
        for btn in self.view_mode_btns:
            btn.set_disabled(False)

    def show_sync_button(self, show: bool):
        if show:
            if self.synchronize_btn not in self.view_mode_btns:
                self.view_mode_btns.insert(2, self.synchronize_btn)
        else:
            if self.synchronize_btn in self.view_mode_btns:
                self.view_mode_btns.remove(self.synchronize_btn)
        self.synchronize_btn.setVisible(show)
        if self.mode == 'view':
            self._layout()

    def set_title_text(self, text: str):
        self.title_btn.set_label(text)
        self.set_widget_width(self.widget_width)
        self._layout()


class CameraView(QGraphicsObject):

    font: QFont = QFont("monospace", 16)
    stick_link_requested = pyqtSignal(StickWidget)
    stick_context_menu = pyqtSignal('PyQt_PyObject', 'PyQt_PyObject')
    stick_widgets_out_of_sync = pyqtSignal('PyQt_PyObject')
    visibility_toggled = pyqtSignal()
    synchronize_clicked = pyqtSignal('PyQt_PyObject')
    previous_photo_clicked = pyqtSignal('PyQt_PyObject')
    next_photo_clicked = pyqtSignal('PyQt_PyObject')
    sync_confirm_clicked = pyqtSignal('PyQt_PyObject')
    sync_cancel_clicked = pyqtSignal('PyQt_PyObject')
    first_photo_clicked = pyqtSignal('PyQt_PyObject')
    enter_pressed = pyqtSignal()

    def __init__(self, scale: float, parent: Optional[QGraphicsItem] = None):
        QGraphicsObject.__init__(self, parent)
        self.current_highlight_color = QColor(0, 0, 0, 0)
        self.current_timer = -1
        self.scaling = scale
        self.pixmap = QGraphicsPixmapItem(self)
        self.stick_widgets: List[StickWidget] = []
        self.link_cam_text = QGraphicsSimpleTextItem("Link camera...", self)
        self.link_cam_text.setZValue(40)
        self.link_cam_text.setVisible(False)
        self.link_cam_text.setFont(CameraView.font)
        self.link_cam_text.setPos(0, 0)
        self.link_cam_text.setPen(QPen(QColor(255, 255, 255, 255)))
        self.link_cam_text.setBrush(QBrush(QColor(255, 255, 255, 255)))

        self.show_add_buttons = False
        self.camera = None

        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)
        self.show_stick_widgets = False
        self.setAcceptHoverEvents(True)
        self.stick_edit_mode = False

        self.original_pixmap = self.pixmap.pixmap()

        self.hovered = False

        self.mode = 0 # TODO make enum Mode
        self.click_handler = None
        self.double_click_handler: Callable[[int, int], None] = None

        self.stick_widget_mode = StickMode.Display

        self.highlight_animation = QPropertyAnimation(self, b"highlight_color")
        self.highlight_animation.setEasingCurve(QEasingCurve.Linear)
        self.highlight_animation.valueChanged.connect(self.handle_highlight_color_changed)
        self.highlight_rect = QGraphicsRectItem(self)
        self.highlight_rect.setZValue(4)
        self.highlight_rect.setPen(QPen(QColor(0, 0, 0, 0)))
        self.title_btn = Button('btn_title', '', parent=self)
        self.title_btn.setFlag(QGraphicsItem.ItemIgnoresTransformations, False)
        self.title_btn.setZValue(5)
        self.title_btn.setVisible(False)

        self.sticks_without_width: List[Stick] = []
        self.current_image_name: str = ''
        self.control_widget = ControlWidget(parent=self)
        self.control_widget.setFlag(QGraphicsItem.ItemIgnoresTransformations, False)
        self.control_widget.setVisible(True)
        self._connect_control_buttons()
        self.image_available = True
        self.blur_eff = QGraphicsBlurEffect()
        self.blur_eff.setBlurRadius(5.0)
        self.blur_eff.setEnabled(False)
        self.pixmap.setGraphicsEffect(self.blur_eff)
        self.overlay_message = QGraphicsSimpleTextItem('not available', parent=self)
        font = self.title_btn.font
        font.setPointSize(48)
        self.overlay_message.setFont(font)
        self.overlay_message.setBrush(QBrush(QColor(200, 200, 200, 200)))
        self.overlay_message.setPen(QPen(QColor(0, 0, 0, 200), 2.0))
        self.overlay_message.setVisible(False)
        self.overlay_message.setZValue(6)

        self.stick_box = QGraphicsRectItem(parent=self)
        self.stick_box.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.stick_box.setVisible(False)
        self.stick_box_start_pos = QPoint()

    def _connect_control_buttons(self):
        self.control_widget.synchronize_btn.clicked.connect(lambda: self.synchronize_clicked.emit(self))
        self.control_widget.prev_photo_btn.clicked.connect(lambda: self.previous_photo_clicked.emit(self))
        self.control_widget.next_photo_btn.clicked.connect(lambda: self.next_photo_clicked.emit(self))
        self.control_widget.accept_btn.clicked.connect(lambda: self.sync_confirm_clicked.emit(self))
        self.control_widget.cancel_btn.clicked.connect(lambda: self.sync_cancel_clicked.emit(self))
        self.control_widget.first_photo_btn.clicked.connect(lambda: self.first_photo_clicked.emit(self))

    def paint(self, painter: QPainter, option: PyQt5.QtWidgets.QStyleOptionGraphicsItem, widget: QWidget):
        if self.pixmap.pixmap().isNull():
            return
        painter.setRenderHint(QPainter.Antialiasing, True)

        if self.show_stick_widgets:
            brush = QBrush(QColor(255, 255, 255, 100))
            painter.fillRect(self.boundingRect(), brush)

        if self.mode and self.hovered:
            pen = QPen(QColor(0, 125, 200, 255))
            pen.setWidth(4)
            painter.setPen(pen)
            #painter.drawRect(self.boundingRect().marginsAdded(QMarginsF(4, 4, 4, 4)))

    def boundingRect(self) -> PyQt5.QtCore.QRectF:
        return self.pixmap.boundingRect().united(self.title_btn.boundingRect().translated(self.title_btn.pos()))

    def initialise_with(self, camera: Camera):
        if self.camera is not None:
            self.camera.stick_added.disconnect(self.handle_stick_created)
            self.camera.sticks_added.disconnect(self.handle_sticks_added)
            self.camera.stick_removed.disconnect(self.handle_stick_removed)
            self.camera.sticks_removed.disconnect(self.handle_sticks_removed)
            self.camera.stick_changed.disconnect(self.handle_stick_changed)
        self.camera = camera
        #self.stick_length_btn.btn_id = str(self.camera.folder)
        self.prepareGeometryChange()
        self.set_image(camera.rep_image, Path(camera.rep_image_path).name)
        self.title_btn.set_label(self.camera.folder.name)
        self.title_btn.set_height(46)
        self.title_btn.fit_to_contents()
        self.title_btn.set_width(int(self.boundingRect().width()))
        self.title_btn.setPos(0, self.boundingRect().height())
        self.control_widget.title_btn.set_label(self.camera.folder.name)
        #self.stick_length_btn.setVisible(True)
        #self.stick_length_lbl.setVisible(True)
        self.camera.stick_added.connect(self.handle_stick_created)
        self.camera.sticks_added.connect(self.handle_sticks_added)
        self.camera.stick_removed.connect(self.handle_stick_removed)
        self.camera.sticks_removed.connect(self.handle_sticks_removed)
        self.camera.stick_changed.connect(self.handle_stick_changed)

        self.control_widget.set_font_height(32)
        self.control_widget.set_widget_height(self.title_btn.boundingRect().height())
        self.control_widget.set_widget_width(int(self.boundingRect().width()))
        self.control_widget.setPos(0, self.pixmap.boundingRect().height()) #self.boundingRect().height())
        self.control_widget.set_mode('view')
        self.update_stick_widgets()

    def set_image(self, img: Optional[np.ndarray] = None, image_name: Optional[str] = None):
        if img is None:
            #self.overlay_message.setPos(self.pixmap.boundingRect().center() - QPointF(
            #    0.5 * self.overlay_message.boundingRect().width(),
            #    0.5 * self.overlay_message.boundingRect().height()
            #))
            #self.overlay_message.setVisible(True)
            #self.image_available = False
            #self.blur_eff.setEnabled(True)
            #self.update()
            self.show_overlay_message('not available')
            return
        #self.overlay_message.setVisible(False)
        #self.image_available = True
        #self.blur_eff.setEnabled(False)
        self.show_overlay_message(None)
        self.prepareGeometryChange()
        barray = QByteArray(img.tobytes())
        image = QImage(barray, img.shape[1], img.shape[0], QImage.Format_BGR888)
        self.original_pixmap = QPixmap.fromImage(image)
        self.pixmap.setPixmap(self.original_pixmap)
        self.highlight_rect.setRect(self.boundingRect())
        self.current_image_name = image_name

    def update_stick_widgets(self):
        stick_length = 60
        for stick in self.camera.sticks:
            sw = StickWidget(stick, self.camera, self)
            sw.set_mode(self.stick_widget_mode)
            self.connect_stick_widget_signals(sw)
            self.stick_widgets.append(sw)
            stick_length = stick.length_cm
        #self.stick_length_btn.set_label(str(stick_length) + " cm")
        #self.stick_length_btn.fit_to_contents()
        #self.stick_length_input.set_value(str(stick_length))
        #self.layout_title_area()
        self.update_stick_box()
        self.scene().update()

    def scale_item(self, factor: float):
        self.prepareGeometryChange()
        pixmap = self.original_pixmap.scaledToHeight(int(self.original_pixmap.height() * factor))
        self.pixmap.setPixmap(pixmap)
        self.__update_title()

    def set_show_stick_widgets(self, value: bool):
        for sw in self.stick_widgets:
            sw.setVisible(value)
        self.scene().update()

    def hoverEnterEvent(self, e: QGraphicsSceneHoverEvent):
        self.hovered = True
        self.scene().update(self.sceneBoundingRect())
    
    def hoverLeaveEvent(self, e: QGraphicsSceneHoverEvent):
        self.hovered = False
        self.scene().update(self.sceneBoundingRect())
    
    def mousePressEvent(self, e: QGraphicsSceneMouseEvent):
        super().mousePressEvent(e)

    def mouseReleaseEvent(self, e: QGraphicsSceneMouseEvent):
        if self.mode == 1:
            self.click_handler(self.camera)

    def mouseDoubleClickEvent(self, event: QGraphicsSceneMouseEvent):
        if self.stick_widget_mode == StickMode.EditDelete:
            x = event.pos().toPoint().x()
            y = event.pos().toPoint().y()
            stick = self.camera.create_new_sticks([(np.array([[x, y - 50], [x, y + 50]]), 3)], self.current_image_name)[0] #self.dataset.create_new_stick(self.camera)
            self.sticks_without_width.append(stick)

    def set_button_mode(self, click_handler: Callable[[Camera], None], data: str):
        self.mode = 1 # TODO make a proper ENUM
        self.click_handler = lambda c: click_handler(c, data)
    
    def set_display_mode(self):
        self.mode = 0 # TODO make a proper ENUM
        self.click_handler = None

    def _remove_stick_widgets(self):
        for sw in self.stick_widgets:
            sw.setParentItem(None)
            self.scene().removeItem(sw)
            sw.deleteLater()
        self.stick_widgets.clear()
    
    #def set_stick_edit_mode(self, value: bool):
    #    self.stick_edit_mode = value

    def handle_stick_created(self, stick: Stick):
        if stick.camera_id != self.camera.id:
            return
        sw = StickWidget(stick, self.camera, self)
        sw.set_mode(self.stick_widget_mode)
        self.connect_stick_widget_signals(sw)
        self.stick_widgets.append(sw)
        self.stick_widgets_out_of_sync.emit(self)
        self.update()

    def handle_stick_removed(self, stick: Stick):
        if stick.camera_id != self.camera.id:
            return
        stick_widget = next(filter(lambda sw: sw.stick.id == stick.id, self.stick_widgets))
        self.disconnect_stick_widget_signals(stick_widget)
        self.stick_widgets.remove(stick_widget)
        stick_widget.setParentItem(None)
        self.scene().removeItem(stick_widget)
        stick_widget.deleteLater()
        self.update()

    def handle_sticks_removed(self, sticks: List[Stick]):
        if sticks[0].camera_id != self.camera.id:
            return
        for stick in sticks:
            to_remove: StickWidget = None
            for sw in self.stick_widgets:
                if sw.stick.id == stick.id:
                    to_remove = sw
                    break
            self.stick_widgets.remove(to_remove)
            to_remove.setParentItem(None)
            if self.scene() is not None:
                self.scene().removeItem(to_remove)
            to_remove.deleteLater()
        self.update()

    def handle_sticks_added(self, sticks: List[Stick]):
        if len(sticks) == 0:
            return
        if sticks[0].camera_id != self.camera.id:
            return
        for stick in sticks:
            sw = StickWidget(stick, self.camera, self)
            sw.set_mode(self.stick_widget_mode)
            self.connect_stick_widget_signals(sw)
            self.stick_widgets.append(sw)
        self.update_stick_box()
        self.stick_widgets_out_of_sync.emit(self)
        self.update()

    def connect_stick_widget_signals(self, stick_widget: StickWidget):
        stick_widget.delete_clicked.connect(self.handle_stick_widget_delete_clicked)
        stick_widget.stick_changed.connect(self.handle_stick_widget_changed)
        stick_widget.link_initiated.connect(self.handle_stick_link_initiated)
        stick_widget.right_clicked.connect(self.handle_stick_widget_context_menu)

    def disconnect_stick_widget_signals(self, stick_widget: StickWidget):
        stick_widget.delete_clicked.disconnect(self.handle_stick_widget_delete_clicked)
        stick_widget.stick_changed.disconnect(self.handle_stick_widget_changed)
        stick_widget.link_initiated.disconnect(self.handle_stick_link_initiated)
        stick_widget.right_clicked.disconnect(self.handle_stick_widget_context_menu)

    def handle_stick_widget_delete_clicked(self, stick: Stick):
        self.camera.remove_stick(stick)

    def set_stick_widgets_mode(self, mode: StickMode):
        self.stick_widget_mode = mode
        for sw in self.stick_widgets:
            sw.set_mode(mode)
        self.set_stick_edit_mode(mode == StickMode.Edit)

    def handle_stick_widget_changed(self, stick_widget: StickWidget):
        self.camera.stick_changed.emit(stick_widget.stick)

    def handle_stick_changed(self, stick: Stick):
        if stick.camera_id != self.camera.id:
            return
        sw = next(filter(lambda _sw: _sw.stick.id == stick.id, self.stick_widgets))
        sw.adjust_line()
        sw.update_tooltip()
        #self.stick_length_btn.set_label(str(stick.length_cm) + " cm")
        #self.stick_length_btn.fit_to_contents()
        #self.layout_title_area()

    def handle_stick_link_initiated(self, stick_widget: StickWidget):
        self.stick_link_requested.emit(stick_widget)

    def handle_stick_length_clicked(self):
        #self.stick_length_input.setVisible(self.stick_length_btn.is_on())
        rect = self.scene().views()[0].size()
        #self.stick_length_input.setPos(rect.width() * 0.5 - self.stick_length_input.boundingRect().width() * 0.0,
        #                                       rect.height() * 0.5 - self.stick_length_input.boundingRect().height() * 0.0)
        #self.stick_length_input.set_focus()
        self.startTimer(1000)

    #def handle_stick_length_input_entered(self):
    #    length = int(self.stick_length_input.get_value())
    #    #self.stick_length_btn.set_label(str(length) + " cm")
    #    #self.stick_length_btn.fit_to_contents()
    #    self.layout_title_area()
    #    #self.stick_length_btn.click_button(artificial_emit=True)
    #    for stick in self.camera.sticks:
    #        stick.length_cm = length
    #    self.camera.stick_changed.emit(self.camera.sticks[0]) #TODO handle empty stick list

    def handle_stick_length_input_cancelled(self):
        #old_length = int(self.stick_length_btn.label.text()[:-3])
        #self.stick_length_input.set_length(old_length)
        #self.layout_title_area()
        #self.stick_length_btn.click_button(artificial_emit=True)
        pass

    def get_top_left(self) -> QPointF:
        return self.sceneBoundingRect().topLeft()

    def get_top_right(self) -> QPointF:
        return self.sceneBoundingRect().topRight()

    def highlight(self, color: Optional[QColor]):
        if color is None:
            self.highlight_animation.stop()
            self.highlight_rect.setVisible(False)
            return
        alpha = color.alpha()
        color.setAlpha(0)
        self.highlight_animation.setStartValue(color)
        self.highlight_animation.setEndValue(color)
        color.setAlpha(alpha)
        self.highlight_animation.setKeyValueAt(0.5, color)
        self.highlight_animation.setDuration(2000)
        self.highlight_animation.setLoopCount(-1)
        self.highlight_rect.setPen(QPen(color))
        self.highlight_rect.setVisible(True)
        self.highlight_animation.start()

    @pyqtProperty(QColor)
    def highlight_color(self) -> QColor:
        return self.current_highlight_color

    @highlight_color.setter
    def highlight_color(self, color: QColor):
        self.current_highlight_color = color

    def handle_highlight_color_changed(self, color: QColor):
        self.highlight_rect.setBrush(QBrush(color))
        self.update()

    def handle_stick_widget_context_menu(self, sender: Dict[str, StickWidget]):
        self.stick_context_menu.emit(sender['stick_widget'], self)

    def show_overlay_message(self, msg: Optional[str]):
        if msg is None:
            self.overlay_message.setVisible(False)
            self.blur_eff.setEnabled(False)
            return
        self.overlay_message.setText(msg)
        self.overlay_message.setPos(self.pixmap.boundingRect().center() - QPointF(
            0.5 * self.overlay_message.boundingRect().width(),
            0.5 * self.overlay_message.boundingRect().height()
        ))
        self.overlay_message.setVisible(True)
        self.blur_eff.setEnabled(True)

    def show_status_message(self, msg: Optional[str]):
        if msg is None:
            self.control_widget.set_title_text(self.camera.folder.name)
        else:
            self.control_widget.set_title_text(msg)

    def update_stick_box(self):
        left = 9000
        right = 0
        top = 9000
        bottom = -1

        for stick in self.camera.sticks:
            left = min(left, min(stick.top[0], stick.bottom[0]))
            right = max(right, max(stick.top[0], stick.bottom[0]))
            top = min(top, min(stick.top[1], stick.bottom[1]))
            bottom = max(bottom, max(stick.top[1], stick.bottom[1]))
        left -= 100
        right += 100
        top -= 100
        bottom += 100
        self.stick_box.setRect(left, top,
                               right - left, bottom - top)
        pen = QPen(QColor(0, 100, 200, 200))
        pen.setWidth(2)
        pen.setStyle(Qt.DashLine)
        self.stick_box.setPen(pen)

    def set_stick_edit_mode(self, is_edit: bool):
        if is_edit:
            self.update_stick_box()
            self.stick_box_start_pos = self.stick_box.pos()
            for sw in self.stick_widgets:
                sw.setParentItem(self.stick_box)
        else:
            offset = self.stick_box.pos() - self.stick_box_start_pos
            for sw in self.stick_widgets:
                stick = sw.stick
                stick.translate(np.array([int(offset.x()), int(offset.y())]))
                sw.setParentItem(self)
                sw.set_stick(stick)
            self.stick_box.setParentItem(None)
            self.stick_box = QGraphicsRectItem(self)
            self.stick_box.setFlag(QGraphicsItem.ItemIsMovable, True)
            self.stick_box.setVisible(False)
        self.stick_box.setVisible(is_edit)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        pass

    def keyReleaseEvent(self, event: QKeyEvent) -> None:
        if event.key() in [Qt.Key_Right, Qt.Key_Tab, Qt.Key_Space]:
            self.control_widget.next_photo_btn.click_button(True)
        elif event.key() in [Qt.Key_Left]:
            self.control_widget.prev_photo_btn.click_button(True)
        elif event.key() == Qt.Key_S:
            self.enter_pressed.emit()
