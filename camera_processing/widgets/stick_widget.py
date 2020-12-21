from enum import Enum
from typing import Optional, Any, Dict

import PyQt5
import numpy as np
from PyQt5.Qt import (QPointF)
from PyQt5.QtCore import QLine, QLineF, Qt, pyqtSignal, pyqtSlot, QMarginsF, QPropertyAnimation, pyqtProperty, QPoint
from PyQt5.QtGui import QBrush, QColor, QFont, QPainter, QPen, QStaticText
from PyQt5.QtWidgets import (QGraphicsEllipseItem, QGraphicsItem,
                             QGraphicsLineItem, QGraphicsObject,
                             QGraphicsRectItem, QGraphicsSceneHoverEvent,
                             QGraphicsSceneMouseEvent, QGraphicsTextItem,
                             QStyleOptionGraphicsItem, QGraphicsSimpleTextItem, QGraphicsSceneContextMenuEvent)

from camera_processing.widgets.button import Button, ButtonColor
from stick import Stick


class StickMode(Enum):
    Display = 0
    Edit = 1
    EditDelete = 2
    LinkSource = 3
    LinkTarget = 4
    Measurement = 5


class StickWidget(QGraphicsObject):

    font: QFont = QFont("monospace", 32)

    delete_clicked = pyqtSignal(Stick)
    link_initiated = pyqtSignal('PyQt_PyObject') # Actually StickWidget
    link_accepted = pyqtSignal('PyQt_PyObject')
    hovered = pyqtSignal(['PyQt_PyObject', 'PyQt_PyObject'])
    stick_changed = pyqtSignal('PyQt_PyObject')
    sibling_changed = pyqtSignal(bool)
    right_clicked = pyqtSignal('PyQt_PyObject')

    handle_idle_brush = QBrush(QColor(0, 125, 125, 50))
    handle_hover_brush = QBrush(QColor(125, 125, 0, 50))
    handle_press_brush = QBrush(QColor(200, 200, 0, 0))
    handle_idle_pen = QPen(QColor(0, 0, 0, 255))
    handle_press_pen = QPen(QColor(200, 200, 0, 255))
    handle_size = 20

    normal_color = QColor(0, 200, 120)
    negative_color = QColor(200, 0, 0)
    positive_color = QColor(0, 200, 0)

    def __init__(self, stick: Stick, parent: Optional[QGraphicsItem] = None):
        QGraphicsObject.__init__(self, parent)
        self.stick = stick
        self.line = QLineF()
        self.gline = QGraphicsLineItem(self.line)

        self.stick_label_text = QGraphicsSimpleTextItem("0", self)
        self.stick_label_text.setFont(StickWidget.font)
        #self.stick_label_text.setPos(-QPointF(0, self.stick_label_text.boundingRect().height()))
        self.stick_label_text.setPos(self.line.p1() - QPoint(0, 24))
        self.stick_label_text.setBrush(QBrush(QColor(0, 255, 0)))
        self.stick_label_text.hide()
        #self.stick_label_text = QStaticText(self.stick.label)
        self.setZValue(10)

        self.mode = StickMode.Display

        self.btn_delete = Button("delete", "x", parent=self)
        self.btn_delete.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
        self.btn_delete.set_base_color([ButtonColor.RED])
        self.btn_delete.setVisible(False)
        btn_size = max(int(np.linalg.norm(self.stick.top - self.stick.bottom) / 5.0), 15)
        self.btn_delete.set_height(12)
        #self.btn_delete.set_height(btn_size)
        #self.btn_delete.set_width(btn_size)
        self.btn_delete.clicked.connect(self.handle_btn_delete_clicked)
        self.btn_delete.setPos(self.line.p1() - QPointF(0.5 * self.btn_delete.boundingRect().width(), 1.1 * self.btn_delete.boundingRect().height()))
        self.btn_delete.set_opacity(0.7)

        #self.top_handle = QGraphicsRectItem(0, 0, self.handle_size, self.handle_size, self)
        #self.mid_handle = QGraphicsRectItem(0, 0, self.handle_size, self.handle_size, self)
        #self.bottom_handle = QGraphicsRectItem(0, 0, self.handle_size, self.handle_size, self)

        self.top_handle = QGraphicsEllipseItem(0, 0, self.handle_size, self.handle_size, self)
        self.mid_handle = QGraphicsEllipseItem(0, 0, self.handle_size, self.handle_size, self)
        self.bottom_handle = QGraphicsEllipseItem(0, 0, self.handle_size, self.handle_size, self)

        self.top_handle.setAcceptedMouseButtons(Qt.NoButton)
        self.mid_handle.setAcceptedMouseButtons(Qt.NoButton)
        self.bottom_handle.setAcceptedMouseButtons(Qt.NoButton)
        self.top_handle.setBrush(self.handle_idle_brush)
        self.top_handle.setPen(self.handle_idle_pen)
        self.mid_handle.setBrush(self.handle_idle_brush)
        self.mid_handle.setPen(self.handle_idle_pen)
        self.bottom_handle.setBrush(self.handle_idle_brush)
        self.bottom_handle.setPen(self.handle_idle_pen)

        self.hovered_handle: Optional[QGraphicsRectItem] = None
        self.handles = [self.top_handle, self.mid_handle, self.bottom_handle]

        self.link_button = Button("link", "Link to...", parent=self)
        self.link_button.set_base_color([ButtonColor.GREEN])
        self.link_button.set_height(12)
        self.link_button.set_label("Link", direction="vertical")
        self.link_button.fit_to_contents()
        self.link_button.clicked.connect(lambda: self.link_initiated.emit(self))
        self.link_button.setVisible(False)
        self.link_button.setFlag(QGraphicsObject.ItemIgnoresTransformations, False)

        self.adjust_line()

        self.setAcceptHoverEvents(True)
        self.top_handle.setZValue(4)
        self.bottom_handle.setZValue(4)
        self.mid_handle.setZValue(4)

        self.top_handle.hide()
        self.mid_handle.hide()
        self.bottom_handle.hide()

        self.handle_mouse_offset = QPointF(0, 0)
        self.available_for_linking = False
        self.link_source = False
        self.current_highlight_color: QColor = StickWidget.normal_color
        self.highlighted = False
        self.frame_color: Optional[None] = self.normal_color
        self.is_linked = False

        self.is_master = True
        self.selected = False

        self.measured_height: int = -1
        self.current_color = self.normal_color

        self.show_label = False
        self.highlight_animation = QPropertyAnimation(self, b"highlight_color")
        self.highlight_animation.valueChanged.connect(self.handle_highlight_animation_value_changed)
        self.deleting = False
        self.update_tooltip()
        self.show_measurements: bool = False
        self.proposed_snow_height: int = -1

    @pyqtSlot()
    def handle_btn_delete_clicked(self):
        self.delete_clicked.emit(self.stick)

    def prepare_for_deleting(self):
        self.deleting = True
        self.highlight_animation.stop()
        self.btn_delete.setParentItem(None)
        self.scene().removeItem(self.btn_delete)
        self.btn_delete.deleteLater()

    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem,
              widget: Optional[PyQt5.QtWidgets.QWidget] = ...):
        painter.setPen(QPen(self.current_color, 1.0))

        #if self.current_highlight_color is not None:
        brush = QBrush(self.current_highlight_color)
        pen = QPen(brush, 4)
        painter.setPen(pen)
        #painter.drawLine(self.line.p1(), self.line.p2())
        if self.highlighted:
            painter.fillRect(self.boundingRect(), QBrush(self.current_highlight_color))

        if self.frame_color is not None and self.mode != StickMode.Edit and self.mode != StickMode.EditDelete:
            painter.setPen(QPen(self.frame_color, 4))
            painter.drawRect(self.boundingRect())

        #painter.drawRect(self.boundingRect().marginsAdded(QMarginsF(5, 5, 5, 5)))

        pen = QPen(QColor(0, 255, 0, 255))

        pen.setWidth(1.0)
        pen.setColor(QColor(255, 0, 255, 255))
        pen.setStyle(Qt.DotLine)
        painter.setPen(pen)
        off = 10
        painter.drawLine(self.line.p1() - QPointF(0, off), self.line.p1() + QPointF(0, off))
        painter.drawLine(self.line.p1() - QPointF(off, 0), self.line.p1() + QPointF(off, 0))
        painter.drawLine(self.line.p2() - QPointF(0, off), self.line.p2() + QPointF(0, off))
        painter.drawLine(self.line.p2() - QPointF(off, 0), self.line.p2() + QPointF(off, 0))
        pen.setStyle(Qt.SolidLine)
        pen.setColor(QColor(0, 255, 0, 255))
        painter.setPen(pen)

        if self.mode != StickMode.EditDelete:
            #pen.setStyle(Qt.DotLine)
            pen.setWidth(2.0)
            #painter.setPen(pen)
            #painter.drawEllipse(self.line.p1(), 6, 6)
            #painter.drawEllipse(self.line.p2(), 6, 6)
            br = painter.brush()
            #painter.setBrush(QBrush(QColor(255, 0, 0, 60)))
            painter.setPen(pen)
            painter.drawEllipse(self.line.p1(), 10, 10)
            painter.drawEllipse(self.line.p2(), 10, 10)
            painter.setBrush(br)

            if self.mode == StickMode.Measurement and self.proposed_snow_height >= 0:
                point = QPointF(self.boundingRect().x(), -self.proposed_snow_height + self.line.p2().y())
                pen = QPen(QColor(200, 100, 0, 255), 3.0)
                painter.setPen(pen)
                painter.drawLine(point,
                                 point + QPointF(self.boundingRect().width(), 0.0))

            if self.measured_height >= 0:
                vec = (self.stick.top - self.stick.bottom) / np.linalg.norm(self.stick.top - self.stick.bottom)
                dist_along_stick = self.measured_height / np.dot(np.array([0.0, -1.0]), vec)
                point = self.line.p2() + dist_along_stick * QPointF(vec[0], vec[1])
                point = QPointF(self.boundingRect().x(), point.y())
                pen = QPen(QColor(0, 100, 200, 255), 3.0)
                painter.setPen(pen)
                painter.drawLine(point,
                                 point + QPointF(self.boundingRect().width(), 0.0))
        else:
            painter.drawLine(self.line.p1(), self.line.p2())

        if self.selected:
            pen.setColor(QColor(255, 125, 0, 255))
            pen.setStyle(Qt.DashLine)
            painter.setPen(pen)
            painter.drawRect(self.boundingRect().marginsAdded(QMarginsF(5, 5, 5, 5)))

        if self.show_measurements:
            #painter.setFont(StickWidget.font)
            #painter.drawStaticText(self.line.p2(), self.stick_label_text)
            painter.fillRect(self.stick_label_text.boundingRect().translated(self.stick_label_text.pos()),
                             QBrush(QColor(0, 0, 0, 120)))

    def boundingRect(self) -> PyQt5.QtCore.QRectF:
        return self.gline.boundingRect().united(self.top_handle.boundingRect()).\
            united(self.mid_handle.boundingRect()).united(self.bottom_handle.boundingRect())
    
    def set_edit_mode(self, value: bool):
        if value:
            self.set_mode(StickMode.EditDelete)
        else:
            self.set_mode(StickMode.Display)
    
    def set_mode(self, mode: StickMode):
        if mode == StickMode.Display:
            self.btn_delete.setVisible(False)
            self.top_handle.setVisible(False)
            self.mid_handle.setVisible(False)
            self.bottom_handle.setVisible(False)
            self.link_button.setVisible(False)
            self.available_for_linking = False
            self.link_source = False
        elif mode == StickMode.EditDelete:
            self.set_mode(StickMode.Display)
            self.top_handle.setVisible(True)
            self.mid_handle.setVisible(True)
            self.bottom_handle.setVisible(True)
            self.available_for_linking = False
            self.link_source = False
            self.btn_delete.setVisible(True)
        elif mode == StickMode.LinkSource:
            #self.set_mode(StickMode.Display)
            #self.link_button.setVisible(True)
            self.set_mode(StickMode.Display)
            self.link_source = True
            self.available_for_linking = False
            self.link_button.setPos(self.boundingRect().topLeft())
            self.link_button.set_width(int(self.boundingRect().width()))
            self.link_button.set_button_height(int(self.boundingRect().height()))
            self.link_button.adjust_text_to_button()
        elif mode == StickMode.LinkTarget:
            self.set_mode(StickMode.Display)
            self.link_source = False
            self.available_for_linking = True
        elif mode == StickMode.Edit:
            self.set_mode(StickMode.EditDelete)
            self.btn_delete.setVisible(False)

        self.mode = mode
        self.update_tooltip()
        self.update()

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent):
        if self.mode != StickMode.EditDelete:
            return
        if self.hovered_handle is None:
            return

        self.hovered_handle.setBrush(self.handle_press_brush)
        if self.hovered_handle == self.mid_handle:
            self.bottom_handle.setBrush(self.handle_press_brush)
            self.bottom_handle.setPen(self.handle_press_pen)
            self.bottom_handle.setOpacity(0.5)
            self.top_handle.setBrush(self.handle_press_brush)
            self.top_handle.setPen(self.handle_press_pen)
            self.top_handle.setOpacity(0.5)
        self.hovered_handle.setPen(self.handle_press_pen)
        self.hovered_handle.setOpacity(0.5)
        self.handle_mouse_offset = self.hovered_handle.rect().center() - event.pos()
        self.btn_delete.setVisible(False)

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent):
        if self.available_for_linking:
            self.link_accepted.emit(self)
            return

        if self.mode == StickMode.Measurement:
            self.measured_height = self.proposed_snow_height
            self.stick.set_snow_height_px(self.proposed_snow_height)
            self.proposed_snow_height = -1

        if self.mode != StickMode.EditDelete and self.mode != StickMode.Edit:
            return

        if self.hovered_handle is not None:
            self.hovered_handle.setBrush(self.handle_hover_brush)
            self.hovered_handle.setPen(self.handle_idle_pen)
            self.hovered_handle.setOpacity(1.0)
            if self.hovered_handle == self.mid_handle:
                self.bottom_handle.setBrush(self.handle_idle_brush)
                self.bottom_handle.setPen(self.handle_idle_pen)
                self.bottom_handle.setOpacity(1.0)
                self.top_handle.setBrush(self.handle_idle_brush)
                self.top_handle.setPen(self.handle_idle_pen)
                self.top_handle.setOpacity(1.0)
            self.stick_changed.emit(self)
        self.hovered_handle = None
        if self.mode == StickMode.EditDelete:
            self.btn_delete.setVisible(True)
    
    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent):
        if self.hovered_handle is None:
            return
        if self.hovered_handle == self.top_handle:
            self.line.setP1((event.pos() + self.handle_mouse_offset).toPoint())
        elif self.hovered_handle == self.bottom_handle:
            self.line.setP2((event.pos() + self.handle_mouse_offset).toPoint())
        else:
            displacement = event.pos() - event.lastPos()
            self.setPos(self.pos() + displacement)
        self.adjust_handles()
        self.adjust_stick()
        self.scene().update()

    def set_top(self, pos: QPoint):
        self.line.setP1(pos)
        self.adjust_handles()
        self.adjust_stick()
        self.scene().update()

    def set_bottom(self, pos: QPoint):
        self.line.setP2(pos)
        self.adjust_handles()
        self.adjust_stick()
        self.scene().update()

    def hoverEnterEvent(self, event: QGraphicsSceneHoverEvent):
        if self.available_for_linking:
            self.hovered.emit(True, self)
        elif self.link_source:
            self.link_button.setVisible(True)
        #self.show_label = True
        #self.stick_label_text.show()
        self.scene().update()

    def hoverLeaveEvent(self, event: QGraphicsSceneHoverEvent):
        for h in self.handles:
            h.setBrush(self.handle_idle_brush)
        self.hovered_handle = None
        if self.available_for_linking:
            self.hovered.emit(False, self)
        self.link_button.setVisible(False)
        #self.show_label = False
        #self.stick_label_text.hide()
        self.proposed_snow_height = -1
        self.scene().update()
    
    def hoverMoveEvent(self, event: QGraphicsSceneHoverEvent):
        if self.mode != StickMode.EditDelete and self.mode != StickMode.Edit and self.mode != StickMode.Measurement:
            return
        if self.mode == StickMode.Measurement:
            self.proposed_snow_height = max(self.line.p2().y() - event.pos().y(), 0)
            self.update()
            return
        hovered_handle = list(filter(lambda h: h.rect().contains(event.pos()), self.handles))
        if len(hovered_handle) == 0:
            if self.hovered_handle is not None:
                self.hovered_handle.setBrush(self.handle_idle_brush)
                self.hovered_handle = None
            return
        if self.hovered_handle is not None and self.hovered_handle != hovered_handle[0]:
            self.hovered_handle.setBrush(self.handle_idle_brush)
        self.hovered_handle = hovered_handle[0]
        if self.hovered_handle == self.top_handle:
            self.top_handle.setBrush(self.handle_hover_brush)
        elif self.hovered_handle == self.bottom_handle:
            self.bottom_handle.setBrush(self.handle_hover_brush)
        else:
            self.mid_handle.setBrush(self.handle_hover_brush)

        self.scene().update()
    
    def adjust_stick(self):
        self.stick.top[0] = self.pos().x() + self.line.p1().x()
        self.stick.top[1] = self.pos().y() + self.line.p1().y()
        self.stick.bottom[0] = self.pos().x() + self.line.p2().x()
        self.stick.bottom[1] = self.pos().y() + self.line.p2().y()

    def adjust_handles(self):
        if self.line.p1().y() > self.line.p2().y():
            p1, p2 = self.line.p1(), self.line.p2()
            self.line.setP1(p2)
            self.line.setP2(p1)
            if self.hovered_handle is not None:
                self.hovered_handle.setBrush(self.handle_idle_brush)
                self.hovered_handle.setPen(self.handle_idle_pen)
                self.hovered_handle = self.top_handle if self.hovered_handle == self.bottom_handle else self.bottom_handle
                self.hovered_handle.setBrush(self.handle_press_brush)
                self.hovered_handle.setPen(self.handle_press_pen)
        rect = self.top_handle.rect()
        rect.moveCenter(self.line.p1())
        self.top_handle.setRect(rect)
        rect = self.bottom_handle.rect()
        rect.moveCenter(self.line.p2())
        self.bottom_handle.setRect(rect)
        rect = self.mid_handle.rect()
        rect.moveCenter(self.line.center())
        self.mid_handle.setRect(rect)
        #self.btn_delete.setPos(self.line.p1() - QPointF(0.5 * self.btn_delete.boundingRect().width(), 1.1 * self.btn_delete.boundingRect().height()))
        self.btn_delete.setPos(self.top_handle.rect().center() - QPointF(self.btn_delete.boundingRect().width() / 2,
                                                               self.btn_delete.boundingRect().height() + self.top_handle.boundingRect().height() / 2))
        #self.link_button.setPos(self.bottom_handle.rect().bottomRight() - QPointF(self.link_button.boundingRect().width() / 2, 0))
    
    def set_available_for_linking(self, available: bool):
        self.available_for_linking = available
        #if not self.is_linked:
        #    #if self.available_for_linking:
        #    #    #self.set_highlight_color(QColor(0, 255, 0, 100))
        #    #    #self.highlight(QColor(255, 125, 0, 200), animated=True)
        #    #else:
        #    #    #self.set_highlight_color(None)
        #    #    #self.highlight(None)

    def set_is_link_source(self, is_source: bool):
        #if is_source:
        #    self.highlight(QColor(255, 125, 0, 100), animated=True)
        #else:
        #    self.highlight(None)
        self.link_source = is_source
        self.link_button.setPos(self.boundingRect().topLeft())
        self.link_button.set_width(int(self.boundingRect().width()))
        self.link_button.set_button_height(int(self.boundingRect().height()))
        self.link_button.adjust_text_to_button()
    
    def set_frame_color(self, color: Optional[QColor]):
        self.frame_color = color if color is not None else self.normal_color
        self.update()

    def set_is_linked(self, value: bool):
        self.is_linked = value
        if not self.is_linked:
            self.set_frame_color(None)
            if self.available_for_linking:
                self.highlight(QColor(0, 255, 0, 100))
            else:
                self.highlight(None)
        self.update_tooltip()

    def adjust_line(self):
        self.setPos(QPointF(0.5 * (self.stick.top[0] + self.stick.bottom[0]), 0.5 * (self.stick.top[1] + self.stick.bottom[1])))
        vec = 0.5 * (self.stick.top - self.stick.bottom)
        self.line.setP1(QPointF(vec[0], vec[1]))
        self.line.setP2(-self.line.p1())
        self.gline.setLine(self.line)
        self.adjust_handles()
        self.stick_label_text.setPos(self.line.p1() - QPointF(0.5 * self.stick_label_text.boundingRect().width(),
                                                             1.3 * self.stick_label_text.boundingRect().height()))
        self.update()

    def set_selected(self, selected: bool):
        self.selected = selected
        self.update()

    def is_selected(self) -> bool:
        return self.selected

    def set_snow_height(self, height: int):
        self.measured_height = height
        self.update()

    def border_normal(self):
        self.current_color = self.normal_color
        self.update()

    def border_positive(self):
        self.current_color = self.positive_color
        self.update()

    def border_negative(self):
        self.current_color = self.negative_color
        self.update()

    @pyqtProperty(QColor)
    def highlight_color(self) -> QColor:
        return self.current_highlight_color

    @highlight_color.setter
    def highlight_color(self, color: QColor):
        self.current_highlight_color = color

    def highlight(self, color: Optional[QColor], animated: bool = False):
        self.highlighted = color is not None
        if not animated or color is None:
            self.highlight_animation.stop()
            self.current_highlight_color = self.normal_color if color is None else color
            self.update()
            return
        self.highlight_animation.setStartValue(color)
        self.highlight_animation.setEndValue(color)
        self.highlight_animation.setKeyValueAt(0.5, color.darker())
        self.highlight_animation.setDuration(2000)
        self.highlight_animation.setLoopCount(-1)
        self.highlight_animation.start()

    def handle_link_button_hovered(self, btn: Dict[str, Any]):
        self.link_button.setVisible(btn['hovered'])

    def handle_highlight_animation_value_changed(self, new: QColor):
        if not self.deleting:
            self.update(self.boundingRect().marginsAdded(QMarginsF(10, 10, 10, 10)))

    def contextMenuEvent(self, event: QGraphicsSceneContextMenuEvent) -> None:
        self.right_clicked.emit({'stick_widget': self})

    def set_stick_label(self, label: str):
        self.stick.label = label
        self.stick_label_text.setText(label)
        self.update_tooltip()
        self.update()

    def get_stick_label(self) -> str:
        return self.stick.label

    def get_stick_length_cm(self) -> int:
        return self.stick.length_cm

    def set_stick_length_cm(self, length: int):
        self.stick.length_cm = length
        self.update_tooltip()
        self.update()

    def update_tooltip(self):
        if self.mode != StickMode.Display or self.mode == StickMode.Measurement:
            self.setToolTip("")
            return
        snow_txt = "Snow height: "
        if self.stick.snow_height_px >= 0:
            snow_txt += str(self.stick.snow_height_cm) + " cm"
            self.stick_label_text.setText(str(self.stick.snow_height_cm))
            #self.stick_label_text.setVisible(True)
        else:
            snow_txt = "not measured"
            self.stick_label_text.setVisible(False)
        self.stick_label_text.setText(self.stick.label)
        self.stick_label_text.setVisible(True)
        stick_view_text = ''
        role = ''
        if self.stick.alternative_view is not None:
            alt_view = self.stick.alternative_view
            role = " - primary"
            alt = "Secondary"
            if not self.stick.primary:
                role = " - secondary"
                alt = "Primary"
            stick_view_text = f'\n{alt} view: {alt_view.label} in {alt_view.camera_folder.name}\n'
        mark = '*' if self.stick.determines_quality else ''
        self.setToolTip(f'{mark}{self.stick.label}{role}{stick_view_text}\nLength: {self.stick.length_cm} cm\n{snow_txt}')

    def set_stick(self, stick: Stick):
        self.stick = stick
        self.adjust_line()
        self.adjust_handles()
        self.set_snow_height(stick.snow_height_px)
        self.update_tooltip()
        self.setVisible(self.stick.is_visible)
        #self.line.setP1(QPoint(*self.stick.top))
        #self.line.setP2(QPoint(*self.stick.bottom))
        #self.adjust_line()

    def set_show_measurements(self, show: bool):
        self.show_measurements = show
        self.stick_label_text.setVisible(show)
        self.update()