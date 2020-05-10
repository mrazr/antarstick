from enum import Enum
from multiprocessing import Process
from typing import Any, Callable, List, Optional

import numpy as np
import PyQt5
from PyQt5.Qt import QMimeData, QPoint, QPointF, QRectF, QGraphicsDropShadowEffect
from PyQt5.QtCore import QLine, QLineF, QRect, Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QBrush, QColor, QDrag, QFont, QPainter, QPen
from PyQt5.QtWidgets import (QGraphicsEllipseItem, QGraphicsItem,
                             QGraphicsLineItem, QGraphicsObject,
                             QGraphicsRectItem, QGraphicsSceneDragDropEvent,
                             QGraphicsSceneHoverEvent,
                             QGraphicsSceneMouseEvent, QGraphicsTextItem,
                             QStyleOptionGraphicsItem)

from camera import Camera
from camera_processing.widgets.button import Button
from dataset import Dataset
from stick import Stick


class StickMode(Enum):
    DISPLAY = 0
    EDIT = 1
    LINK = 2

class StickWidget(QGraphicsObject):

    font: QFont = QFont("monospace", 16)

    delete_clicked = pyqtSignal(Stick)
    link_initiated = pyqtSignal('PyQt_PyObject') # Actually StickWidget
    link_accepted = pyqtSignal('PyQt_PyObject')
    hovered = pyqtSignal(['PyQt_PyObject', 'PyQt_PyObject'])

    handle_idle_brush = QBrush(QColor(0, 125, 125, 100))
    handle_hover_brush = QBrush(QColor(125, 125, 0, 150))
    handle_press_brush = QBrush(QColor(200, 200, 0, 0))
    handle_idle_pen = QPen(QColor(0, 0, 0, 255))
    handle_press_pen = QPen(QColor(200, 200, 0, 255))
    handle_size = 8

    def __init__(self, stick: Stick, parent: Optional[QGraphicsItem] = None):
        QGraphicsObject.__init__(self, parent)
        self.setPos(QPointF(0.5 * (stick.top[0] + stick.bottom[0]), 0.5 * (stick.top[1] + stick.bottom[1])))
        self.stick = stick
        vec = 0.5 * (stick.top - stick.bottom)
        self.line = QLine(vec[0], vec[1], -vec[0], -vec[1])
        self.gline = QGraphicsLineItem(QLineF(self.line))
        self.stick_label_text = QGraphicsTextItem(stick.label, self)
        self.stick_label_text.setFont(StickWidget.font)
        self.stick_label_text.setPos(-QPointF(0, self.stick_label_text.boundingRect().height()))
        self.stick_label_text.hide()
        self.setZValue(3)

        self.mode = StickMode.DISPLAY

        self.btn_delete = Button("x", self)
        self.btn_delete.set_base_color("red")
        self.btn_delete.setVisible(False)
        btn_size = max(int(np.linalg.norm(self.stick.top - self.stick.bottom) / 5.0), 15)
        self.btn_delete.set_height(btn_size)
        self.btn_delete.set_width(btn_size)
        self.btn_delete.clicked.connect(self.handle_btn_delete_clicked)
        self.btn_delete.setPos(self.line.p1() - QPointF(0.5 * self.btn_delete.boundingRect().width(), 1.1 * self.btn_delete.boundingRect().height()))

        self.top_handle = QGraphicsRectItem(0, 0, self.handle_size, self.handle_size, self)
        self.mid_handle = QGraphicsRectItem(0, 0, self.handle_size, self.handle_size, self)
        self.bottom_handle = QGraphicsRectItem(0, 0, self.handle_size, self.handle_size, self)
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

        self.link_button = Button("Link to...", self)
        self.link_button.clicked.connect(lambda: self.link_initiated.emit(self))
        self.link_button.set_height(15)

        self.adjust_handles()
        self.setAcceptHoverEvents(True)
        self.top_handle.setZValue(4)
        self.bottom_handle.setZValue(4)
        self.mid_handle.setZValue(4)

        self.top_handle.hide()
        self.mid_handle.hide()
        self.bottom_handle.hide()

        self.handle_mouse_offset = QPointF(0, 0)
        self.available_for_linking = False
        self.highlight_color: QColor = None


    @pyqtSlot()
    def handle_btn_delete_clicked(self):
        self.btn_delete.deleteLater()
        self.delete_clicked.emit(self.stick)
    
    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem,
              widget: Optional[PyQt5.QtWidgets.QWidget] = ...):
            
        if self.highlight_color is not None:
            brush = QBrush(self.highlight_color)
            pen = QPen(brush, 4)
            painter.setPen(pen)
            painter.drawLine(self.line.p1(), self.line.p2())

        pen = QPen(QColor(0, 125, 125, 255))
        pen.setStyle(Qt.DotLine)
        pen.setWidth(1.0)
        painter.setPen(pen)
        painter.drawLine(self.line.p1(), self.line.p2())


    def boundingRect(self) -> PyQt5.QtCore.QRectF:
        return self.gline.boundingRect().united(self.top_handle.boundingRect()).united(self.mid_handle.boundingRect()).united(self.bottom_handle.boundingRect())
    
    def set_edit_mode(self, value: bool):
        if value:
            self.set_mode(StickMode.EDIT)
        else:
            self.set_mode(StickMode.DISPLAY)
    
    def set_mode(self, mode: StickMode):
        if mode == StickMode.DISPLAY:
            self.btn_delete.setVisible(False)
            self.top_handle.setVisible(False)
            self.mid_handle.setVisible(False)
            self.bottom_handle.setVisible(False)
            self.link_button.setVisible(False)
            self.available_for_linking = False
        elif mode == StickMode.EDIT:
            self.set_mode(StickMode.DISPLAY)
            self.btn_delete.setVisible(True)
            self.top_handle.setVisible(True)
            self.mid_handle.setVisible(True)
            self.bottom_handle.setVisible(True)
        else:
            self.set_mode(StickMode.DISPLAY)
            self.link_button.setVisible(True)
        self.mode = mode

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent):
        if self.mode != StickMode.EDIT:
            return
        if self.hovered_handle is None:
            return

        self.hovered_handle.setBrush(self.handle_press_brush)
        self.hovered_handle.setPen(self.handle_press_pen)
        self.handle_mouse_offset = self.hovered_handle.rect().center() - event.pos()

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent):
        if self.available_for_linking:
            self.link_accepted.emit(self)
            return

        if self.mode != StickMode.EDIT:
            return

        if self.hovered_handle is not None:
            self.hovered_handle.setBrush(self.handle_hover_brush)
            self.hovered_handle.setPen(self.handle_idle_pen)
    
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

        
    def hoverEnterEvent(self, event: QGraphicsSceneHoverEvent):
        if self.available_for_linking:
            self.hovered.emit(True, self)

    def hoverLeaveEvent(self, event: QGraphicsSceneHoverEvent):
        for h in self.handles:
            h.setBrush(self.handle_idle_brush)
        if self.available_for_linking:
            self.hovered.emit(False, self)
        self.scene().update()
    
    def hoverMoveEvent(self, event: QGraphicsSceneHoverEvent):
        if self.mode != StickMode.EDIT:
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
        self.btn_delete.setPos(self.line.p1() - QPointF(0.5 * self.btn_delete.boundingRect().width(), 1.1 * self.btn_delete.boundingRect().height()))
        self.link_button.setPos(self.bottom_handle.rect().bottomRight() - QPointF(self.link_button.boundingRect().width() / 2, 0))
    
    def set_available_for_linking(self, value: bool):
        self.available_for_linking = value
        if self.available_for_linking:
            self.set_highlight_color(QColor(0, 255, 0, 100))
        else:
            self.set_highlight_color(None)
    
    def set_highlight_color(self, color: Optional[QColor]):
        self.highlight_color = color
        self.update()
    
    