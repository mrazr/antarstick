from typing import Optional

import PyQt5
from PyQt5.QtCore import QLine, QLineF, QRect, Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QBrush, QColor, QFont, QPainter, QPen
from PyQt5.QtWidgets import (QGraphicsObject, QGraphicsItem, QGraphicsLineItem, QGraphicsRectItem,
                             QStyleOptionGraphicsItem, QGraphicsTextItem, QGraphicsSceneMouseEvent, QGraphicsSceneHoverEvent)

from stick import Stick
from PyQt5.Qt import QPoint, QPointF
import numpy as np

from camera_processing.widgets.button import Button

class StickWidget(QGraphicsObject):

    font: QFont = QFont("monospace", 16)

    delete_clicked = pyqtSignal(int)

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

        self.edit_mode = False

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
        self.adjust_handles()
        self.setAcceptHoverEvents(True)
        self.top_handle.setZValue(4)
        self.bottom_handle.setZValue(4)
        self.mid_handle.setZValue(4)

        self.top_handle.hide()
        self.mid_handle.hide()
        self.bottom_handle.hide()

        self.handle_mouse_offset = QPointF(0, 0)
        
    @pyqtSlot()
    def handle_btn_delete_clicked(self):
        self.btn_delete.deleteLater()
        self.delete_clicked.emit(self.stick.id)
    
    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem,
              widget: Optional[PyQt5.QtWidgets.QWidget] = ...):
        pen = QPen(QColor(0, 125, 125, 255))
        pen.setStyle(Qt.DotLine)
        pen.setWidth(1.0)
        painter.setPen(pen)
       # painter.setBrush(QBrush(QColor(0, 125, 125, 255)))
       # rect = QRect(0, 0, 5, 5)
       # rect.moveCenter(self.line.p1())
       # painter.fillRect(rect, QColor(0, 125, 125, 125))
       # rect.moveCenter(self.line.p2())
       # painter.fillRect(rect, QColor(0, 125, 125, 125))
        painter.drawLine(self.line.p1(), self.line.p2())


    def boundingRect(self) -> PyQt5.QtCore.QRectF:
        return self.gline.boundingRect().united(self.top_handle.boundingRect()).united(self.mid_handle.boundingRect()).united(self.bottom_handle.boundingRect())
    
    def set_edit_mode(self, value: bool):
        self.edit_mode = value
        self.btn_delete.setVisible(self.edit_mode)
        self.top_handle.setVisible(self.edit_mode)
        self.mid_handle.setVisible(self.edit_mode)
        self.bottom_handle.setVisible(self.edit_mode)

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent):
        if not self.edit_mode or self.hovered_handle is None:
            return
        self.hovered_handle.setBrush(self.handle_press_brush)
        self.hovered_handle.setPen(self.handle_press_pen)
        self.handle_mouse_offset = self.hovered_handle.rect().center() - event.pos()

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent):
        if not self.edit_mode or self.hovered_handle is None:
            return
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
            displace = event.pos() - event.lastPos()
            self.setPos(self.pos() + displace)
        self.adjust_handles()
        self.adjust_stick()
        self.scene().update()

        
    def hoverEnterEvent(self, event: QGraphicsSceneHoverEvent):
        pass

    def hoverLeaveEvent(self, event: QGraphicsSceneHoverEvent):
        for h in self.handles:
            h.setBrush(self.handle_idle_brush)
        self.scene().update()
    
    def hoverMoveEvent(self, event: QGraphicsSceneHoverEvent):
        if not self.edit_mode:
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
        vec = 0.5 * (self.line.p1() - self.line.p2())
        self.stick.top[0] = self.pos().x() + self.line.p1().x()
        self.stick.top[1] = self.pos().y() + self.line.p1().y()
        self.stick.bottom[0] = self.pos().x() - self.line.p2().x()
        self.stick.bottom[1] = self.pos().y() - self.line.p2().y()

    def adjust_handles(self):
        if self.line.p1().y() > self.line.p2().y():
            p1, p2 = self.line.p1(), self.line.p2()
            self.line.setP1(p2)
            self.line.setP2(p1)
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