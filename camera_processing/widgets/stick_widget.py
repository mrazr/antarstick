from typing import Optional

import PyQt5
from PyQt5.QtCore import QLine, QLineF, QRect, Qt
from PyQt5.QtGui import QBrush, QColor, QPainter, QPen
from PyQt5.QtWidgets import (QGraphicsItem, QGraphicsLineItem,
                             QStyleOptionGraphicsItem)

from stick import Stick


class StickWidget(QGraphicsItem):
    def __init__(self, stick: Stick, parent: Optional[QGraphicsItem] = None):
        QGraphicsItem.__init__(self, parent)
        self.line = QLine(stick.top[0], stick.top[1],
                          stick.bottom[0], stick.bottom[1])
        self.gline = QGraphicsLineItem(QLineF(self.line))

    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem,
              widget: Optional[PyQt5.QtWidgets.QWidget] = ...):
        pen = QPen(QColor(0, 125, 125, 255))
        pen.setStyle(Qt.DotLine)
        pen.setWidth(2.0)
        painter.setPen(pen)
        painter.setBrush(QBrush(QColor(0, 125, 125, 255)))
        rect = QRect(0, 0, 5, 5)
        rect.moveCenter(self.line.p1())
        painter.fillRect(rect, QColor(0, 125, 125, 125))
        rect.moveCenter(self.line.p2())
        painter.fillRect(rect, QColor(0, 125, 125, 125))
        painter.drawLine(self.line.p1(), self.line.p2())




    def boundingRect(self) -> PyQt5.QtCore.QRectF:
        return self.gline.boundingRect()
