import PySide2
from PySide2.QtWidgets import QGraphicsItem, QStyleOptionGraphicsItem, QWidget, QGraphicsLineItem
from PySide2.QtCore import QLine, Qt, QRect
from PySide2.QtGui import QPainter, QPen, QColor, QBrush
from stick import Stick
from typing import Optional


class StickWidget(QGraphicsItem):
    def __init__(self, stick: Stick, parent: Optional[QGraphicsItem] = None):
        QGraphicsItem.__init__(self, parent)
        self.line = QLine(stick.top[0], stick.top[1],
                          stick.bottom[0], stick.bottom[1])
        self.gline = QGraphicsLineItem(self.line)

    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem,
              widget: Optional[PySide2.QtWidgets.QWidget] = ...):
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




    def boundingRect(self) -> PySide2.QtCore.QRectF:
        return self.gline.boundingRect()
