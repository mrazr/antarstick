from PySide2.QtWidgets import QGraphicsEllipseItem, QGraphicsSceneMouseEvent
import PySide2
from PySide2.QtGui import QColor, QPen
from PySide2.QtCore import QRectF
from typing import Optional


class LinkCameraButton(QGraphicsEllipseItem):

    def __init__(self, radius: int = 30, parent: QGraphicsEllipseItem = None):
        QGraphicsEllipseItem.__init__(self, parent)
        self.radius = radius
        self.setRect(0, 0, self.radius, self.radius)
        self.setVisible(False)
        self.setBrush(QColor(0, 200, 0, 200))
        self.setPen(QPen(QColor(0, 0, 0, 0)))
        self.vertical_rect = QRectF(0, 0, 0.2 * self.radius, 0.8 * self.radius)
        self.horizontal_rect = QRectF(0, 0, 0.8 * self.radius, 0.2 * self.radius)

    def paint(self, painter: PySide2.QtGui.QPainter, option: PySide2.QtWidgets.QStyleOptionGraphicsItem, widget: Optional[PySide2.QtWidgets.QWidget]=...):
        QGraphicsEllipseItem.paint(self, painter, option, widget)
        self.vertical_rect.moveCenter(self.rect().center())
        self.horizontal_rect.moveCenter(self.rect().center())

        painter.fillRect(self.horizontal_rect, QColor(255, 255, 255))
        painter.fillRect(self.vertical_rect, QColor(255, 255, 255))

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent):
        pass

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent):
        print("Added")
