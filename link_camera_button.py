from PySide2.QtWidgets import QGraphicsEllipseItem, QGraphicsSceneMouseEvent, QGraphicsSimpleTextItem,\
    QGraphicsRectItem, QGraphicsItem, QGraphicsObject
import PySide2
from PySide2.QtGui import QColor, QPen, QBrush
from PySide2.QtCore import QRectF, QMarginsF, QObject, Signal
from typing import Optional


class LinkCameraButton(QGraphicsObject):

    clicked = Signal(str)

    def __init__(self, text_item: QGraphicsSimpleTextItem, name: str, radius: int = 30, parent: QGraphicsEllipseItem = None):
        QGraphicsObject.__init__(self, parent)
        self.radius = radius
        self.setVisible(False)
        self.name = name
        self.circle = QGraphicsEllipseItem(0, 0, self.radius, self.radius, self)
        self.circle.setBrush(QColor(0, 200, 0, 200))
        self.circle.setPen(QPen(QColor(255, 255, 255, 0)))
        self.circle.setZValue(1)
        self.vertical_rect = QGraphicsRectItem(QRectF(0, 0, 0.2 * self.radius, 0.8 * self.radius), self)
        self.horizontal_rect = QGraphicsRectItem(QRectF(0, 0, 0.8 * self.radius, 0.2 * self.radius), self)

        rect = self.vertical_rect.rect()
        rect.moveCenter(self.circle.rect().center())
        self.vertical_rect.setRect(rect)
        rect = self.horizontal_rect.rect()
        rect.moveCenter(self.circle.rect().center())
        self.horizontal_rect.setRect(rect)

        self.horizontal_rect.setZValue(2)
        self.vertical_rect.setZValue(2)
        self.horizontal_rect.setPen(QPen(QColor(0, 0, 0, 0)))
        self.vertical_rect.setPen(QPen(QColor(0, 0, 0, 0)))

        self.vertical_rect.setBrush(QBrush(QColor(255, 255, 255)))
        self.horizontal_rect.setBrush(QBrush(QColor(255, 255, 255)))
        self.link_cam_text = text_item
        self.setAcceptHoverEvents(True)
        self.text_rect = QGraphicsRectItem(self.link_cam_text.boundingRect().marginsAdded(QMarginsF(5, 5, 5, 5)), self.link_cam_text)
        self.text_rect.setFlag(QGraphicsItem.ItemStacksBehindParent)
        self.text_rect.setBrush(QBrush(QColor(50, 50, 50, 100)))
        self.text_rect.setPen(QPen(QColor()))
       #self.text_rect.setVisible(False)

    def paint(self, painter: PySide2.QtGui.QPainter, option: PySide2.QtWidgets.QStyleOptionGraphicsItem, widget: Optional[PySide2.QtWidgets.QWidget]=...):
        pass

    def boundingRect(self) -> PySide2.QtCore.QRectF:
        return self.circle.boundingRect()

    def hoverEnterEvent(self, event: PySide2.QtWidgets.QGraphicsSceneHoverEvent):
        self.text_rect.setPos(self.link_cam_text.boundingRect().topLeft())
        self.text_rect.setVisible(True)
        self.link_cam_text.setPos(self.pos().x(), self.pos().y() - 1.5 * self.link_cam_text.boundingRect().height())
        self.link_cam_text.setVisible(True)

    def hoverLeaveEvent(self, event: PySide2.QtWidgets.QGraphicsSceneHoverEvent):
        self.link_cam_text.setVisible(False)
        self.text_rect.setVisible(False)

    def hoverMoveEvent(self, event: PySide2.QtWidgets.QGraphicsSceneHoverEvent):
        self.hoverEnterEvent(event)

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent):
        pass

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent):
        self.clicked.emit(self.name)
