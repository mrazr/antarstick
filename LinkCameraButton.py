from PySide2.QtWidgets import QGraphicsEllipseItem, QGraphicsSceneMouseEvent, QGraphicsSimpleTextItem, QGraphicsRectItem
import PySide2
from PySide2.QtGui import QColor, QPen, QBrush
from PySide2.QtCore import QRectF, QMarginsF
from typing import Optional


class LinkCameraButton(QGraphicsEllipseItem):

    def __init__(self, text_item: QGraphicsSimpleTextItem, radius: int = 30, parent: QGraphicsEllipseItem = None):
        QGraphicsEllipseItem.__init__(self, parent)
        self.radius = radius
        self.setRect(0, 0, self.radius, self.radius)
        self.setVisible(False)
        self.setBrush(QColor(0, 200, 0, 200))
        self.setPen(QPen(QColor(0, 0, 0, 0)))
        self.vertical_rect = QRectF(0, 0, 0.2 * self.radius, 0.8 * self.radius)
        self.horizontal_rect = QRectF(0, 0, 0.8 * self.radius, 0.2 * self.radius)
        self.link_cam_text = text_item
        self.setAcceptHoverEvents(True)
        self.text_rect = QGraphicsRectItem(self.link_cam_text.boundingRect().marginsAdded(QMarginsF(5, 5, 5, 5)), self.link_cam_text)
        self.text_rect.setBrush(QBrush(QColor(50, 50, 50, 100)))
        self.text_rect.setPen(QPen(QColor()))
        self.text_rect.setVisible(False)
        self.text_rect.setZValue(41)

    def paint(self, painter: PySide2.QtGui.QPainter, option: PySide2.QtWidgets.QStyleOptionGraphicsItem, widget: Optional[PySide2.QtWidgets.QWidget]=...):
        QGraphicsEllipseItem.paint(self, painter, option, widget)
        self.vertical_rect.moveCenter(self.rect().center())
        self.horizontal_rect.moveCenter(self.rect().center())

        painter.fillRect(self.horizontal_rect, QColor(255, 255, 255))
        painter.fillRect(self.vertical_rect, QColor(255, 255, 255))



    def hoverEnterEvent(self, event: PySide2.QtWidgets.QGraphicsSceneHoverEvent):
        pos = self.mapToParent(event.pos())
        self.link_cam_text.setPos(self.pos().x(), self.pos().y() - 1.5 * self.link_cam_text.boundingRect().height())
        self.text_rect.setPos(self.link_cam_text.boundingRect().topLeft())
        self.link_cam_text.setVisible(True)
        self.text_rect.setVisible(True)

    def hoverLeaveEvent(self, event: PySide2.QtWidgets.QGraphicsSceneHoverEvent):
        self.link_cam_text.setVisible(False)
        self.text_rect.setVisible(False)

    def hoverMoveEvent(self, event:PySide2.QtWidgets.QGraphicsSceneHoverEvent):
        self.hoverEnterEvent(event)

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent):
        pass

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent):
        print("Added")
