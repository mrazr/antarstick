from PySide2.QtGui import QBrush, QColor, QPen
from PySide2.QtWidgets import  QGraphicsRectItem, QGraphicsItem, QGraphicsObject
import PySide2
from typing import Optional, List
from PySide2.QtCore import Signal, QMarginsF

from camera import Camera
from .custom_pixmap import CustomPixmap


class LinkCameraMenu(QGraphicsObject):

    link_camera_selected = Signal(Camera)

    def __init__(self, parent: Optional[QGraphicsItem] = None):
        QGraphicsObject.__init__(self, parent)
        self.cameras = []
        self.camera_pixmaps = []
        self.background_rect = QGraphicsRectItem(parent=self)
        self.background_rect.setBrush(QBrush(QColor(10, 10, 10, 150)))
        self.background_rect.setPen(QPen(QColor(0, 0, 0, 0)))
        self.pixmap_scale = 0.25
        self.pixmap_bottom_margin = 20
        self.setAcceptHoverEvents(True)

    def paint(self, painter: PySide2.QtGui.QPainter, option: PySide2.QtWidgets.QStyleOptionGraphicsItem, widget: Optional[PySide2.QtWidgets.QWidget]=...):
        pass

    def boundingRect(self) -> PySide2.QtCore.QRectF:
        return self.background_rect.boundingRect()

    def initialise_with(self, cameras: List[Camera]):
        for i, camera in enumerate(cameras):
            c_pixmap = CustomPixmap(CustomPixmap.font, parent=self)
            c_pixmap.initialise_with(camera)
            c_pixmap.scale_item(self.pixmap_scale)
            c_pixmap.setPos(0, i * (c_pixmap.sceneBoundingRect().height() + self.pixmap_bottom_margin))
            self.camera_pixmaps.append(c_pixmap)

        self.background_rect.setRect(0, 0,
                                     self.camera_pixmaps[0].sceneBoundingRect().width(),
                                     len(self.camera_pixmaps) * self.camera_pixmaps[0].sceneBoundingRect().height()
                                     + (len(self.camera_pixmaps) - 1) * self.pixmap_bottom_margin)
        self.background_rect.setRect(self.background_rect.rect().marginsAdded(QMarginsF(10, 10, 10, 10)))
