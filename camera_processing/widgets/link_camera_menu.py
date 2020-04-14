from typing import List, Optional, Callable

import PyQt5
from PyQt5.QtCore import QMarginsF
from PyQt5.QtCore import pyqtSignal as Signal
from PyQt5.QtGui import QBrush, QColor, QPen
from PyQt5.QtWidgets import QGraphicsItem, QGraphicsObject, QGraphicsRectItem

from camera import Camera
from camera_processing.widgets.custom_pixmap import CustomPixmap


class LinkCameraMenu(QGraphicsObject):
    link_camera_selected = Signal(Camera)

    def __init__(self, position: str, parent: Optional[QGraphicsItem] = None):
        QGraphicsObject.__init__(self, parent)
        self.cameras = []
        self.camera_pixmaps = []
        self.background_rect = QGraphicsRectItem(parent=self)
        self.background_rect.setBrush(QBrush(QColor(10, 10, 10, 150)))
        self.background_rect.setPen(QPen(QColor(0, 0, 0, 0)))
        self.pixmap_scale = 0.25
        self.pixmap_bottom_margin = 20
        self.setAcceptHoverEvents(True)
        self.position = position
        self.background_rect_margin = 10

    def paint(self, painter: PyQt5.QtGui.QPainter, option: PyQt5.QtWidgets.QStyleOptionGraphicsItem,
              widget: Optional[PyQt5.QtWidgets.QWidget] = ...):
        pass

    def boundingRect(self) -> PyQt5.QtCore.QRectF:
        return self.background_rect.rect()

    def initialise_with(self, cameras: List[Camera], click_handler: Callable[[Camera], None]):
        for i, camera in enumerate(cameras):
            c_pixmap = CustomPixmap(self)
            c_pixmap.setAcceptHoverEvents(True)
            c_pixmap.initialise_with(camera)
            c_pixmap.set_button_mode(click_handler, self.position)
            c_pixmap.scale_item(self.pixmap_scale)
            c_pixmap.setPos(self.background_rect_margin, i * (c_pixmap.sceneBoundingRect().height() + self.pixmap_bottom_margin) + self.background_rect_margin)
            self.camera_pixmaps.append(c_pixmap)


        self.background_rect.setRect(0, 0,
                                     self.camera_pixmaps[0].sceneBoundingRect().width() + 2 * self.background_rect_margin,
                                     len(self.camera_pixmaps) * self.camera_pixmaps[0].sceneBoundingRect().height()
                                     + (len(self.camera_pixmaps) - 1) * self.pixmap_bottom_margin + 2 * self.background_rect_margin)
        self.prepareGeometryChange()
        self.scene().update()
