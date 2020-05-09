from PyQt5.QtWidgets import QGraphicsView, QWidget, QSizePolicy, QGraphicsRectItem
from PyQt5.QtGui import QPixmap, QPainter, QBrush, QPen, QColor, QWheelEvent, QResizeEvent, QMouseEvent
from PyQt5.QtCore import Qt, QRectF, QPointF, pyqtSignal

import sys
from pathlib import Path
from camera_processing.widgets.stick_link_manager import StickLinkManager


class CamGraphicsView(QGraphicsView):

    view_changed = pyqtSignal()

    def __init__(self, link_manager: StickLinkManager, parent: QWidget = None):
        QGraphicsView.__init__(self, parent)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.setRenderHint(QPainter.Antialiasing)

        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        #self.gui = QGraphicsRectItem()
        #self.gui.setBrush(QBrush(QColor(255, 255, 255, 20)))
        #self.gui.setZValue(999)

        self.verticalScrollBar().valueChanged.connect(lambda _: self.view_changed.emit())
        self.horizontalScrollBar().valueChanged.connect(lambda _: self.view_changed.emit())
        
        self.stick_link_manager = link_manager

    def drawForeground(self, painter: QPainter, rect: QRectF) -> None:
        pass
        #painter.save()

        #painter.setWorldMatrixEnabled(False)
        #painter.setPen(Qt.NoPen)
        #painter.setBrush(QBrush(QColor("#aa555555")))
        #painter.setRenderHint(QPainter.Antialiasing, True)
        #painter.setRenderHint(QPainter.TextAntialiasing, True)
        #painter.drawRoundedRect(QRectF(5, 5, self.mouse_pan_pic.width() * 1.3,
        #                               3 * self.mouse_pan_pic.height()), 20, 20)

        #painter.setPen(QPen(QColor("#e5c15f")))
        #painter.drawPixmap(QPointF(15, 5), self.mouse_pan_pic, QRectF(self.mouse_pan_pic.rect()))
        #rect = QRectF(self.mouse_pan_pic.rect())
        #painter.drawPixmap(QPointF(15, 1.5 * rect.height()), self.mouse_zoom_pic,
        #                   QRectF(self.mouse_zoom_pic.rect()))
        #font = painter.font()
        #font.setPointSize(10)
        #painter.drawText(QRectF(15, 1.1 * rect.height(), rect.width(), 30),
        #                 Qt.AlignHCenter, "Pan view")
        #painter.drawText(QRectF(15, 2.6 * rect.height(), rect.width(), 30),
        #                 Qt.AlignHCenter, "Zoom in/out")
        #painter.setWorldMatrixEnabled(True)
        #painter.restore()

    def wheelEvent(self, event: QWheelEvent) -> None:
        pass
        delta = 1
        if event.angleDelta().y() < 0:
            delta = -1

        m = self.transform()
        m11 = m.m11() * (1 + delta * 0.05)
        m31 = 0
        m22 = m.m22() * (1 + delta * 0.05)
        m32 = 0

        m.setMatrix(m11, m.m12(), m.m13(), m.m21(), m22, m.m23(),
                    m.m31() + m31, m.m32() + m32, m.m33())
        self.setTransform(m, False)
        #self.gui.setRect(self.mapToScene(self.viewport().rect()).boundingRect())
        self.scene().update()
        self.view_changed.emit()
    
    def resizeEvent(self, event: QResizeEvent):
        self.view_changed.emit()

    def mousePressEvent(self, event: QMouseEvent):
        QGraphicsView.mousePressEvent(self, event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.RightButton:
            self.stick_link_manager.cancel()
        elif event.button() == Qt.LeftButton:
            self.stick_link_manager.accept()
        QGraphicsView.mouseReleaseEvent(self, event)

    def mouseMoveEvent(self, event: QMouseEvent):
        if not self.stick_link_manager.anchored:
            self.stick_link_manager.set_target(self.mapToScene(event.pos()))
        QGraphicsView.mouseMoveEvent(self, event)