from PyQt5.QtWidgets import QGraphicsView, QWidget, QSizePolicy
from PyQt5.QtGui import QPixmap, QPainter, QBrush, QPen, QColor, QWheelEvent
from PyQt5.QtCore import Qt, QRectF, QPointF

import sys
from pathlib import Path


class CamGraphicsView(QGraphicsView):

    def __init__(self, parent: QWidget = None):
        QGraphicsView.__init__(self, parent)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)

        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        path = Path(sys.argv[0]).parent / "camera_processing/gui_resources/"

        self.mouse_zoom_pic = QPixmap(str(path / "mouse_zoom.png"))
        self.mouse_zoom_pic = self.mouse_zoom_pic.scaledToWidth(80, Qt.SmoothTransformation)

        self.mouse_pan_pic = QPixmap(str(path / "mouse_pan.png"))
        self.mouse_pan_pic = self.mouse_pan_pic.scaledToWidth(80, Qt.SmoothTransformation)

    def drawForeground(self, painter: QPainter, rect: QRectF) -> None:
        painter.save()

        painter.setWorldMatrixEnabled(False)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor("#aa555555")))
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.TextAntialiasing, True)
        painter.drawRoundedRect(QRectF(5, 5, self.mouse_pan_pic.width() * 1.3,
                                       3 * self.mouse_pan_pic.height()), 20, 20)

        painter.setPen(QPen(QColor("#e5c15f")))
        painter.drawPixmap(QPointF(15, 5), self.mouse_pan_pic, QRectF(self.mouse_pan_pic.rect()))
        rect = QRectF(self.mouse_pan_pic.rect())
        painter.drawPixmap(QPointF(15, 1.5 * rect.height()), self.mouse_zoom_pic,
                           QRectF(self.mouse_zoom_pic.rect()))
        font = painter.font()
        font.setPointSize(10)
        painter.drawText(QRectF(15, 1.1 * rect.height(), rect.width(), 30),
                         Qt.AlignHCenter, "Pan view")
        painter.drawText(QRectF(15, 2.6 * rect.height(), rect.width(), 30),
                         Qt.AlignHCenter, "Zoom in/out")
        painter.setWorldMatrixEnabled(True)
        painter.restore()

    def wheelEvent(self, event: QWheelEvent) -> None:
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
