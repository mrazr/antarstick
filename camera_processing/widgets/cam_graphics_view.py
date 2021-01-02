from PyQt5.QtCore import Qt, pyqtSignal, QEvent, QPointF
from PyQt5.QtGui import QPainter, QWheelEvent, QResizeEvent, QMouseEvent, QKeyEvent
from PyQt5.QtWidgets import QGraphicsView, QWidget, QSizePolicy

from camera_processing.widgets.stick_link_manager import StickLinkingStrategy


class CamGraphicsView(QGraphicsView):

    view_changed = pyqtSignal()
    rubber_band_started = pyqtSignal()
    mouse_move = pyqtSignal(QPointF)

    def __init__(self, link_manager: StickLinkingStrategy, parent: QWidget = None):
        QGraphicsView.__init__(self, parent)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setViewportUpdateMode(QGraphicsView.SmartViewportUpdate)
        self.setRenderHint(QPainter.Antialiasing)

        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)

        self.verticalScrollBar().valueChanged.connect(lambda _: self.view_changed.emit())
        self.horizontalScrollBar().valueChanged.connect(lambda _: self.view_changed.emit())
        
        self.stick_link_manager = link_manager

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
        srect = self.sceneRect()
        rrect = self.mapToScene(self.rect()).boundingRect()
        if delta < 0 and rrect.height() > srect.height():
            self.fitInView(srect, Qt.KeepAspectRatio)
        self.scene().update()
        self.view_changed.emit()
    
    def resizeEvent(self, event: QResizeEvent):
        self.view_changed.emit()

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MidButton:
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            self.viewport().setCursor(Qt.ClosedHandCursor)
            self.original_event = event
            handmade_event = QMouseEvent(QEvent.MouseButtonPress, QPointF(event.pos()), Qt.LeftButton, event.buttons(),
                                         Qt.KeyboardModifiers())
            self.mousePressEvent(handmade_event)
            return None
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.RightButton:
            self.stick_link_manager.cancel()
        elif event.button() == Qt.LeftButton:
            self.stick_link_manager.accept()
        elif event.button() == Qt.MidButton:
            self.viewport().setCursor(Qt.OpenHandCursor)
            handmade_event = QMouseEvent(QEvent.MouseButtonRelease, QPointF(event.pos()), Qt.LeftButton,
                                         event.buttons(), Qt.KeyboardModifiers())
            self.mouseReleaseEvent(handmade_event)
            self.setDragMode(QGraphicsView.NoDrag)
        super().mouseReleaseEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        self.stick_link_manager.set_target(self.mapToScene(event.pos()))
        self.mouse_move.emit(self.mapToScene(event.pos()))
        QGraphicsView.mouseMoveEvent(self, event)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key_Control:
            self.setDragMode(QGraphicsView.ScrollHandDrag)
        elif event.key() == Qt.Key_Shift:
            self.rubber_band_started.emit()
            self.setDragMode(QGraphicsView.RubberBandDrag)
            self.setRubberBandSelectionMode(Qt.ContainsItemBoundingRect)
        QGraphicsView.keyPressEvent(self, event)

    def keyReleaseEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key_Control or event.key() == Qt.Key_Shift:
            self.setDragMode(QGraphicsView.NoDrag)
        QGraphicsView.keyReleaseEvent(self, event)
