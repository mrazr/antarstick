import sys
from pathlib import Path

from PyQt5.Qt import QBrush, QColor, QPen
from PyQt5.QtCore import QPoint, QPointF, QRectF, Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QPainter, QPixmap
from PyQt5.QtWidgets import QGraphicsItem, QGraphicsObject

from camera_processing.widgets.button_menu import ButtonMenu
from camera_processing.widgets.cam_graphics_view import CamGraphicsView


class OverlayGui(QGraphicsObject):

    reset_view_requested = pyqtSignal()
    edit_sticks_clicked = pyqtSignal()
    link_sticks_clicked = pyqtSignal()

    def __init__(self, view: CamGraphicsView, parent: QGraphicsItem = None):
        QGraphicsObject.__init__(self, parent)

        self.view = view
        self.view.view_changed.connect(self.handle_cam_view_changed)
        self.setZValue(43)
        self.setAcceptHoverEvents(False)
        self.setAcceptedMouseButtons(Qt.AllButtons)
        self.top_menu = ButtonMenu(self)
        self.mouse_zoom_pic = QPixmap()
        self.mouse_pan_pic = QPixmap()
        self.icons_rect = QRectF(0, 0, 0, 0)
        self.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)

    def initialize(self):
        path = Path(sys.argv[0]).parent / "camera_processing/gui_resources/"

        self.mouse_zoom_pic = QPixmap(str(path / "mouse_zoom.png"))
        self.mouse_zoom_pic = self.mouse_zoom_pic.scaledToWidth(80, Qt.SmoothTransformation)

        self.mouse_pan_pic = QPixmap(str(path / "mouse_pan.png"))
        self.mouse_pan_pic = self.mouse_pan_pic.scaledToWidth(80, Qt.SmoothTransformation)

        self.top_menu.add_button("edit_sticks", "Edit sticks", is_checkable=True, call_back=self.edit_sticks_clicked.emit)
        self.top_menu.add_button("link_sticks", "Link sticks", is_checkable=True, call_back=self.link_sticks_clicked.emit)
        self.top_menu.add_button("show_overaly", "Show overlay")
        self.top_menu.add_button("show_linked_cameras", "Show linked cameras")
        self.top_menu.add_button("reset_view", "Reset view", call_back=self.reset_view_requested.emit)
        self.top_menu.set_height(40)

        self.top_menu.setPos(QPoint(0, 0))

        self.icons_rect = QRectF(5, 5, self.mouse_pan_pic.width() * 1.3,
                                       3 * self.mouse_pan_pic.height())
    
    @pyqtSlot()
    def handle_cam_view_changed(self):
        self.prepareGeometryChange()
        self.setPos(self.view.mapToScene(QPoint(0, 0)))
        self.top_menu.setPos(self.boundingRect().width() / 2.0 - self.top_menu.boundingRect().width() / 2.0, 2)

    def boundingRect(self):
        return QRectF(self.view.viewport().rect())
    
    def paint(self, painter: QPainter, options, widget=None):
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
    
    def edit_sticks_button_pushed(self) -> bool:
        return self.top_menu.is_button_checked("edit_sticks")
    
    def link_sticks_button_pushed(self) -> bool:
        return self.top_menu.is_button_checked("link_sticks")
