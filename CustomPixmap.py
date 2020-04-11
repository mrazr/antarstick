import PySide2
from PySide2.QtCore import QMarginsF, QLine, QPoint, Slot
from PySide2.QtGui import QBrush, QColor, QPainter, QFont
from PySide2.QtWidgets import QGraphicsPixmapItem, QGraphicsItem, QWidget, QGraphicsSimpleTextItem
from PySide2.QtGui import QPixmap
from typing import Optional, List
from stick_widget import StickWidget
from LinkCameraButton import LinkCameraButton


class CustomPixmap(QGraphicsPixmapItem):
    def __init__(self, font: QFont, parent: Optional[QGraphicsItem] = None):
        QGraphicsPixmapItem.__init__(self, parent)
        self.stick_widgets: List[StickWidget] = []
        self.reference_line = QLine()
        self.link_cam_text = QGraphicsSimpleTextItem("Link camera...", self)
        self.link_cam_text.setZValue(42)
        self.link_cam_text.setVisible(False)
        self.link_cam_text.setFont(font)
        self.link_cam_text.setPos(0, 0)
        self.left_add_button = LinkCameraButton(self.link_cam_text, parent=self)
        self.right_add_button = LinkCameraButton(self.link_cam_text, parent=self)
        self.show_add_buttons = False

    def paint(self, painter: QPainter, option: PySide2.QtWidgets.QStyleOptionGraphicsItem, widget: QWidget):
        if self.pixmap().isNull():
            return
        painter.setRenderHint(QPainter.Antialiasing, True)
        QGraphicsPixmapItem.paint(self, painter, option, widget)
        brush = QBrush(QColor(255, 255, 255, 100))
        painter.fillRect(self.boundingRect(), brush)

        painter.drawLine(self.reference_line)

        for sw in self.stick_widgets:
            painter.drawPixmap(sw.gline.boundingRect().marginsAdded(QMarginsF(10, 10, 10, 10)),
                               self.pixmap(), sw.gline.boundingRect().marginsAdded(QMarginsF(10, 10, 10, 10)))

    def set_reference_line_percentage(self, percentage: float):
        if self.pixmap().isNull():
            return
        pixmap = self.pixmap()
        self.reference_line.setP1(QPoint(int(pixmap.width() * 0.5), int(pixmap.height() - 1.0)))
        self.reference_line.setP2(QPoint(int(pixmap.width() * 0.5), int(pixmap.height() * (1 - percentage))))
        self.scene().update()

    @Slot(bool)
    def set_link_cameras_enabled(self, value: bool):
        self.show_add_buttons = value
        if self.show_add_buttons:
            offset = 0.5 * self.right_add_button.radius
            self.right_add_button.setPos(self.pixmap().width() - offset, self.pixmap().height() * 0.5 - offset)
            self.left_add_button.setPos(-offset, self.pixmap().height() * 0.5 - offset)
            self.right_add_button.setVisible(True)
            self.left_add_button.setVisible(True)
        else:
            self.right_add_button.setVisible(False)
            self.left_add_button.setVisible(False)
        self.scene().update()

