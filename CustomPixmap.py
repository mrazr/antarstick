import PySide2
from PySide2.QtCore import QMarginsF, QLine, QPoint, Slot
from PySide2.QtGui import QBrush, QColor, QPainter
from PySide2.QtWidgets import QGraphicsPixmapItem, QGraphicsItem, QWidget, QGraphicsRectItem
from PySide2.QtGui import QPixmap
from typing import Optional, List
from stick_widget import StickWidget
from LinkCameraButton import LinkCameraButton


class CustomPixmap(QGraphicsPixmapItem):
    def __init__(self, parent: Optional[QGraphicsItem] = None):
        QGraphicsPixmapItem.__init__(self, parent)
        self.stick_widgets: List[StickWidget] = []
        self.reference_line = QLine()
        self.left_add_button = LinkCameraButton(parent=self)
        self.right_add_button = LinkCameraButton(parent=self)
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

        if self.show_add_buttons:
            pass
            #self.right_add_button.setRect(self.pixmap().width() - 5, self.pixmap().height() * 0.5, 10, 10)
            #self.left_add_button.setRect(-5, self.pixmap().height() * 0.5, 10, 10)
            #painter.setBrush(QBrush(QColor(0, 200, 50, 200)))
            #painter.drawEllipse(self.left_add_button.rect().center(), 10, 10)
            #painter.drawEllipse(self.right_add_button.rect().center(), 10, 10)

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
        #self.scene().update()

