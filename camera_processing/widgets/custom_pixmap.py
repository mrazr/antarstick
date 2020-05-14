from typing import Callable, List, Optional

import cv2 as cv
import PyQt5
from numpy import ndarray
from PyQt5.QtCore import QByteArray, QLine, QMarginsF, QPoint
from PyQt5.QtGui import QBrush, QColor, QFont, QImage, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (QGraphicsItem, QGraphicsPixmapItem,
                             QGraphicsRectItem, QGraphicsSceneHoverEvent,
                             QGraphicsSceneMouseEvent, QGraphicsSimpleTextItem,
                             QWidget)

from camera import Camera
from camera_processing.widgets.link_camera_button import LinkCameraButton
from camera_processing.widgets.stick_widget import StickWidget


class CustomPixmap(QGraphicsPixmapItem):

    font: QFont = QFont("monospace", 16)

    def __init__(self, parent: Optional[QGraphicsItem] = None):
        QGraphicsPixmapItem.__init__(self, parent)
        self.stick_widgets: List[StickWidget] = []
        self.reference_line = QLine()
        self.link_cam_text = QGraphicsSimpleTextItem("Link camera...", self)
        self.link_cam_text.setZValue(40)
        self.link_cam_text.setVisible(False)
        self.link_cam_text.setFont(CustomPixmap.font)
        self.link_cam_text.setPos(0, 0)
        self.link_cam_text.setPen(QPen(QColor(255, 255, 255, 255)))
        self.link_cam_text.setBrush(QBrush(QColor(255, 255, 255, 255)))
        self.left_add_button = LinkCameraButton(self.link_cam_text, name="left", parent=self)
        self.right_add_button = LinkCameraButton(self.link_cam_text, name="right", parent=self)
        self.left_add_button.setZValue(3)
        self.right_add_button.setZValue(3)

        self.show_add_buttons = False
        self.camera = None
        self.title_rect = QGraphicsRectItem(self)
        self.title_rect.setBrush(QBrush(QColor(50, 50, 50, 150)))
        self.title = QGraphicsSimpleTextItem("Nothing", self.title_rect)
        self.title.setFont(CustomPixmap.font)
        self.title.setBrush(QBrush(QColor(255, 255, 255, 255)))
        self.title.setPen(QPen(QColor(255, 255, 255, 255)))
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)
        self.show_stick_widgets = False
        self.setAcceptHoverEvents(True)
        self.stick_edit_mode = False

        self.original_pixmap = self.pixmap()

        self.hovered = False

        self.mode = 0 # TODO make enum Mode
        self.click_handler = None
        self.double_click_handler: Callable[[int, int], None] = None

    def paint(self, painter: QPainter, option: PyQt5.QtWidgets.QStyleOptionGraphicsItem, widget: QWidget):
        QGraphicsPixmapItem.paint(self, painter, option, widget)
        if self.pixmap().isNull():
            return
        painter.setRenderHint(QPainter.Antialiasing, True)
        QGraphicsPixmapItem.paint(self, painter, option, widget)

        if self.show_stick_widgets:
            brush = QBrush(QColor(255, 255, 255, 100))
            painter.fillRect(self.boundingRect(), brush)

            #for sw in self.stick_widgets:
            #    painter.drawPixmap(sw.gline.boundingRect().marginsAdded(QMarginsF(10, 10, 10, 10)),
            #                       self.pixmap(), sw.gline.boundingRect().marginsAdded(QMarginsF(10, 10, 10, 10)))

        if self.mode and self.hovered:
            pen = QPen(QColor(0, 125, 200, 255))
            pen.setWidth(4)
            painter.setPen(pen)
            painter.drawRect(self.boundingRect().marginsAdded(QMarginsF(4, 4, 4, 4)))

        painter.drawLine(self.reference_line)

    def set_reference_line_percentage(self, percentage: float):
        if self.pixmap().isNull():
            return
        pixmap = self.pixmap()
        self.reference_line.setP1(QPoint(int(pixmap.width() * 0.5), int(pixmap.height() - 1.0)))
        self.reference_line.setP2(QPoint(int(pixmap.width() * 0.5), int(pixmap.height() * (1 - percentage))))
        self.scene().update()

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

    def boundingRect(self) -> PyQt5.QtCore.QRectF:
        #return QGraphicsPixmapItem.boundingRect(self).united(self.title_rect.boundingRect().translated(
        #    QPoint(0, - 0 * self.title.boundingRect().height())
        #))
        return QGraphicsPixmapItem.boundingRect(self)

    def initialise_with(self, camera: Camera) -> List[StickWidget]:
        self.camera = camera
        #img = cv.imread(str(camera.rep_image_path))
        #img = cv.resize(img, (0, 0), fx=0.25, fy=0.25)


        self.prepareGeometryChange()
        self.set_image(camera.rep_image)

        return self.update_stick_widgets()

    def set_image(self, img: ndarray):
        barray = QByteArray(img.tobytes())
        image = QImage(barray, img.shape[1], img.shape[0], QImage.Format_BGR888)
        self.original_pixmap = QPixmap.fromImage(image)
        self.setPixmap(self.original_pixmap)

        self.title_rect.setRect(0, 0, self.pixmap().width(), self.title.boundingRect().height())
        self.title_rect.setPos(0, - 0 * self.title.boundingRect().height())
        self.title_rect.setVisible(True)

        self.title.setText(str(self.camera.folder.name))
        self.title.setPos(self.title_rect.boundingRect().width() / 2 - self.title.boundingRect().width() / 2,
                          0)
        self.title.setVisible(True)

    def set_show_title(self, value: bool):
        self.title_rect.setVisible(value)
        self.title.setVisible(value)

    def update_stick_widgets(self) -> List[StickWidget]:
        self._remove_stick_widgets()

        for stick in self.camera.sticks:
            sw = StickWidget(stick, self)
            sw.set_edit_mode(self.stick_edit_mode)
            self.stick_widgets.append(sw)
        
        if len(self.stick_widgets) > 0:
            self.show_stick_widgets = True
            _f = self.stick_widgets[0]

        self.scene().update()
        return self.stick_widgets

    def scale_item(self, factor: float):
        self.prepareGeometryChange()
        pixmap = self.original_pixmap.scaledToHeight(int(self.original_pixmap.height() * factor))
        self.setPixmap(pixmap)
        self.__update_title()

    def __update_title(self):
        self.title_rect.setRect(0, 0, self.pixmap().width(), self.title.boundingRect().height())
        self.title_rect.setPos(0, - 0 * self.title.boundingRect().height())
        self.title.setPos(self.title_rect.boundingRect().width() / 2 - self.title.boundingRect().width() / 2, 0)

    def set_show_stick_widgets(self, value: bool):
        for sw in self.stick_widgets:
            sw.setVisible(value)
        self.scene().update()

    def hoverEnterEvent(self, e: QGraphicsSceneHoverEvent):
        self.hovered = True
        self.scene().update(self.sceneBoundingRect())
    
    def hoverLeaveEvent(self, e: QGraphicsSceneHoverEvent):
        self.hovered = False
        self.scene().update(self.sceneBoundingRect())
    
    def mousePressEvent(self, e: QGraphicsSceneMouseEvent):
        pass
    
    def mouseReleaseEvent(self, e: QGraphicsSceneMouseEvent):
        if self.mode == 1:
            self.click_handler(self.camera)

    def mouseDoubleClickEvent(self, event: QGraphicsSceneMouseEvent):
        if self.stick_edit_mode:
            self.double_click_handler(event.pos().toPoint().x(), event.pos().toPoint().y())

    def set_button_mode(self, click_handler: Callable[[Camera], None], data: str):
        self.mode = 1 # TODO make a proper ENUM
        self.click_handler = lambda c: click_handler(c, data)
    
    def set_display_mode(self):
        self.mode = 0 # TODO make a proper ENUM
        self.click_handler = None

    def disable_link_button(self, btn_position: str):
        if btn_position == "left":
            self.left_add_button.setVisible(False)
        else:
            self.right_add_button.setVisible(False)
    
    def enable_link_button(self, btn_position: str):
        if btn_position == "left":
            self.left_add_button.setVisible(True)
        else:
            self.right_add_button.setVisible(True)

    def _remove_stick_widgets(self):
        for sw in self.stick_widgets:
            sw.setParentItem(None)
            self.scene().removeItem(sw)
            sw.deleteLater()
        self.stick_widgets.clear()
    
    def set_stick_edit_mode(self, value: bool):
        self.stick_edit_mode = value