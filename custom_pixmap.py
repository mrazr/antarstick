from typing import Optional, List

import PySide2
from PySide2.QtCore import QMarginsF, QLine, QPoint, Slot, QByteArray, Qt, QRectF
from PySide2.QtGui import QBrush, QColor, QPainter, QFont, QPen, QImage, QPixmap, QTransform
from PySide2.QtWidgets import QGraphicsPixmapItem, QGraphicsItem, QWidget, QGraphicsSimpleTextItem, QGraphicsRectItem

from camera import Camera
from link_camera_button import LinkCameraButton
from stick_widget import StickWidget
from numpy import ndarray
from skimage.io import imread
from skimage.transform import rescale
import cv2 as cv


class QTransfrom(object):
    pass


class CustomPixmap(QGraphicsPixmapItem):

    font: QFont = QFont("monospace", 16)

    def __init__(self, font: QFont, parent: Optional[QGraphicsItem] = None):
        QGraphicsPixmapItem.__init__(self, parent)
        self.stick_widgets: List[StickWidget] = []
        self.reference_line = QLine()
        self.link_cam_text = QGraphicsSimpleTextItem("Link camera...", self)
        self.link_cam_text.setZValue(42)
        self.link_cam_text.setVisible(False)
        self.link_cam_text.setFont(font)
        self.link_cam_text.setPos(0, 0)
        self.link_cam_text.setPen(QPen(QColor(255, 255, 255, 255)))
        self.link_cam_text.setBrush(QBrush(QColor(255, 255, 255, 255)))
        self.left_add_button = LinkCameraButton(self.link_cam_text, name="left", parent=self)
        self.right_add_button = LinkCameraButton(self.link_cam_text, name="right", parent=self)

        self.show_add_buttons = False
        self.camera = None
        self.title_rect = QGraphicsRectItem(self)
        self.title_rect.setBrush(QBrush(QColor(50, 50, 50, 150)))
        self.title = QGraphicsSimpleTextItem("Nothing", self.title_rect)
        self.title.setFont(font)
        self.title.setBrush(QBrush(QColor(255, 255, 255, 255)))
        self.title.setPen(QPen(QColor(255, 255, 255, 255)))
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)

    def paint(self, painter: QPainter, option: PySide2.QtWidgets.QStyleOptionGraphicsItem, widget: QWidget):
        QGraphicsPixmapItem.paint(self, painter, option, widget)
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

    def boundingRect(self) -> PySide2.QtCore.QRectF:
        return QGraphicsPixmapItem.boundingRect(self).united(self.title_rect.boundingRect().translated(
            QPoint(0, - 0 * self.title.boundingRect().height())
        ))

    def initialise_with(self, camera: Camera):
        self.camera = camera
        img = cv.imread(str(camera.rep_image))
        img = cv.resize(img, (0, 0), fx=0.25, fy=0.25)

        self.prepareGeometryChange()
        self.set_image(img)
        #for sw in self.stick_widgets:
        #    self.scene().removeItem(sw)
        #self.stick_widgets.clear()

        #for stick in camera.sticks:
        #    self.stick_widgets.append(StickWidget(stick, self))

        self.title.setText(str(camera.folder.name))
        self.title.setPos(self.title_rect.boundingRect().width() / 2 - self.title.boundingRect().width() / 2,
                          0)
        self.title.setVisible(True)

    def set_image(self, img: ndarray):
        barray = QByteArray(img.tobytes())
        image = QImage(barray, img.shape[1], img.shape[0], QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(image)
        self.setPixmap(pixmap)

        self.title_rect.setRect(0, 0, self.pixmap().width(), self.title.boundingRect().height())
        self.title_rect.setPos(0, - 0 * self.title.boundingRect().height())
        self.title_rect.setVisible(True)
        re = self.scene().sceneRect()
        re.setWidth(1893)  # TODO replace this magic number
        self.scene().setSceneRect(re)
        #self.setPos(1893 / 2 - self.boundingRect().width() / 2, 0)

    def show_title(self, value: bool):
        self.title_rect.setVisible(value)
        self.title.setVisible(value)

    def update_stick_widgets(self):
        for sw in self.stick_widgets:
            self.scene().removeItem(sw)
        self.stick_widgets.clear()

        for stick in self.camera.sticks:
            self.stick_widgets.append(StickWidget(stick, self))

        self.scene().update()

    def scale_item(self, factor: float):
        self.prepareGeometryChange()
        pixmap = self.pixmap().scaledToHeight(int(self.pixmap().height() * factor))
        self.setPixmap(pixmap)
        self.__update_title()

    def __update_title(self):
        self.title_rect.setRect(0, 0, self.pixmap().width(), self.title.boundingRect().height())
        self.title_rect.setPos(0, - 0 * self.title.boundingRect().height())
        self.title.setPos(self.title_rect.boundingRect().width() / 2 - self.title.boundingRect().width() / 2, 0)

