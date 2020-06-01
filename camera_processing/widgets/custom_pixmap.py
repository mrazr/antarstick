from typing import Callable, List, Optional

import PyQt5
from PyQt5.QtCore import QByteArray, QLine, QMarginsF, QPoint, pyqtSignal, QPointF, Qt
from PyQt5.QtGui import QBrush, QColor, QFont, QImage, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (QGraphicsItem, QGraphicsPixmapItem,
                             QGraphicsRectItem, QGraphicsSceneHoverEvent,
                             QGraphicsSceneMouseEvent, QGraphicsSimpleTextItem,
                             QWidget, QGraphicsObject)
from numpy import ndarray

from camera import Camera
from camera_processing.widgets.link_camera_button import LinkCameraButton
from camera_processing.widgets.stick_length_input import StickLengthInput
from camera_processing.widgets.stick_widget import StickWidget, StickMode
from camera_processing.widgets.button import Button
from dataset import Dataset
from stick import Stick


class CustomPixmap(QGraphicsObject):

    font: QFont = QFont("monospace", 16)
    stick_link_requested = pyqtSignal(StickWidget)
    stick_widgets_out_of_sync = pyqtSignal('PyQt_PyObject')

    def __init__(self, dataset: Dataset, parent: Optional[QGraphicsItem] = None):
        QGraphicsObject.__init__(self, parent)
        self.gpixmap = QGraphicsPixmapItem(self)
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

        self.original_pixmap = self.gpixmap.pixmap()

        self.hovered = False

        self.mode = 0 # TODO make enum Mode
        self.click_handler = None
        self.double_click_handler: Callable[[int, int], None] = None

        self.dataset = dataset

        self.stick_widget_mode = StickMode.DISPLAY

        self.stick_length_lbl = QGraphicsSimpleTextItem("sticks length: ", self.title_rect)
        self.stick_length_lbl.setFont(Button.font)
        self.stick_length_lbl.setBrush(QBrush(Qt.white))
        self.stick_length_lbl.setVisible(False)
        self.stick_length_btn = Button("btn_stick_length", "60 cm", self.title_rect)
        self.stick_length_btn.clicked.connect(self.handle_stick_length_clicked)
        self.stick_length_btn.setVisible(False)
        self.stick_length_btn.set_is_check_button(True)

        self.stick_length_input = StickLengthInput(self)
        self.stick_length_input.input_entered.connect(self.handle_stick_length_input_entered)
        self.stick_length_input.input_cancelled.connect(self.handle_stick_length_input_cancelled)

    def paint(self, painter: QPainter, option: PyQt5.QtWidgets.QStyleOptionGraphicsItem, widget: QWidget):
        if self.gpixmap.pixmap().isNull():
            return
        painter.setRenderHint(QPainter.Antialiasing, True)

        if self.show_stick_widgets:
            brush = QBrush(QColor(255, 255, 255, 100))
            painter.fillRect(self.boundingRect(), brush)

        if self.mode and self.hovered:
            pen = QPen(QColor(0, 125, 200, 255))
            pen.setWidth(4)
            painter.setPen(pen)
            painter.drawRect(self.boundingRect().marginsAdded(QMarginsF(4, 4, 4, 4)))

        painter.drawLine(self.reference_line)

    def set_reference_line_percentage(self, percentage: float):
        if self.gpixmap.pixmap().isNull():
            return
        pixmap = self.gpixmap.pixmap()
        self.reference_line.setP1(QPoint(int(pixmap.width() * 0.5), int(pixmap.height() - 1.0)))
        self.reference_line.setP2(QPoint(int(pixmap.width() * 0.5), int(pixmap.height() * (1 - percentage))))
        self.scene().update()

    def set_link_cameras_enabled(self, value: bool):
        self.show_add_buttons = value
        if self.show_add_buttons:
            offset = 0.5 * self.right_add_button.radius
            self.right_add_button.setPos(self.gpixmap.pixmap().width() - offset, self.gpixmap.pixmap().height() * 0.5 - offset)
            self.left_add_button.setPos(-offset, self.gpixmap.pixmap().height() * 0.5 - offset)
            self.right_add_button.setVisible(True)
            self.left_add_button.setVisible(True)
        else:
            self.right_add_button.setVisible(False)
            self.left_add_button.setVisible(False)
        self.scene().update()

    def boundingRect(self) -> PyQt5.QtCore.QRectF:
        return self.gpixmap.boundingRect()

    def initialise_with(self, camera: Camera) -> List[StickWidget]:
        self.camera = camera
        self.prepareGeometryChange()
        self.set_image(camera.rep_image)
        self.stick_length_btn.setVisible(True)
        self.stick_length_lbl.setVisible(True)
        self.camera.stick_added.connect(self.handle_stick_created)
        self.camera.sticks_added.connect(self.handle_sticks_added)
        self.camera.stick_removed.connect(self.handle_stick_removed)
        self.camera.sticks_removed.connect(self.handle_sticks_removed)
        self.camera.stick_changed.connect(self.handle_stick_changed)

        return self.update_stick_widgets()

    def set_image(self, img: ndarray):
        barray = QByteArray(img.tobytes())
        image = QImage(barray, img.shape[1], img.shape[0], QImage.Format_BGR888)
        self.original_pixmap = QPixmap.fromImage(image)
        self.gpixmap.setPixmap(self.original_pixmap)
        self.layout_title_area()

    def layout_title_area(self):
        self.title.setText(str(self.camera.folder.name))
        self.title_rect.setRect(0, 0, self.gpixmap.pixmap().width(), self.title.boundingRect().height())
        self.title_rect.setPos(0, - 0 * self.title.boundingRect().height())
        self.title_rect.setVisible(True)

        self.title.setPos(self.title_rect.boundingRect().width() / 2 - self.title.boundingRect().width() / 2,
                          0)
        self.title.setVisible(True)
        self.stick_length_btn.set_height(self.title_rect.boundingRect().height() - 4)
        self.stick_length_btn.setPos(self.title_rect.boundingRect().width() - 5 - self.stick_length_btn.boundingRect().width(),
                                     2)
        #self.stick_length_btn.setPos(self.stick_length_lbl.pos() + QPointF(self.stick_length_lbl.boundingRect().width(), 0))
        #self.stick_length_lbl.setPos(self.title.pos() + QPointF(self.title.boundingRect().width(), 0))
        self.stick_length_lbl.setPos(self.stick_length_btn.pos().x() - self.stick_length_lbl.boundingRect().width(), 0)

        self.stick_length_input.adjust_layout()
        self.stick_length_input.setPos(QPointF(0.5 * self.boundingRect().width(),
                                               0.5 * self.boundingRect().height()))
        self.stick_length_input.setVisible(self.stick_length_btn.is_on())

    def set_show_title(self, value: bool):
        self.title_rect.setVisible(value)
        self.title.setVisible(value)

    def update_stick_widgets(self):
        stick_length = 60
        for stick in self.camera.sticks:
            sw = StickWidget(stick, self)
            sw.set_mode(self.stick_widget_mode)
            self.connect_stick_widget_signals(sw)
            self.stick_widgets.append(sw)
            stick_length = stick.length_cm
        self.stick_length_btn.set_label(str(stick_length) + " cm")
        self.stick_length_input.set_length(stick_length)
        self.layout_title_area()
        self.scene().update()

    def scale_item(self, factor: float):
        self.prepareGeometryChange()
        pixmap = self.original_pixmap.scaledToHeight(int(self.original_pixmap.height() * factor))
        self.gpixmap.setPixmap(pixmap)
        self.__update_title()

    def __update_title(self):
        self.title_rect.setRect(0, 0, self.gpixmap.pixmap().width(), self.title.boundingRect().height())
        self.title_rect.setPos(0, - 0 * self.title.boundingRect().height())
        self.title.setPos(self.title_rect.boundingRect().width() / 2 - self.title.boundingRect().width() / 2, 0)
        self.stick_length_btn.setPos(self.title.pos() + QPointF(self.title.boundingRect().width(), 0))

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
        if self.stick_widget_mode == StickMode.EDIT:
            x = event.pos().toPoint().x()
            y = event.pos().toPoint().y()
            stick = self.dataset.create_new_stick(self.camera)
            stick.set_endpoints(x, y - 50, x, y + 50)
            self.camera.add_stick(stick)

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

    def handle_stick_created(self, stick: Stick):
        if stick.camera_id != self.camera.id:
            return
        sw = StickWidget(stick, self)
        sw.set_mode(self.stick_widget_mode)
        self.connect_stick_widget_signals(sw)
        self.stick_widgets.append(sw)
        self.stick_widgets_out_of_sync.emit(self)
        self.update()

    def handle_stick_removed(self, stick: Stick):
        if stick.camera_id != self.camera.id:
            return
        stick_widget = next(filter(lambda sw: sw.stick.id == stick.id, self.stick_widgets))
        self.disconnect_stick_widget_signals(stick_widget)
        self.stick_widgets.remove(stick_widget)
        stick_widget.setParentItem(None)
        self.scene().removeItem(stick_widget)
        stick_widget.deleteLater()
        self.update()

    def handle_sticks_removed(self, sticks: List[Stick]):
        if sticks[0].camera_id != self.camera.id:
            return
        for stick in sticks:
            to_remove: StickWidget = None
            for sw in self.stick_widgets:
                if sw.stick.id == stick.id:
                    to_remove = sw
                    break
            self.stick_widgets.remove(to_remove)
            to_remove.setParentItem(None)
            self.scene().removeItem(to_remove)
            to_remove.deleteLater()
        self.update()

    def handle_sticks_added(self, sticks: List[Stick]):
        if sticks[0].camera_id != self.camera.id:
            return
        for stick in sticks:
            sw = StickWidget(stick, self)
            sw.set_mode(self.stick_widget_mode)
            self.connect_stick_widget_signals(sw)
            self.stick_widgets.append(sw)
        self.stick_widgets_out_of_sync.emit(self)
        self.update()

    def connect_stick_widget_signals(self, stick_widget: StickWidget):
        stick_widget.delete_clicked.connect(self.handle_stick_widget_delete_clicked)
        stick_widget.stick_changed.connect(self.handle_stick_widget_changed)
        stick_widget.link_initiated.connect(self.handle_stick_link_initiated)

    def disconnect_stick_widget_signals(self, stick_widget: StickWidget):
        stick_widget.delete_clicked.disconnect(self.handle_stick_widget_delete_clicked)
        stick_widget.stick_changed.disconnect(self.handle_stick_widget_changed)
        stick_widget.link_initiated.disconnect(self.handle_stick_link_initiated)

    def handle_stick_widget_delete_clicked(self, stick: Stick):
        self.camera.remove_stick(stick)

    def set_stick_widgets_mode(self, mode: StickMode):
        self.stick_widget_mode = mode
        for sw in self.stick_widgets:
            sw.set_mode(mode)

    def handle_stick_widget_changed(self, stick_widget: StickWidget):
        self.camera.stick_changed.emit(stick_widget.stick)

    def handle_stick_changed(self, stick: Stick):
        if stick.camera_id != self.camera.id:
            return
        sw = next(filter(lambda _sw: _sw.stick.id == stick.id, self.stick_widgets))
        sw.adjust_line()
        self.stick_length_btn.set_label(str(stick.length_cm) + " cm")
        self.layout_title_area()

    def handle_stick_link_initiated(self, stick_widget: StickWidget):
        self.stick_link_requested.emit(stick_widget)

    def handle_stick_length_clicked(self):
        self.stick_length_input.setVisible(self.stick_length_btn.is_on())
        self.stick_length_input.set_focus()

    def handle_stick_length_input_entered(self):
        length = self.stick_length_input.get_length()
        self.stick_length_btn.set_label(str(length) + " cm")
        self.layout_title_area()
        self.stick_length_btn.click_button(artificial_emit=True)
        # TODO actually set the sticks lengths
        for stick in self.camera.sticks:
            stick.length_cm = length
        self.camera.stick_changed.emit(self.camera.sticks[0]) #TODO handle empty stick list

    def handle_stick_length_input_cancelled(self):
        old_length = int(self.stick_length_btn.label.text()[:-3])
        self.stick_length_input.set_length(old_length)
        self.layout_title_area()
        self.stick_length_btn.click_button(artificial_emit=True)
