import multiprocessing
from typing import List, Tuple, Optional

from PyQt5.QtCore import QRectF, QMarginsF, QPointF, pyqtSignal, Qt
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QKeyEvent
from PyQt5.QtWidgets import QGraphicsObject, QGraphicsPixmapItem, QStyleOptionGraphicsItem, QWidget, QGraphicsItem
import numpy as np

from camera import Camera
from camera_processing.widgets.button import Button, ButtonColor
from camera_processing.widgets.stick_widget import StickWidget, StickMode
from stick import Stick
from camera_processing.antarstick_processing import Measurement


class SplitView(QGraphicsObject):
    confirmed = pyqtSignal()
    skipped = pyqtSignal(bool)

    def __init__(self, result: Measurement, parent=None):
        super().__init__(parent)
        self.measurement = result
        self.camera = result.camera

        if len(result.last_img) == 0:
            self.source_img = self.camera.folder / self.camera.sticks[0].view
            self.source_sticks = self.camera.sticks
        else:
            self.source_img = self.camera.folder / result.last_img
            self.source_sticks = result.last_valid_sticks
        self.target_img = self.camera.folder / result.current_img

        self.source_pixmap = QGraphicsPixmapItem(parent=self)
        self.target_pixmap = QGraphicsPixmapItem(parent=self)

        self.target_sticks = result.sticks_to_confirm

        self.source_box, self.target_box = self._sticks_bounding_box()

        self.source_widgets: List[StickWidget] = []
        self.target_widgets: List[StickWidget] = []

        self.confirm_button = Button('btn_confirm', "Confirm(Enter)", parent=self)
        self.skip_button = Button('btn_skip', "Skip(Esc)", parent=self)
        self.skip_batch_button = Button('btn_skip_batch', "Skip this batch", parent=self)

        self.confirm_button.clicked.connect(self.handle_confirmed)
        self.skip_button.clicked.connect(self.handle_skipped)
        self.skip_batch_button.clicked.connect(self.handle_skipped_batch)

    def initialise(self):
        top_source = self.source_box[0, 1]
        bottom_source = self.source_box[1, 1]
        print(f's: {self.source_img}, t: {self.target_img}')
        source_pixmap = QPixmap(str(self.source_img))
        source_pixmap = source_pixmap.scaledToWidth(int(0.5 * source_pixmap.width()))
        source_pixmap = source_pixmap.copy(0, top_source, source_pixmap.width(), bottom_source - top_source)
        self.source_pixmap = QGraphicsPixmapItem(source_pixmap, self)
        self.source_pixmap.setPos(0, 0)

        top_target = self.target_box[0, 1]
        bottom_target = self.target_box[1, 1]

        target_pixmap = QPixmap(str(self.target_img))
        target_pixmap = target_pixmap.scaledToWidth(int(0.5 * target_pixmap.width()))
        target_pixmap = target_pixmap.copy(0, top_target, target_pixmap.width(), bottom_target - top_target)
        self.target_pixmap = QGraphicsPixmapItem(target_pixmap, self)
        self.target_pixmap.setPos(0, bottom_source - top_source)

        for st in self.source_sticks:
            st.translate(np.array([0, -top_source]))

        for st in self.target_sticks:
            st.translate(np.array([0, -top_target]))

        self.source_widgets = list(map(lambda stick: StickWidget(stick, self.source_pixmap), self.source_sticks))
        self.target_widgets = list(map(lambda stick: StickWidget(stick, self.target_pixmap), self.target_sticks))

        #for sw in self.source_widgets:
        #    #sw.moveBy(0, -top_source)
        #    sw.set_top(sw.top_handle.pos() - QPointF(0, -top_source))
        #    sw.set_bottom(sw.bottom_handle.pos() - QPointF(0, -top_source))

        #for sw in self.target_widgets:
        #    #sw.moveBy(0, -top_target)
        #    sw.set_top(sw.top_handle.pos() - QPointF(0, -top_target))
        #    sw.set_bottom(sw.bottom_handle.pos() - QPointF(0, -top_target))

        f = 1 / 3
        self.confirm_button.set_height(16)
        self.confirm_button.set_width(int(f * self.target_pixmap.boundingRect().width()))
        self.confirm_button.setPos(self.target_pixmap.pos() + QPointF(0, self.target_pixmap.boundingRect().height()))
        self.skip_button.set_height(16)
        self.skip_button.set_width(int(f * self.target_pixmap.boundingRect().width()))
        self.skip_button.setPos(self.target_pixmap.pos() + QPointF(self.confirm_button.boundingRect().width(),
                                                                   self.target_pixmap.boundingRect().height()))
        self.skip_batch_button.set_height(16)
        self.skip_batch_button.set_width(int(f * self.target_pixmap.boundingRect().width()))
        self.skip_batch_button.setPos(self.skip_button.pos() + QPointF(self.skip_button.boundingRect().width(), 0))

        self.confirm_button.setVisible(True)
        self.confirm_button.set_base_color([ButtonColor.GREEN])
        self.confirm_button.setFlag(QGraphicsItem.ItemIgnoresTransformations, False)

        self.skip_button.setVisible(True)
        self.skip_button.set_base_color([ButtonColor.RED])
        self.skip_button.setFlag(QGraphicsItem.ItemIgnoresTransformations, False)

        self.skip_batch_button.setVisible(True)
        self.skip_batch_button.set_base_color([ButtonColor.RED])
        self.skip_batch_button.setFlag(QGraphicsItem.ItemIgnoresTransformations, False)
        self.grabKeyboard()

    def _sticks_bounding_box(self) -> Tuple[np.ndarray, np.ndarray]:
        top_left1 = np.min(list(map(lambda s: s.top, self.source_sticks)), axis=0)
        bottom_right1 = np.max(list(map(lambda s: s.bottom, self.source_sticks)), axis=0)

        top_left2 = np.min(list(map(lambda s: s.top, self.target_sticks)), axis=0)
        bottom_right2 = np.max(list(map(lambda s: s.bottom, self.target_sticks)), axis=0)

        return np.array([top_left1 - [0, 30], bottom_right1 + [0, 30]]), np.array([top_left2 - [0, 30], bottom_right2 + [0, 30]])

    def boundingRect(self) -> QRectF:
        #return self.source_pixmap.boundingRect().united(self.target_pixmap.boundingRect())
        btn_rects = self.confirm_button.boundingRect().united(self.skip_button.boundingRect()).\
            translated(self.confirm_button.pos())
        return self.source_pixmap.boundingRect().\
            united(self.mapRectFromItem(self.target_pixmap, self.target_pixmap.boundingRect())).united(btn_rects)

    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem, widget: Optional[QWidget] = ...) -> None:
        pass
        #painter.setPen(QPen(QColor(0, 255, 0), 2.0))
        #painter.drawRect(self.boundingRect().marginsAdded(QMarginsF(10, 10, 10, 10)))
        #painter.setPen(QPen(QColor(0, 0, 255), 2.0))
        #painter.drawRect(self.source_pixmap.boundingRect().marginsAdded(QMarginsF(8, 8, 8, 8)))
        #painter.setPen(QPen(QColor(0, 255, 255), 1.0))
        #painter.drawRect(self.target_pixmap.boundingRect().marginsAdded(QMarginsF(5, 5, 5, 5)))

    def set_stick_widgets_mode(self, mode: StickMode):
        for sw in self.source_widgets:
            sw.set_mode(mode)
        for sw in self.target_widgets:
            sw.set_mode(mode)

    def _offset_sticks(self):
        offset = np.array([0, self.source_box[0, 1]])
        for st in self.source_sticks:
            st.translate(offset)
        offset[1] = self.target_box[0, 1]
        for st in self.target_sticks:
            st.translate(offset)

    def handle_confirmed(self):
        self.ungrabKeyboard()
        self._offset_sticks()
        self.confirmed.emit()

    def handle_skipped(self):
        self.ungrabKeyboard()
        self._offset_sticks()
        self.skipped.emit(False)

    def handle_skipped_batch(self):
        self.ungrabKeyboard()
        self._offset_sticks()
        self.skipped.emit(True)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key_Enter or event.key() == Qt.Key_Return:
            self.handle_confirmed()
        elif event.key() == Qt.Key_Escape:
            self.handle_skipped()
