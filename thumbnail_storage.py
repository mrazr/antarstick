from pathlib import Path
from typing import List, Optional, Tuple
from time import time

from PyQt5 import QtGui, QtCore
from PyQt5.QtCore import QObject, pyqtSignal, QThread, Qt, QSize, QRect, QRectF, QMargins, QRunnable, QThreadPool, \
    QPoint, QPointF, QSizeF
from PyQt5.QtGui import QPixmap, QImage, QColor
import exifread
import cv2 as cv
import numpy as np
from PyQt5.QtWidgets import QStyledItemDelegate, QStyleOptionViewItem, QStyle

from camera import Camera


class ThumbnailLoader(QThread):
    finished = pyqtSignal([object, object])

    def __init__(self, indexes: List[int], images: List[str], folder: Path):
        QThread.__init__(self)
        self.folder = folder
        self.images = images
        self.indexes = indexes
        self.thumbnails: List[np.ndarray] = []

    def run(self) -> None:
        for img_name in self.images:
            with open(self.folder / img_name, "rb") as f:
                tags = exifread.process_file(f)
            self.thumbnails.append(tags['JPEGThumbnail'])
        self.finished.emit(self.indexes, self.thumbnails)


class ThumbnailStorage(QObject):
    thumbnails_loaded = pyqtSignal('PyQt_PyObject')

    def __init__(self, max_thumbnails: int = 200, parent: QObject = None):
        QObject.__init__(self, parent)
        self.thumbnail_count = max_thumbnails
        self.camera = None
        self.thumbnails: List[Optional[bytes]] = []
        self.start = 0
        self.end = 0
        self.lowest_index = 0
        self.highest_index = 0
        self.thumbnail_size = QSize()
        self.thumbnail_placeholder = QImage(':/icons/thumbnail.png')
        self.loader = QThread()

    def initialize(self, camera: Camera) -> QSize:
        self.camera = camera
        self.thumbnails = [None for _ in range(self.camera.get_photo_count())]
        self.load_thumbnail(0)
        dec = QImage()
        dec.loadFromData(self.thumbnails[0])
        self.thumbnail_size = dec.size()
        self.thumbnail_placeholder = self.thumbnail_placeholder.scaledToHeight(self.thumbnail_size.height(), Qt.SmoothTransformation)
        return self.thumbnail_size

    def get_thumbnail(self, idx: int, dragging: bool) -> Optional[QImage]:
        if self.thumbnails[idx] is None:
            if not dragging:
                self.load_thumbnail(idx)
            else:
                return self.thumbnail_placeholder
        thumb = self.thumbnails[idx]
        pix = QImage()
        pix.loadFromData(thumb)
        return pix

    def load_thumbnail(self, idx: int):
        with open(self.camera.folder / self.camera.image_list[idx], "rb") as f:
            tags = exifread.process_file(f)
        thumb = tags['JPEGThumbnail']
        self.thumbnails[idx] = thumb

    def load_thumbnails(self, first_idx: int, last_idx: int):
        images: List[str] = []
        indexes = []
        for idx in range(first_idx - 10, last_idx + 10):
            if self.thumbnails[idx] is not None or first_idx < 0 or last_idx >= len(self.camera.image_list):
                continue
            images.append(self.camera.image_list[idx])
            indexes.append(idx)
        if len(indexes) == 0:
            return
        self.loader = ThumbnailLoader(indexes, images, self.camera.folder)
        self.loader.finished.connect(self.handle_load_finished)
        self.loader.start()

    def handle_load_finished(self, indexes: List[int], thumbnails: List[bytes]):
        for idx, thumb in zip(indexes, thumbnails):
            self.thumbnails[idx] = thumb
        self.thumbnails_loaded.emit(indexes)


class ThumbnailDelegate(QStyledItemDelegate):
    def __init__(self, thumbnails: ThumbnailStorage, parent: QObject = None):
        QStyledItemDelegate.__init__(self, parent)
        self.thumbnails = thumbnails
        self.snow_indicator = QImage(':/icons/snowflake.svg')
        self.snow_indicator = self.snow_indicator.scaledToWidth(24)

    def sizeHint(self, option: QStyleOptionViewItem, index: QtCore.QModelIndex) -> QtCore.QSize:
        sz = QSize(self.thumbnails.thumbnail_size)
        sz.setHeight(sz.height())# + int(round(1.66 * option.font.pointSize())))
        sz.setWidth(sz.width())
        return sz

    def paint(self, painter: QtGui.QPainter, option: QStyleOptionViewItem, index: QtCore.QModelIndex) -> None:
        img_name = index.data(Qt.DisplayRole)
        thumbnail = index.data(Qt.DecorationRole)
        quality_color = index.data(Qt.BackgroundRole)
        rect = option.rect
        pic_rect = QRectF(rect.center().x() - 0.5 * self.thumbnails.thumbnail_size.width(),
                          rect.y() - 0.0 * self.thumbnails.thumbnail_size.height(),
                          self.thumbnails.thumbnail_size.width(),
                          self.thumbnails.thumbnail_size.height())
        text_rect = QRectF(pic_rect.x(), rect.y() + pic_rect.height() - self.snow_indicator.height(),
                           pic_rect.width(), self.snow_indicator.height())
        painter.setRenderHint(painter.SmoothPixmapTransform, True)
        painter.drawImage(pic_rect, thumbnail)
        painter.fillRect(text_rect, quality_color)
        painter.setPen(QColor(255, 255, 255, 255))
        if index.data(Qt.UserRole + 1):
            indic_rect = QRectF(text_rect.topLeft(),
                                QSizeF(self.snow_indicator.size()))
            painter.drawImage(indic_rect, self.snow_indicator)
        painter.drawText(text_rect, Qt.AlignCenter, img_name)
        if option.state & QStyle.State_Selected:
            color = option.palette.highlight().color()
            color.setAlpha(100)
            painter.fillRect(rect, color)
