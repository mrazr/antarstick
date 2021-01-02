from enum import IntEnum
from multiprocessing import Process, Queue
from pathlib import Path
from time import sleep, time
from typing import List, Optional, Tuple

import exifread
import numpy as np
from PyQt5 import QtGui, QtCore
from PyQt5.QtCore import QObject, pyqtSignal, QThread, Qt, QSize, QRectF, QSizeF, QTimerEvent
from PyQt5.QtGui import QImage, QColor
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


def thumbnail_load_loop(folder: Path, in_queue: Queue, out_queue: Queue):
    while True:
        if in_queue.empty():
            sleep(0.1)
            continue
        while not in_queue.empty():
            idx, img_name = in_queue.get_nowait()
            if idx < 0:
                return
            with open(folder / img_name, "rb") as f:
                tags = exifread.process_file(f)
            out_queue.put_nowait((idx, tags['JPEGThumbnail']))


class ThumbnailState(IntEnum):
    NotLoaded = 0
    Loading = 1
    LoadedBytes = 2
    Decoded = 3


class ThumbnailStorage(QObject):
    thumbnails_loaded = pyqtSignal('PyQt_PyObject')

    def __init__(self, max_thumbnails: int = 200, parent: QObject = None):
        QObject.__init__(self, parent)
        self.thumbnail_count = max_thumbnails
        self.camera = None
        self.thumbnail_bytes: List[Tuple[ThumbnailState, Optional[bytes]]] = []
        self.thumbnails: List[Optional[QImage]] = []
        self.thumbnail_hits: List[int] = []
        self.start = 0
        self.end = 0
        self.lowest_index = 0
        self.highest_index = 0
        self.thumbnail_size = QSize()
        self.thumbnail_placeholder = QImage(':/icons/thumbnail.png')
        self.loader = QThread()
        self.in_queue: Queue = Queue()
        self.out_queue: Queue = Queue()
        self.preload_queue: Queue = Queue()
        self.load_process = Process()
        self.last_thumbnail_sweep: int = 0
        self.recent_thumbnail_idxs: List[int] = []
        self.recent_thumbnail_flags: List[bool] = []

    def initialize(self, camera: Camera) -> QSize:
        self.camera = camera
        self.thumbnail_bytes = [(ThumbnailState.NotLoaded, None) for _ in range(self.camera.get_photo_count())]
        self.thumbnails = [None for _ in range(self.camera.get_photo_count())]
        self.thumbnail_hits = [0 for _ in range(self.camera.get_photo_count())]
        self.recent_thumbnail_flags = [False for _ in range(self.camera.get_photo_count())]
        self.load_thumbnail(0)
        self.load_process = Process(target=thumbnail_load_loop,
                                    args=(self.camera.folder, self.in_queue, self.out_queue))
        self.load_process.start()
        self.startTimer(50)
        dec = QImage()
        dec.loadFromData(self.thumbnail_bytes[0][1])
        self.thumbnail_size = dec.size()
        self.thumbnail_placeholder = self.thumbnail_placeholder.scaledToHeight(self.thumbnail_size.height(), Qt.SmoothTransformation)
        self.last_thumbnail_sweep = time()
        return self.thumbnail_size

    def get_thumbnail(self, idx: int, dragging: bool) -> Optional[QImage]:
        if self.thumbnail_bytes[idx][0] == ThumbnailState.NotLoaded:
            if not dragging:
                self.load_thumbnails(idx, idx)
            return self.thumbnail_placeholder
        state, thumb = self.thumbnail_bytes[idx]
        if state == ThumbnailState.Decoded:
            self.thumbnail_hits[idx] += 1
            if not self.recent_thumbnail_flags[idx]:
                self.recent_thumbnail_idxs.append(idx)
                self.recent_thumbnail_flags[idx] = True
            return self.thumbnails[idx]
        if state == ThumbnailState.Loading:
            return self.thumbnail_placeholder
        pix = QImage()
        pix.loadFromData(thumb)
        self.thumbnails[idx] = pix
        self.thumbnail_hits[idx] += 1
        self.thumbnail_bytes[idx] = (ThumbnailState.Decoded, self.thumbnail_bytes[idx][1])
        if not self.recent_thumbnail_flags[idx]:
            self.recent_thumbnail_idxs.append(idx)
            self.recent_thumbnail_flags[idx] = True
        return pix

    def load_thumbnail(self, idx: int):
        with open(self.camera.folder / self.camera.image_list[idx], "rb") as f:
            tags = exifread.process_file(f)
        thumb = tags['JPEGThumbnail']
        self.thumbnail_bytes[idx] = (ThumbnailState.LoadedBytes, thumb)

    def load_thumbnails(self, first_idx: int, last_idx: int):
        for idx in range(first_idx, last_idx+1):
            if idx < 0 or last_idx >= len(self.camera.image_list):
                continue
            if self.thumbnail_bytes[idx][0] != ThumbnailState.NotLoaded or first_idx < 0 or last_idx >= len(self.camera.image_list):
                continue
            self.preload_queue.put_nowait((idx, self.camera.image_list[idx]))

    def handle_load_finished(self, indexes: List[int], thumbnails: List[bytes]):
        for idx, thumb in zip(indexes, thumbnails):
            self.thumbnail_bytes[idx] = (ThumbnailState.LoadedBytes, thumb)
        self.thumbnails_loaded.emit(indexes)

    def timerEvent(self, a0: QTimerEvent) -> None:
        if not self.out_queue.empty():
            indexes: List[int] = []
            while not self.out_queue.empty():
                idx, thumb = self.out_queue.get_nowait()
                self.thumbnail_bytes[idx] = (ThumbnailState.LoadedBytes, thumb)
                indexes.append(idx)
                self.thumbnail_hits[idx] = 10
            self.thumbnails_loaded.emit(indexes)
        while not self.preload_queue.empty():
            thumbs_to_load = set()
            j = 0
            while j < 50 and not self.preload_queue.empty():
                idx, th = self.preload_queue.get_nowait()
                s = self.thumbnail_bytes[idx][0]
                if s == ThumbnailState.NotLoaded:
                    self.thumbnail_bytes[idx] = (ThumbnailState.Loading, None)
                    thumbs_to_load.add((idx, th))
                    j += 1
            thumbs_to_load = sorted(list(thumbs_to_load), key=lambda t: t[0])
            for t in thumbs_to_load:
                self.in_queue.put_nowait(t)
        if time() - self.last_thumbnail_sweep > 10:
            recent_thumbnail_idxs = self.recent_thumbnail_idxs[-200:]
            min_idx = min(recent_thumbnail_idxs)
            max_idx = max(recent_thumbnail_idxs)
            for i in range(len(self.thumbnails)):
                if min_idx <= i <= max_idx or self.thumbnail_bytes[i][0] != ThumbnailState.Decoded:
                    continue
                self.recent_thumbnail_flags[i] = False
                self.thumbnail_bytes[i] = (ThumbnailState.LoadedBytes, self.thumbnail_bytes[i][1])
                self.thumbnails[i] = None
            self.last_thumbnail_sweep = time()

    def stop(self):
        self.load_process.kill()


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
        quality_color = QColor(00, 00, 00, 200) if quality_color is None else quality_color
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
