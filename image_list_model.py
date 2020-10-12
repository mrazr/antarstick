from PyQt5.Qt import QAbstractListModel, QWidget, QModelIndex, QAbstractTableModel
from PyQt5.QtCore import Qt, QSize
from typing import List, Optional
from pathlib import Path
import random
from os import scandir
import sys

from PyQt5.QtGui import QBrush, QColor, QFontDatabase, QFont, QIcon

from camera import Camera
import resources_rc


class ImageListModel(QAbstractTableModel):

    def __init__(self, parent: QWidget = None):
        QAbstractTableModel.__init__(self, parent)

        self.image_names: List[Path] = []
        self.processed_images_count: int = 0
        self.camera: Optional[Camera] = None
        #family_id = QFontDatabase.addApplicationFont(':/fonts/camera_processing/TwitterEmoji.ttf')
        #fam = QFontDatabase.applicationFontFamilies(family_id)[0]
        #self.emoji_font = QFont(fam)
        #self.emoji_font.setPointSize(16)
        #self.snow = "❄"
        self.snow = QIcon(':/icons/snowflake.svg')
        self.sun = QIcon(':/icons/sun.svg')
        self.moon = QIcon(':/icons/moon.svg')
        #self.sun = "☀"
        #self.moon = "🌙"
        #self.hourglass = "⏳"

    def initialize(self, camera: Camera, processed_count: int) -> bool:
        self.camera = camera
        folder = self.camera.folder
        self.image_names.clear()
        self.processed_images_count = processed_count
        if not folder.exists():
            raise FileNotFoundError("This should not happen")
        #only_images = filter(lambda f: not f.is_dir(), scandir(folder))
        self.beginResetModel()
        #self.image_names = list(map(lambda f: Path(f.path), sorted(only_images, key=lambda f: f.name)))
        self.image_names = list(map(lambda name: self.camera.folder / name, self.camera.image_list))
        self.endResetModel()

    def rowCount(self, parent: QModelIndex = QModelIndex()):
        return len(self.image_names)

    def columnCount(self, parent: QModelIndex = QModelIndex()):
        return 3

    def data(self, index: QModelIndex, role=Qt.DisplayRole):
        if not index.isValid():
            return None

        if role == Qt.DisplayRole:
            if index.column() == 0:
                return self.image_names[index.row()].name
            elif index.column() == 1:
                return None
                text = ""
                daytime = self.camera.photo_is_daytime(self.image_names[index.row()].name)
                if daytime is None:
                    text = self.hourglass
                else:
                    text += self.sun if daytime else self.moon
                    snow = self.camera.photo_is_snow(self.image_names[index.row()].name)
                    text += " " + self.snow if snow else ""

                #if daytime is None:
                #    text += self.hourglass
                #else:
                #    text += self.sun if daytime else self.moon

                #snow = self.camera.photo_is_snow(self.image_names[index.row()].name)
                #if snow is None:
                #    text += " " + self.hourglass
                #else:
                #    text += " " + self.snow if snow else ""
                return text
                #if index.row() % 2 == 0:
                #    return self.sun + self.snow
                #else:
                #    return self.moon + self.snow
            #elif index.column() == 2:
            #    return "❄"

        #if role == Qt.SizeHintRole:
        #    if index.column() == 1:
        #        return QSize(64, 64)

        if role == Qt.DecorationRole and index.column() == 1:
            r = random.random()
            if r < .5:
                return self.sun
            return self.moon
        if role == Qt.DecorationRole and index.column() == 2:
            if random.random() > .75:
                return self.snow
            return None

        if role == Qt.SizeHintRole:
            if index.column() == 1 or index.column() == 2:
                return QSize(32, 32)

        if role == Qt.ForegroundRole:
            if index.column() == 1:
                return QBrush(QColor(180, 200, 0))
            elif index.column() == 2:
                return QBrush(QColor(0, 100, 200))

        if role == Qt.UserRole:
            return self.image_names[index.row()]

        #if role == Qt.FontRole:
        #    if index.column() == 1 or index.column() == 2:
        #        return self.emoji_font

        if role == Qt.ForegroundRole and index.row() < self.processed_images_count:
            return QBrush(QColor(0, 150, 0))
        return None

    def set_processed_count(self, count: int):
        self.processed_images_count = count
        self.dataChanged.emit(self.index(0, 0), self.index(self.processed_images_count - 1, 0), [Qt.BackgroundRole])
