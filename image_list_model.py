from PyQt5.Qt import QAbstractListModel, QWidget, QModelIndex, QVariant
from PyQt5.QtCore import Qt
from typing import List
from pathlib import Path
from os import scandir

from PyQt5.QtGui import QBrush, QColor


class ImageListModel(QAbstractListModel):

    def __init__(self, parent: QWidget = None):
        QAbstractListModel.__init__(self, parent)

        self.image_names: List[Path] = []
        self.processed_images_count: int = 0
    
    def initialize(self, folder: Path, processed_count: int) -> bool:
        self.image_names.clear()
        self.processed_images_count = processed_count
        if not folder.exists():
            raise FileNotFoundError("This should not happen")
        only_images = filter(lambda f: not f.is_dir(), scandir(folder))
        self.beginResetModel()
        self.image_names = list(map(lambda f: Path(f.path), sorted(only_images, key=lambda f: f.name)))
        self.endResetModel()

    def rowCount(self, parent):
        return len(self.image_names)
    
    def data(self, index: QModelIndex, role):
        if not index.isValid():
            return None
        if role == Qt.DisplayRole:
            return self.image_names[index.row()].name
        if role == Qt.UserRole:
            return self.image_names[index.row()]
        if role == Qt.ForegroundRole and index.row() < self.processed_images_count:
            return QBrush(QColor(0, 150, 0))
        return None

    def set_processed_count(self, count: int):
        self.processed_images_count = count
        self.dataChanged.emit(self.index(0, 0), self.index(self.processed_images_count - 1, 0), [Qt.BackgroundRole])
