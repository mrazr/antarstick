from PyQt5.Qt import QAbstractListModel, QWidget, QModelIndex, QVariant
from PyQt5.QtCore import Qt
from typing import List
from pathlib import Path
from os import scandir

class ImageListModel(QAbstractListModel):

    def __init__(self, parent: QWidget = None):
        QAbstractListModel.__init__(self, parent)

        self.image_names: List[Path] = []
    
    def initialize(self, folder: Path) -> bool:
        self.image_names.clear()
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
        return None
    
