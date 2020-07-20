# This Python file uses the following encoding: utf-8
import sys
from pathlib import Path
import json
from typing import List
import os

from PyQt5.QtWidgets import (QAction, QApplication, QFileDialog, QMainWindow,
                             QToolBar, QMenu, QToolButton, QMessageBox)
from PyQt5.QtCore import QThreadPool, QRunnable
from PyQt5.Qt import QKeySequence

from camera_processing.widgets.camera_processing_widget import \
    CameraProcessingWidget
from dataset import Dataset



class CameraLoadWorker(QRunnable):

    def __init__(self, dataset: Dataset, folder: Path) -> None:
        QRunnable.__init__(self)
        self.dataset = dataset
        self.folder = folder

    def run(self) -> None:
        self.dataset.add_camera(self.folder)


class MainWindow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.toolbar = QToolBar()
        self.action_ = QAction("&Add camera")
        self.action_.triggered.connect(self.handle_add_camera_triggered)
        self.toolbar.addAction(self.action_)
        self.open_dataset_btn = QToolButton()

        self.recent_datasets: List[Path] = []
        _recent_datasets: List[Path] = MainWindow.get_config_state()
        self.recent_menu = QMenu()
        self.recent_menu.setToolTipsVisible(True)
        self.dataset_actions = []

        for ds in _recent_datasets:
            self.add_dataset_entry_to_recent(ds)
        self.recent_menu.triggered.connect(lambda action: self.open_dataset(Path(action.toolTip())))

        self.open_dataset_btn.setMenu(self.recent_menu if len(self.recent_menu.actions()) > 0 else None)

        self.open_dataset_btn.setMenu(self.recent_menu)
        self.open_dataset_btn.setPopupMode(QToolButton.MenuButtonPopup)
        self.save_action = self.toolbar.addAction("&Save dataset")
        self.save_action.setShortcut(QKeySequence.fromString("Ctrl+S"))
        self.save_action.triggered.connect(self.handle_save_dataset_triggered)

        self.open_dataset_action = QAction("&Open dataset")
        self.open_dataset_action.setShortcut(QKeySequence.fromString("Ctrl+O"))
        self.open_dataset_btn.setDefaultAction(self.open_dataset_action)
        self.open_dataset_action.triggered.connect(self.handle_open_dataset_triggered)

        self.toolbar.addWidget(self.open_dataset_btn)
        self.addToolBar(self.toolbar)
        self.dataset = Dataset()
        self.analyzer_widget = CameraProcessingWidget(self.dataset)
        self.setCentralWidget(self.analyzer_widget)
        self.thread_pool = QThreadPool()

    def handle_save_dataset_triggered(self, checked: bool):
        if self.dataset.path == Path("."):
            file_dialog = QFileDialog(self)
            file_dialog.setWindowTitle("Save dataset")
            file_dialog.setFileMode(QFileDialog.AnyFile)
            file_dialog.setNameFilter("*.json")
            file_dialog.setAcceptMode(QFileDialog.AcceptSave)
            if file_dialog.exec_():
                file_path = Path(file_dialog.selectedFiles()[0])
                self.dataset.save_as(file_path)
                self.add_dataset_entry_to_recent(file_path)
                self.open_dataset_btn.setMenu(self.recent_menu)
                self.save_state()
        else:
            self.dataset.save()

    def handle_add_camera_triggered(self, checked: bool):
        file_dialog = QFileDialog(self)
        file_dialog.setFileMode(QFileDialog.Directory)
        if file_dialog.exec_():
            self.setWindowTitle(file_dialog.selectedFiles()[0])
            self.dataset.add_camera(Path(file_dialog.selectedFiles()[0]))

    def handle_open_dataset_triggered(self, checked: bool):
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter("*.json")
        file_dialog.setFileMode(QFileDialog.AnyFile)
        file_dialog.setWindowTitle("Open dataset file")
        if file_dialog.exec_():
            file_path = Path(file_dialog.selectedFiles()[0])
            self.open_dataset(file_path)
            #self.dataset.load_from(Path(file_path))  # TODO handle the bool return value
            self.setWindowTitle(str(self.dataset.path))
            self.add_dataset_entry_to_recent(file_path)
            self.open_dataset_btn.setMenu(self.recent_menu)
            self.save_state()

    def connect_dataset_signals(self):
        self.dataset.camera_added.connect(self.handle_camera_added)

    def save_state(self):
        try:
            with open(Path(sys.argv[0]).parent / 'state.json', 'w') as f:
                json.dump(list(map(lambda p: str(p), self.recent_datasets)), f)
        except PermissionError:
            pass

    def add_dataset_entry_to_recent(self, p: Path):
        if p not in self.recent_datasets:
            self.recent_datasets.append(p)
            action = QAction(str(p.name))
            action.setToolTip(str(p))
            self.dataset_actions.append(action)
            self.recent_menu.addAction(action)

    def open_dataset(self, path: Path):
        if len(self.dataset.cameras) > 0:
            self.dataset.save()

        if not os.access(path, mode=os.F_OK):
            msg_box = QMessageBox(QMessageBox.Warning, 'File not found', f'The file {str(path)} does not exist.',
                                  QMessageBox.Open | QMessageBox.Close)

            if msg_box.exec_() == QMessageBox.Open:
                self.open_dataset_action.trigger()
            else:
                msg_box.close()
            return
        elif not os.access(path, mode=os.W_OK):
            msg_box = QMessageBox(QMessageBox.Warning, 'Permission denied', f'The application can\'t modify the '
                                                                            f'file {str(path)}', QMessageBox.Close)
            msg_box.exec_()
            return
        self.analyzer_widget.cleanup()
        self.dataset = Dataset()
        self.analyzer_widget.set_dataset(self.dataset)
        if not self.dataset.load_from(path):
            self.analyzer_widget.cleanup()

    @staticmethod
    def get_config_state():
        try:
            with open(Path(sys.argv[0]).parent / 'state.json', 'r') as f:
                state = json.load(f)
                return list(map(lambda p: Path(p), list(state)))
        except FileNotFoundError:
            return []


if __name__ == "__main__":
    app = QApplication([])

    window = MainWindow()
    window.showMaximized()
    sys.exit(app.exec_())
