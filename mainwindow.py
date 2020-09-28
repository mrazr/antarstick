# This Python file uses the following encoding: utf-8
import sys
from pathlib import Path
import json
from typing import List
import os

from PyQt5.QtWidgets import (QAction, QApplication, QFileDialog, QMainWindow,
                             QToolBar, QMenu, QToolButton, QMessageBox, QPushButton)
from PyQt5.QtCore import QThreadPool, QRunnable
from PyQt5.Qt import QKeySequence, Qt
from PyQt5.QtGui import QCloseEvent, QIcon, QFontDatabase, QFont

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
        family_id = QFontDatabase.addApplicationFont(str(Path(sys.argv[0]).parent / 'camera_processing/TwitterEmoji.ttf'))
        family = QFontDatabase.applicationFontFamilies(family_id)[0]
        self.icon_font = QFont(family)
        self.setWindowTitle('Antarstick')
        self.toolbar = QToolBar()
        self.add_camera = QAction("&Add camera")
        self.add_camera.setShortcut(QKeySequence.fromString('Ctrl+A'))
        self.add_camera.triggered.connect(self.handle_add_camera_triggered)
        self.add_camera.setIcon(QIcon.fromTheme('camera-photo'))
        self.add_camera_btn = QToolButton()
        self.add_camera_btn.setDefaultAction(self.add_camera)
        self.add_camera_btn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        #self.add_camera_btn.clicked.connect(self.handle_add_camera_triggered)
        self.toolbar.addWidget(self.add_camera_btn)
        #self.toolbar.addAction(self.action_)

        self.open_dataset_action = QAction("&Open dataset")
        self.open_dataset_action.setShortcut(QKeySequence.fromString("Ctrl+O"))
        self.open_dataset_action.setIcon(QIcon.fromTheme('document-open'))
        self.open_dataset_action.triggered.connect(self.handle_open_dataset_triggered)

        self.open_dataset_btn = QToolButton()
        self.open_dataset_btn.setDefaultAction(self.open_dataset_action)

        self.state = self.get_config_state()

        self.recent_datasets: List[Path] = []
        self.recent_menu = QMenu()
        self.recent_menu.setToolTipsVisible(True)
        self.dataset_actions = []

        for ds in self.state['recent_datasets']:
            self.add_dataset_entry_to_recent(Path(ds))
        self.recent_menu.triggered.connect(lambda action: self.open_dataset(Path(action.toolTip())))

        self.open_dataset_btn.setMenu(self.recent_menu if len(self.recent_menu.actions()) > 0 else None)

        self.open_dataset_btn.setMenu(self.recent_menu)
        self.open_dataset_btn.setPopupMode(QToolButton.MenuButtonPopup)
        self.save_action = QAction("Save dataset")
        self.save_action.setShortcut(QKeySequence.fromString("Ctrl+S"))
        self.save_action.triggered.connect(self.handle_save_dataset_triggered)
        self.save_action.setIcon(QIcon.fromTheme('document-save'))
        self.save_dataset_btn = QToolButton()
        self.save_dataset_btn.setDefaultAction(self.save_action)
        self.save_dataset_btn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        #self.save_dataset_btn.addAction(self.save_action)
        self.toolbar.addWidget(self.save_dataset_btn)


        self.toolbar.addWidget(self.open_dataset_btn)

        self.dataset = Dataset()
        self.analyzer_widget = CameraProcessingWidget(self.dataset)

        self.toolbar.addWidget(self.analyzer_widget.pause_button)
        self.addToolBar(self.toolbar)
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
            #self.setWindowTitle(file_dialog.selectedFiles()[0])
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
                self.state['recent_datasets'] = list(map(lambda p: str(p), self.recent_datasets))
                json.dump(self.state, f)
                #json.dump(list(map(lambda p: str(p), self.recent_datasets)), f)
        except PermissionError:
            pass

    def add_dataset_entry_to_recent(self, p: Path):
        if p not in self.recent_datasets:
            self.recent_datasets.append(p)
            action = QAction(str(p.name))
            action.setToolTip(str(p))
            self.dataset_actions.append(action)
            self.recent_menu.addAction(action)

    def remove_dataset_entry_from_recent(self, path: Path):
        actions = list(filter(lambda action: action.toolTip() == str(path), self.dataset_actions))
        if len(actions) > 0:
            action = actions[0]
            self.dataset_actions.remove(action)
            self.recent_menu.removeAction(action)
            self.recent_datasets.remove(path)

    def open_dataset(self, path: Path):
        if len(self.dataset.cameras) > 0:
            self.dataset.save()

        if not os.access(path, mode=os.F_OK):
            self.remove_dataset_entry_from_recent(path)
            msg_box = QMessageBox(QMessageBox.Warning, 'File not found', f'The file {str(path)} does not exist.',
                                  QMessageBox.Open | QMessageBox.Close)
            if msg_box.exec_() == QMessageBox.Open:
                self.open_dataset_action.trigger()
            else:
                msg_box.close()
            return
        elif not os.access(path, mode=os.W_OK):
            self.remove_dataset_entry_from_recent(path)
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
                if sorted(list(state.keys())) != ['first_time_startup', 'recent_datasets']:
                    raise AttributeError
                return state
        except (FileNotFoundError, AttributeError) as _:
            return {
                'first_time_startup': True,
                'recent_datasets': [],
            }

    def closeEvent(self, event: QCloseEvent) -> None:
        self.analyzer_widget.cleanup()
        self.save_state()
        os.remove('process.log')
        QMainWindow.closeEvent(self, event)


if __name__ == "__main__":
    app = QApplication([])

    window = MainWindow()
    window.showMaximized()
    sys.exit(app.exec_())
