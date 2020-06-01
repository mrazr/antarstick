# This Python file uses the following encoding: utf-8
import sys
from pathlib import Path

from PyQt5.QtWidgets import (QAction, QApplication, QFileDialog, QMainWindow,
                             QToolBar)
from PyQt5.QtCore import QThreadPool, QRunnable

from camera_processing.widgets.camera_processing_widget import \
    CameraProcessingWidget
from dataset import Dataset

import multiprocessing


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
        self.save_action = self.toolbar.addAction("Save dataset")
        self.save_action.triggered.connect(self.handle_save_dataset_triggered)
        self.open_dataset_action = QAction("Open dataset")
        self.open_dataset_action.triggered.connect(self.handle_open_dataset_triggered)
        self.toolbar.addAction(self.open_dataset_action)
        self.addToolBar(self.toolbar)
        self.dataset = Dataset()
        self.analyzer_widget = CameraProcessingWidget(self.dataset)
        self.setCentralWidget(self.analyzer_widget)
        self.thread_pool = QThreadPool()

    def handle_save_dataset_triggered(self, checked: bool):
        if self.dataset.path == Path("."):
            file_dialog = QFileDialog(self)
            file_dialog.setFileMode(QFileDialog.AnyFile)
            file_dialog.setNameFilter("*.json")
            file_dialog.setAcceptMode(QFileDialog.AcceptSave)
            if file_dialog.exec_():
                file_path = Path(file_dialog.selectedFiles()[0])
                self.dataset.save_as(file_path)
        else:
            self.dataset.save()

    def handle_add_camera_triggered(self, checked: bool):
        file_dialog = QFileDialog(self)
        file_dialog.setFileMode(QFileDialog.Directory)
        if file_dialog.exec_():
            self.setWindowTitle(file_dialog.selectedFiles()[0])
            #process = multiprocessing.Process(target=self.dataset.add_camera, args=(Path(file_dialog.selectedFiles()[0]),))
            #self.workers.append(process)
            #process.start()
            #print('triggered')
            #worker = CameraLoadWorker(self.dataset, Path(file_dialog.selectedFiles()[0]))
            #self.thread_pool.start(worker)
            self.dataset.add_camera(Path(file_dialog.selectedFiles()[0]))

    def handle_open_dataset_triggered(self, checked: bool):
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter("*.json")
        file_dialog.setFileMode(QFileDialog.AnyFile)
        if file_dialog.exec_():
            file_path = Path(file_dialog.selectedFiles()[0])
            process = multiprocessing.Process(target=self.dataset.load_from, args=(file_path,))
            process.start()
            #self.dataset.load_from(Path(file_path))  # TODO handle the bool return value
            self.setWindowTitle(str(self.dataset.path))

    def connect_dataset_signals(self):
        self.dataset.camera_added.connect(self.handle_camera_added)


if __name__ == "__main__":
    app = QApplication([])

    window = MainWindow()
    window.showMaximized()
    sys.exit(app.exec_())
