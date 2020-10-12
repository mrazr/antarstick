# This Python file uses the following encoding: utf-8
import sys
from pathlib import Path
import json
from typing import List, Optional, Dict
import os

from PyQt5.QtWidgets import (QAction, QApplication, QFileDialog, QMainWindow,
                             QToolBar, QMenu, QToolButton, QMessageBox, QPushButton, QWidget, QSizePolicy, QStyle)
from PyQt5.QtCore import QThreadPool, QRunnable, QResource
from PyQt5.Qt import QKeySequence, Qt
from PyQt5.QtGui import QCloseEvent, QIcon, QFontDatabase, QFont, QPicture, QPixmap

from camera_processing.widgets.camera_processing_widget import \
    CameraProcessingWidget
from dataset import Dataset
from ui_mainwindow import Ui_MainWindow
from ui_startup_page import Ui_StartupPage


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
        #if not QResource.registerResource('./resources.rcc'):
        #    print('failure')
        #family_id = QFontDatabase.addApplicationFont(':/fonts/camera_processing/TwitterEmoji.ttf')
        #family = QFontDatabase.applicationFontFamilies(family_id)[0]
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        #self.icon_font = QFont(family)

        self.ui.actionAdd_camera.triggered.connect(self.handle_add_camera_triggered)
        self.ui.actionOpen_dataset.triggered.connect(self.handle_open_dataset_triggered)
        self.ui.actionSave_dataset.triggered.connect(self.handle_save_dataset_triggered)
        self.ui.actionClose_dataset.triggered.connect(self.handle_close_dataset_triggered)

        self.startup_page = QWidget()
        self.startup_page_ui = Ui_StartupPage()
        self.startup_page_ui.setupUi(self.startup_page)

        self.startup_page_ui.addCamera_button.setDefaultAction(self.ui.actionAdd_camera)
        self.startup_page_ui.openDataset_button.setDefaultAction(self.ui.actionOpen_dataset)
        self.startup_page_ui.openDataset_button.setMenu(None)
        self.startup_page_ui.openDataset_button.setArrowType(Qt.NoArrow)

        self.startup_page_ui.label_recent_datasets_image_placeholder.setPixmap(
            QPixmap(':/icons/document-open-recent.svg'))
        self.startup_page_ui.label_recent_cameras_image_placeholder.setPixmap(
            QPixmap(':/icons/document-open-recent.svg'))

        self.startup_dataset_buttons: List[QToolButton] = []
        self.startup_camera_buttons: List[QToolButton] = []

        self.state = self.get_config_state()

        self.recent_datasets: List[Path] = []
        self.recent_datasets_menu = QMenu()
        self.recent_datasets_menu.setToolTipsVisible(True)
        self.recent_dataset_qactions = []

        self.startup_page_ui.verticalLayout_recentDatasets.removeWidget(self.startup_page_ui.label_noRecentDatasets)
        self.startup_page_ui.label_noRecentDatasets.hide()

        for ds in self.state['recent_datasets']:
            self.add_dataset_entry_to_recent(Path(ds))

        if len(self.recent_datasets) == 0:
            self.startup_page_ui.verticalLayout_recentDatasets.addWidget(self.startup_page_ui.label_noRecentDatasets)
            self.startup_page_ui.label_noRecentDatasets.show()

        self.recent_datasets_menu.triggered.connect(lambda action: self.open_dataset(Path(action.toolTip())))

        self.ui.actionOpen_dataset.setMenu(self.recent_datasets_menu)

        self.recent_cameras: List[Path] = []
        self.recent_cameras_menu = QMenu()
        self.recent_cameras_menu.setToolTipsVisible(True)
        self.recent_cameras_qactions: Dict[str, QAction] = {}

        self.startup_page_ui.verticalLayout_recentCameras.removeWidget(self.startup_page_ui.label_noRecentCameras)
        self.startup_page_ui.label_noRecentCameras.hide()

        for cam in self.state['recent_cameras']:
            self.add_camera_entry_to_recent(Path(cam))

        if len(self.recent_cameras) == 0:
            self.startup_page_ui.verticalLayout_recentCameras.addWidget(self.startup_page_ui.label_noRecentCameras)
            self.startup_page_ui.label_noRecentCameras.show()

        self.recent_cameras_menu.triggered.connect(self.handle_add_recent_camera_triggered)
        self.ui.actionAdd_camera.setMenu(self.recent_cameras_menu)

        self.dataset = Dataset()
        self.analyzer_widget = CameraProcessingWidget()
        self.analyzer_widget.set_dataset(self.dataset)
        self.analyzer_widget.camera_loaded.connect(self.handle_camera_added)

        self.ui.stackedWidget.removeWidget(self.ui.page)
        self.ui.stackedWidget.removeWidget(self.ui.page_2)

        self.ui.stackedWidget.addWidget(self.startup_page)
        self.ui.stackedWidget.addWidget(self.analyzer_widget)
        self.ui.stackedWidget.setCurrentIndex(0)
        self.setCentralWidget(self.ui.stackedWidget)
        self.ui.toolBar.hide()

        #self.setCentralWidget(self.ui.centralwidget)

        #self.open_dataset_btn.setMenu(self.recent_menu if len(self.recent_menu.actions()) > 0 else None)

        #self.open_dataset_btn.setMenu(self.recent_menu)

        #self.setWindowTitle('Antarstick')
        #self.toolbar = QToolBar()
        #self.add_camera = QAction("&Add camera")
        #self.add_camera.setShortcut(QKeySequence.fromString('Ctrl+A'))
        #self.add_camera.triggered.connect(self.handle_add_camera_triggered)
        #self.add_camera.setIcon(QIcon.fromTheme('camera-photo'))
        #self.add_camera_btn = QToolButton()
        #self.add_camera_btn.setDefaultAction(self.add_camera)
        #self.add_camera_btn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        ##self.add_camera_btn.clicked.connect(self.handle_add_camera_triggered)
        #self.toolbar.addWidget(self.add_camera_btn)
        ##self.toolbar.addAction(self.action_)

        #self.open_dataset_action = QAction("&Open dataset")
        #self.open_dataset_action.setShortcut(QKeySequence.fromString("Ctrl+O"))
        #self.open_dataset_action.setIcon(QIcon.fromTheme('document-open'))
        #self.open_dataset_action.triggered.connect(self.handle_open_dataset_triggered)

        #self.open_dataset_btn = QToolButton()
        #self.open_dataset_btn.setDefaultAction(self.open_dataset_action)

        #self.open_dataset_btn.setPopupMode(QToolButton.MenuButtonPopup)
        #self.save_action = QAction("Save dataset")
        #self.save_action.setShortcut(QKeySequence.fromString("Ctrl+S"))
        #self.save_action.triggered.connect(self.handle_save_dataset_triggered)
        #self.save_action.setIcon(QIcon.fromTheme('document-save'))
        #self.save_dataset_btn = QToolButton()
        #self.save_dataset_btn.setDefaultAction(self.save_action)
        #self.save_dataset_btn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        ##self.save_dataset_btn.addAction(self.save_action)
        #self.toolbar.addWidget(self.save_dataset_btn)


        #self.toolbar.addWidget(self.open_dataset_btn)

        #self.toolbar.addWidget(self.analyzer_widget.pause_button)
        #self.addToolBar(self.toolbar)
        #self.setCentralWidget(self.analyzer_widget)
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
                self.save_state()
        else:
            self.dataset.save()

    def handle_add_camera_triggered(self, checked: bool):
        file_dialog = QFileDialog(self)
        file_dialog.setFileMode(QFileDialog.Directory)
        if file_dialog.exec_():
            #if self.analyzer_widget is None:
            #    self.analyzer_widget = CameraProcessingWidget(self.dataset)
            #    self.setCentralWidget(self.analyzer_widget)
            #self.setWindowTitle(file_dialog.selectedFiles()[0])
            if self.dataset.add_camera(Path(file_dialog.selectedFiles()[0])):
                self.add_camera_entry_to_recent(Path(file_dialog.selectedFiles()[0]))

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
            self.ui.actionOpen_dataset.setMenu(self.recent_datasets_menu)
            self.save_state()

    def handle_add_recent_camera_triggered(self, action: QAction):
        if self.dataset.add_camera(Path(action.toolTip())):
            self.add_camera_entry_to_recent(Path(action.toolTip()))

    def connect_dataset_signals(self):
        self.dataset.camera_added.connect(self.handle_camera_added)

    def save_state(self):
        try:
            with open(Path(sys.argv[0]).parent / 'state.json', 'w') as f:
                self.state['recent_datasets'] = list(map(lambda p: str(p), self.recent_datasets))
                self.state['recent_cameras'] = list(map(lambda p: str(p), self.recent_cameras))
                json.dump(self.state, f)
                #json.dump(list(map(lambda p: str(p), self.recent_datasets)), f)
        except PermissionError:
            pass

    def add_camera_entry_to_recent(self, p: Path):
        if self.startup_page_ui.label_noRecentCameras.isVisible():
            self.startup_page_ui.label_noRecentCameras.hide()
            self.startup_page_ui.verticalLayout_recentCameras.removeWidget(self.startup_page_ui.label_noRecentCameras)
        if p not in self.recent_cameras:
            self.recent_cameras.append(p)
            action = QAction(str(p.name))
            action.setToolTip(str(p))
            self.recent_cameras_qactions[str(p)] = action
            self.recent_cameras_menu.addAction(action)
            btn = QToolButton()
            btn.setDefaultAction(action)
            btn.setStyleSheet('font-size: 12pt')
            btn.setText(str(p.name))
            btn.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
            self.startup_page_ui.label_noRecentCameras.hide()
            self.startup_page_ui.verticalLayout_recentCameras.addWidget(btn)
        else:
            action_idx = self.recent_cameras_menu.actions().index(self.recent_cameras_qactions[str(p)])
            #btn = self.startup_page_ui.verticalLayout_recentCameras.itemAt(action_idx)
            btn = self.startup_page_ui.verticalLayout_recentCameras.itemAt(action_idx).widget() #self.startup_page_ui.verticalLayout_recentCameras.removeItem(btn)
            self.startup_page_ui.verticalLayout_recentCameras.removeWidget(btn)
            self.startup_page_ui.verticalLayout_recentCameras.insertWidget(0, btn)
            self.recent_cameras.remove(p)
            self.recent_cameras.insert(0, p)
            self.recent_cameras_menu.clear()
            for i in range(self.startup_page_ui.verticalLayout_recentCameras.count()):
                btn: QToolButton = self.startup_page_ui.verticalLayout_recentCameras.itemAt(i).widget()
                self.recent_cameras_menu.addAction(btn.defaultAction())

    def add_dataset_entry_to_recent(self, p: Path):
        if self.startup_page_ui.label_noRecentDatasets.isVisible():
            self.startup_page_ui.label_noRecentDatasets.hide()
            self.startup_page_ui.verticalLayout_recentDatasets.removeWidget(self.startup_page_ui.label_noRecentDatasets)
        if p not in self.recent_datasets:
            self.recent_datasets.append(p)
            action = QAction(str(p.name))
            action.setToolTip(str(p))
            self.recent_dataset_qactions.append(action)
            self.recent_datasets_menu.addAction(action)
            btn = QToolButton()
            btn.setDefaultAction(action)
            btn.setStyleSheet('font-size: 12pt')
            btn.setText(str(p))
            btn.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
            self.startup_page_ui.verticalLayout_recentDatasets.addWidget(btn)
            #self.startup_page_ui.verticalLayout_recentDatasets.setAlignment(btn, Qt.AlignHCenter)
            self.startup_dataset_buttons.append(btn)

    def remove_dataset_entry_from_recent(self, path: Path):
        actions = list(filter(lambda action: action.toolTip() == str(path), self.recent_dataset_qactions))
        if len(actions) > 0:
            action = actions[0]
            self.recent_dataset_qactions.remove(action)
            self.recent_datasets_menu.removeAction(action)
            self.recent_datasets.remove(path)
        if len(self.recent_datasets) == 0:
            self.startup_page_ui.verticalLayout_recentDatasets.addWidget(self.startup_page_ui.label_noRecentDatasets)
            self.startup_page_ui.label_noRecentDatasets.show()

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

        self.dataset = Dataset()
        #self.analyzer_widget.cleanup()
        self.analyzer_widget.set_dataset(self.dataset)
        #self.ui.stackedWidget.setCurrentIndex(1)
        if not self.dataset.load_from(path):
            self.analyzer_widget.cleanup()
            self.ui.stackedWidget.setCurrentIndex(0)

    @staticmethod
    def get_config_state():
        try:
            with open(Path(sys.argv[0]).parent / 'state.json', 'r') as f:
                state = json.load(f)
                if sorted(list(state.keys())) != ['first_time_startup', 'recent_cameras', 'recent_datasets']:
                    raise AttributeError
                return state
        except (FileNotFoundError, AttributeError) as _:
            return {
                'first_time_startup': True,
                'recent_cameras': [],
                'recent_datasets': [],
            }

    def closeEvent(self, event: QCloseEvent) -> None:
        if self.analyzer_widget is not None:
            self.analyzer_widget.cleanup()
        self.save_state()
        #os.remove('process.log')
        #QMainWindow.closeEvent(self, event)
        super().closeEvent(event)

    def handle_close_dataset_triggered(self):
        self.dataset.save()
        self.analyzer_widget.cleanup()
        self.dataset = Dataset()
        self.analyzer_widget.set_dataset(self.dataset)
        self.ui.stackedWidget.setCurrentIndex(0)
        self.ui.toolBar.hide()

    def handle_camera_added(self):
        self.ui.stackedWidget.setCurrentIndex(1)
        self.ui.toolBar.show()


if __name__ == "__main__":
    app = QApplication([])

    window = MainWindow()
    window.showMaximized()
    sys.exit(app.exec_())
