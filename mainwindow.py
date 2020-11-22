# This Python file uses the following encoding: utf-8
import sys
from pathlib import Path
import json
from typing import List, Optional, Dict
import os

from PyQt5.QtWidgets import (QAction, QApplication, QFileDialog, QMainWindow,
                             QToolBar, QMenu, QToolButton, QMessageBox, QPushButton, QWidget, QSizePolicy, QStyle,
                             QProgressBar, QSystemTrayIcon, QLabel, QHBoxLayout)
from PyQt5.QtCore import QThreadPool, QRunnable, QResource
from PyQt5.Qt import QKeySequence, Qt
from PyQt5.QtGui import QCloseEvent, QIcon, QFontDatabase, QFont, QPicture, QPixmap, QColor

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
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

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
            self.startup_page_ui.label_noRecentDatasets.setAlignment(Qt.AlignHCenter)
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
            self.startup_page_ui.label_noRecentCameras.setAlignment(Qt.AlignHCenter)
            self.startup_page_ui.label_noRecentCameras.show()

        self.recent_cameras_menu.triggered.connect(self.handle_add_recent_camera_triggered)
        self.ui.actionAdd_camera.setMenu(self.recent_cameras_menu)

        self.dataset = Dataset()
        self.processing_widget = CameraProcessingWidget()
        self.processing_widget.set_dataset(self.dataset)
        self.processing_widget.camera_loaded.connect(self.handle_camera_added)
        self.processing_widget.processing_started.connect(self.show_progress_bar)
        self.processing_widget.processing_updated.connect(self.update_progress_bar)
        self.processing_widget.processing_stopped.connect(self.hide_progress_bar)
        self.processing_widget.stick_verification_needed.connect(self.notify_user)
        self.processing_widget.no_cameras_open.connect(self.handle_no_cameras_open)

        self.ui.stackedWidget.removeWidget(self.ui.page)
        self.ui.stackedWidget.removeWidget(self.ui.page_2)

        self.ui.stackedWidget.addWidget(self.startup_page)
        self.ui.stackedWidget.addWidget(self.processing_widget)
        self.ui.stackedWidget.setCurrentIndex(0)
        self.setCentralWidget(self.ui.stackedWidget)
        self.ui.toolBar.hide()

        status_bar = self.statusBar()

        self.progress_bar = QProgressBar()

        bad_q_indicator = QPixmap(24, 24)
        bad_q_indicator.fill(QColor(200, 0, 0))
        bad_label = QLabel()
        bad_label.setPixmap(bad_q_indicator)
        ok_q_indicator = QPixmap(24, 24)
        ok_q_indicator.fill(QColor(200, 100, 0))
        ok_label = QLabel()
        ok_label.setPixmap(ok_q_indicator)
        good_q_indicator = QPixmap(24, 24)
        good_q_indicator.fill(QColor(100, 200, 0))
        good_label = QLabel()
        good_label.setPixmap(good_q_indicator)

        status_box = QHBoxLayout()
        indicator_box = QHBoxLayout()

        self.statusBar().hide()
        indicator_box.addWidget(QLabel("Image quality:\t"))
        indicator_box.addWidget(bad_label)
        indicator_box.addWidget(QLabel(" Bad  "))
        indicator_box.addWidget(ok_label)
        indicator_box.addWidget(QLabel(" OK  "))
        indicator_box.addWidget(good_label)
        indicator_box.addWidget(QLabel(" Good  "))
        status_box.addItem(indicator_box)
        status_box.addWidget(self.progress_bar)
        status_box.setAlignment(indicator_box, Qt.AlignLeft)
        status_box.setAlignment(self.progress_bar, Qt.AlignRight)
        status_widget = QWidget()
        status_widget.setLayout(status_box)
        self.statusBar().addPermanentWidget(status_widget, 1)

        self.progress_bar.setFormat("%v / %m")
        self.progress_bar.hide()
        self.sys_tray = QSystemTrayIcon(QIcon(':icons/snowflake.svg'))
        self.sys_tray.show()

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
            if self.dataset.add_camera(Path(file_dialog.selectedFiles()[0])):
                self.add_camera_entry_to_recent(Path(file_dialog.selectedFiles()[0]))

    def handle_open_dataset_triggered(self, checked: bool):
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter("*.json")
        file_dialog.setFileMode(QFileDialog.AnyFile)
        file_dialog.setWindowTitle("Open dataset file")
        if file_dialog.exec_():
            file_path = Path(file_dialog.selectedFiles()[0])
            if not self.open_dataset(file_path):
                return
            #self.dataset.load_from(Path(file_path))  # TODO handle the bool return value
            self.setWindowTitle(str(self.dataset.path))
            self.add_dataset_entry_to_recent(file_path)
            self.ui.actionOpen_dataset.setMenu(self.recent_datasets_menu)
            self.save_state()

    def handle_add_recent_camera_triggered(self, action: QAction):
        if self.dataset.add_camera(Path(action.toolTip())):
            self.add_camera_entry_to_recent(Path(action.toolTip()))
        else:
            QMessageBox.critical(self, "Failed to open camera folder", f'Could not open {action.toolTip()}')

    def connect_dataset_signals(self):
        self.dataset.camera_added.connect(self.handle_camera_added)

    def save_state(self):
        try:
            with open(Path(sys.argv[0]).parent / 'state.json', 'w') as f:
                self.state['recent_datasets'] = list(reversed(list(map(lambda p: str(p), self.recent_datasets))))
                self.state['recent_cameras'] = list(reversed(list(map(lambda p: str(p), self.recent_cameras))))
                json.dump(self.state, f)
        except PermissionError:
            pass

    def add_camera_entry_to_recent(self, p: Path):
        if self.startup_page_ui.label_noRecentCameras.isVisible():
            self.startup_page_ui.label_noRecentCameras.hide()
            self.startup_page_ui.verticalLayout_recentCameras.removeWidget(self.startup_page_ui.label_noRecentCameras)
        if p not in self.recent_cameras:
            self.recent_cameras.insert(0, p)
            action = QAction(str(p.name))
            action.setToolTip(str(p))
            self.recent_cameras_qactions[str(p)] = action
            if len(self.recent_cameras_menu.actions()) > 0:
                self.recent_cameras_menu.insertAction(self.recent_cameras_menu.actions()[0], action)
            else:
                self.recent_cameras_menu.addAction(action)
            btn = QToolButton()
            btn.setDefaultAction(action)
            btn.setStyleSheet('font-size: 12pt')
            btn.setText(str(p.name))
            btn.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
            self.startup_camera_buttons.insert(0, btn)
            self.startup_page_ui.label_noRecentCameras.hide()
            self.startup_page_ui.verticalLayout_recentCameras.insertWidget(0, btn)
        else:
            action_idx = self.recent_cameras_menu.actions().index(self.recent_cameras_qactions[str(p)])
            btn = self.startup_page_ui.verticalLayout_recentCameras.itemAt(action_idx).widget() #self.startup_page_ui.verticalLayout_recentCameras.removeItem(btn)
            self.startup_page_ui.verticalLayout_recentCameras.removeWidget(btn)
            self.startup_page_ui.verticalLayout_recentCameras.insertWidget(0, btn)
            self.recent_cameras.remove(p)
            self.recent_cameras.insert(0, p)
            self.recent_cameras_menu.clear()
            for i in range(self.startup_page_ui.verticalLayout_recentCameras.count()):
                btn: QToolButton = self.startup_page_ui.verticalLayout_recentCameras.itemAt(i).widget()
                self.recent_cameras_menu.addAction(btn.defaultAction())
        if len(self.recent_cameras) > 10:
            self.remove_camera_entry_from_recent(self.recent_cameras[-1])

    def add_dataset_entry_to_recent(self, p: Path):
        if self.startup_page_ui.label_noRecentDatasets.isVisible():
            self.startup_page_ui.label_noRecentDatasets.hide()
            self.startup_page_ui.verticalLayout_recentDatasets.removeWidget(self.startup_page_ui.label_noRecentDatasets)
        if p not in self.recent_datasets:
            self.recent_datasets.insert(0, p)
            action = QAction(str(p.name))
            action.setToolTip(str(p))
            self.recent_dataset_qactions.insert(0, action)
            if len(self.recent_datasets_menu.actions()) > 0:
                self.recent_datasets_menu.insertAction(self.recent_datasets_menu.actions()[0], action)
            else:
                self.recent_datasets_menu.addAction(action)
            btn = QToolButton()
            btn.setDefaultAction(action)
            btn.setStyleSheet('font-size: 12pt')
            btn.setText(str(p))
            btn.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
            self.startup_page_ui.verticalLayout_recentDatasets.insertWidget(0, btn)
            self.startup_dataset_buttons.append(btn)
        if len(self.recent_datasets) > 10:
            self.remove_dataset_entry_from_recent(self.recent_datasets[-1])

    def remove_dataset_entry_from_recent(self, path: Path):
        actions = list(filter(lambda action: action.toolTip() == str(path), self.recent_dataset_qactions))
        if len(actions) > 0:
            action = actions[0]
            btn = list(filter(lambda btn_: btn_.defaultAction() == action, self.startup_dataset_buttons))[0]
            self.startup_dataset_buttons.remove(btn)
            self.startup_page_ui.verticalLayout_recentDatasets.removeWidget(btn)
            self.recent_dataset_qactions.remove(action)
            self.recent_datasets_menu.removeAction(action)
            self.recent_datasets.remove(path)
            btn.setVisible(False)
            btn.deleteLater()
        if len(self.recent_datasets) == 0:
            self.startup_page_ui.verticalLayout_recentDatasets.addWidget(self.startup_page_ui.label_noRecentDatasets)
            self.startup_page_ui.label_noRecentDatasets.setAlignment(Qt.AlignHCenter)
            self.startup_page_ui.label_noRecentDatasets.show()

    def remove_camera_entry_from_recent(self, path: Path):
        action = self.recent_cameras_qactions[str(path)]
        btn = list(filter(lambda btn_: btn_.defaultAction() == action, self.startup_camera_buttons))[0]
        self.startup_camera_buttons.remove(btn)
        self.startup_page_ui.verticalLayout_recentCameras.removeWidget(btn)
        del self.recent_cameras_qactions[str(path)]
        self.recent_cameras_menu.removeAction(action)
        self.recent_cameras.remove(path)
        btn.setVisible(False)
        btn.deleteLater()
        if len(self.recent_cameras) == 0:
            self.startup_page_ui.verticalLayout_recentCameras.addWidget(self.startup_page_ui.label_noRecentCameras)
            self.startup_page_ui.label_noRecentCameras.setAlignment(Qt.AlignHCenter)
            self.startup_page_ui.label_noRecentCameras.show()

    def open_dataset(self, path: Path) -> bool:
        if len(self.dataset.cameras) > 0:
            self.dataset.save()
        if not os.access(path, mode=os.F_OK):
            self.remove_dataset_entry_from_recent(path)
            msg_box = QMessageBox(QMessageBox.Warning, 'File not found', f'The file {str(path)} does not exist.',
                                  QMessageBox.Open | QMessageBox.Close)
            if msg_box.exec_() == QMessageBox.Open:
                self.ui.actionOpen_dataset.trigger()
            else:
                msg_box.close()
            return False
        elif not os.access(path, mode=os.W_OK):
            self.remove_dataset_entry_from_recent(path)
            msg_box = QMessageBox(QMessageBox.Warning, 'Permission denied', f'The application can\'t modify the '
                                                                            f'file {str(path)}', QMessageBox.Close)
            msg_box.exec_()
            return False

        self.dataset = Dataset()
        #self.analyzer_widget.cleanup()
        self.processing_widget.set_dataset(self.dataset)
        #self.ui.stackedWidget.setCurrentIndex(1)
        if not self.dataset.load_from(path):
            self.processing_widget.cleanup()
            self.ui.stackedWidget.setCurrentIndex(0)
            return False
        return True

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
        if self.processing_widget is not None:
            self.processing_widget.cleanup()
        self.save_state()
        #os.remove('process.log')
        #QMainWindow.closeEvent(self, event)
        super().closeEvent(event)

    def handle_close_dataset_triggered(self):
        self.dataset.save()
        self.processing_widget.cleanup()
        self.dataset = Dataset()
        self.processing_widget.set_dataset(self.dataset)
        self.ui.stackedWidget.setCurrentIndex(0)
        self.ui.toolBar.hide()
        self.statusBar().hide()

    def handle_camera_added(self):
        self.ui.stackedWidget.setCurrentIndex(1)
        self.ui.toolBar.show()
        self.statusBar().show()

    def show_progress_bar(self, cam_name: str, photo_count: int):
        self.statusBar().showMessage(f'Processing camera {cam_name}', 0)
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(photo_count)
        self.progress_bar.setValue(0)
        self.progress_bar.show()

    def update_progress_bar(self, value: int):
        self.progress_bar.setValue(value)

    def hide_progress_bar(self, val: int):
        self.statusBar().clearMessage()
        self.progress_bar.hide()

    def notify_user(self, camera_name: str):
        self.sys_tray.showMessage("Verification needed",
                                  f'Camera {camera_name} requires verification of stick positions',
                                  QSystemTrayIcon.NoIcon, 5000)

    def handle_no_cameras_open(self):
        self.ui.stackedWidget.setCurrentIndex(0)
        self.ui.toolBar.hide()
        self.statusBar().hide()


if __name__ == "__main__":
    app = QApplication([])

    window = MainWindow()
    window.showMaximized()
    sys.exit(app.exec_())
