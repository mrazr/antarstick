# This Python file uses the following encoding: utf-8
from PySide2 import QtCore
from PySide2 import QtWidgets
import ui_camera_view
from camera import Camera


class CameraViewWidget(QtWidgets.QWidget):
    def __init__(self, camera: Camera):
        QtWidgets.QWidget.__init__(self)
        self.ui = ui_camera_view.Ui_CameraView()
        self.ui.setupUi(self)
        self.camera = camera
