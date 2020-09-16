from camera_processing.widgets import ui_camera_view_menu
from PyQt5.QtWidgets import QWidget

class CameraViewMenu(QWidget):

    def __init__(self):
        QWidget.__init__(self)

        self.ui = ui_camera_view_menu.Ui_CameraViewMenu()
        self.ui.setupUi(self)