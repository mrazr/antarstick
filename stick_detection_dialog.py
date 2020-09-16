from camera_processing.widgets.stick_detection_params import  Ui_StickDetectionParams

from PyQt5 import QtWidgets


class StickDetectionDialog(QtWidgets.QDialog, Ui_StickDetectionParams):
    def __init__(self, *args, obj=None, **kwargs):
        super(StickDetectionDialog, self).__init__(*args, **kwargs)
        self.setupUi(self)
