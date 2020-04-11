# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'camera_view.ui'
##
## Created by: Qt User Interface Compiler version 5.14.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import (QCoreApplication, QDate, QDateTime, QMetaObject,
    QObject, QPoint, QRect, QSize, QTime, QUrl, Qt)
from PySide2.QtGui import (QBrush, QColor, QConicalGradient, QCursor, QFont,
    QFontDatabase, QIcon, QKeySequence, QLinearGradient, QPalette, QPainter,
    QPixmap, QRadialGradient)
from PySide2.QtWidgets import *


class Ui_CameraView(object):
    def setupUi(self, CameraView):
        if not CameraView.objectName():
            CameraView.setObjectName(u"CameraView")
        CameraView.resize(985, 795)
        self.verticalLayout_2 = QVBoxLayout(CameraView)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.cameraView = QGraphicsView(CameraView)
        self.cameraView.setObjectName(u"cameraView")
        self.cameraView.setMouseTracking(False)
        self.cameraView.setRenderHints(QPainter.Antialiasing|QPainter.TextAntialiasing)

        self.verticalLayout_2.addWidget(self.cameraView)

        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.label = QLabel(CameraView)
        self.label.setObjectName(u"label")

        self.horizontalLayout.addWidget(self.label)

        self.detectionSensitivitySlider = QSlider(CameraView)
        self.detectionSensitivitySlider.setObjectName(u"detectionSensitivitySlider")
        self.detectionSensitivitySlider.setMaximum(100)
        self.detectionSensitivitySlider.setValue(10)
        self.detectionSensitivitySlider.setOrientation(Qt.Horizontal)
        self.detectionSensitivitySlider.setTickPosition(QSlider.NoTicks)
        self.detectionSensitivitySlider.setTickInterval(5)

        self.horizontalLayout.addWidget(self.detectionSensitivitySlider)

        self.btnFindNonSnow = QPushButton(CameraView)
        self.btnFindNonSnow.setObjectName(u"btnFindNonSnow")

        self.horizontalLayout.addWidget(self.btnFindNonSnow)


        self.verticalLayout_2.addLayout(self.horizontalLayout)


        self.retranslateUi(CameraView)

        QMetaObject.connectSlotsByName(CameraView)
    # setupUi

    def retranslateUi(self, CameraView):
        CameraView.setWindowTitle(QCoreApplication.translate("CameraView", u"Form", None))
        self.label.setText(QCoreApplication.translate("CameraView", u"Minimum stick height", None))
        self.btnFindNonSnow.setText(QCoreApplication.translate("CameraView", u"Detect", None))
    # retranslateUi

