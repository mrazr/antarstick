# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'camera_view.ui'
#
# Created by: PyQt5 UI code generator 5.14.2
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_CameraView(object):
    def setupUi(self, CameraView):
        CameraView.setObjectName("CameraView")
        CameraView.resize(985, 795)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(CameraView)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.cameraView = QtWidgets.QGraphicsView(CameraView)
        self.cameraView.setMouseTracking(False)
        self.cameraView.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.cameraView.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.cameraView.setRenderHints(QtGui.QPainter.Antialiasing|QtGui.QPainter.TextAntialiasing)
        self.cameraView.setViewportUpdateMode(QtWidgets.QGraphicsView.FullViewportUpdate)
        self.cameraView.setObjectName("cameraView")
        self.verticalLayout_2.addWidget(self.cameraView)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QtWidgets.QLabel(CameraView)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.detectionSensitivitySlider = QtWidgets.QSlider(CameraView)
        self.detectionSensitivitySlider.setMaximum(100)
        self.detectionSensitivitySlider.setProperty("value", 10)
        self.detectionSensitivitySlider.setOrientation(QtCore.Qt.Horizontal)
        self.detectionSensitivitySlider.setTickPosition(QtWidgets.QSlider.NoTicks)
        self.detectionSensitivitySlider.setTickInterval(5)
        self.detectionSensitivitySlider.setObjectName("detectionSensitivitySlider")
        self.horizontalLayout.addWidget(self.detectionSensitivitySlider)
        self.btnFindNonSnow = QtWidgets.QPushButton(CameraView)
        self.btnFindNonSnow.setObjectName("btnFindNonSnow")
        self.horizontalLayout.addWidget(self.btnFindNonSnow)
        self.verticalLayout_2.addLayout(self.horizontalLayout)

        self.retranslateUi(CameraView)
        QtCore.QMetaObject.connectSlotsByName(CameraView)

    def retranslateUi(self, CameraView):
        _translate = QtCore.QCoreApplication.translate
        CameraView.setWindowTitle(_translate("CameraView", "Form"))
        self.label.setText(_translate("CameraView", "Minimum stick height"))
        self.btnFindNonSnow.setText(_translate("CameraView", "Detect"))
