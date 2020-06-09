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
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(CameraView)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.splitter = QtWidgets.QSplitter(CameraView)
        self.splitter.setOrientation(QtCore.Qt.Horizontal)
        self.splitter.setObjectName("splitter")
        self.image_list = QtWidgets.QListView(self.splitter)
        self.image_list.setObjectName("image_list")
        self.layoutWidget = QtWidgets.QWidget(self.splitter)
        self.layoutWidget.setObjectName("layoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.layoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.cam_view_placeholder = QtWidgets.QVBoxLayout()
        self.cam_view_placeholder.setObjectName("cam_view_placeholder")
        self.verticalLayout.addLayout(self.cam_view_placeholder)
        self.horizontalLayout_3.addWidget(self.splitter)

        self.retranslateUi(CameraView)
        QtCore.QMetaObject.connectSlotsByName(CameraView)

    def retranslateUi(self, CameraView):
        _translate = QtCore.QCoreApplication.translate
        CameraView.setWindowTitle(_translate("CameraView", "Form"))
