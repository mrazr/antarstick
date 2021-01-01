# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'camera_view.ui'
#
# Created by: PyQt5 UI code generator 5.14.1
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_CameraView(object):
    def setupUi(self, CameraView):
        CameraView.setObjectName("CameraView")
        CameraView.resize(985, 795)
        self.horizontalLayout = QtWidgets.QHBoxLayout(CameraView)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.imageListLayout = QtWidgets.QVBoxLayout()
        self.imageListLayout.setSizeConstraint(QtWidgets.QLayout.SetMinimumSize)
        self.imageListLayout.setObjectName("imageListLayout")
        self.viewFilter = QtWidgets.QComboBox(CameraView)
        self.viewFilter.setObjectName("viewFilter")
        self.imageListLayout.addWidget(self.viewFilter)
        self.image_list = QtWidgets.QListView(CameraView)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.image_list.sizePolicy().hasHeightForWidth())
        self.image_list.setSizePolicy(sizePolicy)
        self.image_list.setStyleSheet("")
        self.image_list.setSelectionMode(QtWidgets.QAbstractItemView.ContiguousSelection)
        self.image_list.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.image_list.setObjectName("image_list")
        self.imageListLayout.addWidget(self.image_list)
        self.horizontalLayout.addLayout(self.imageListLayout)
        self.graphicsViewLayout = QtWidgets.QVBoxLayout()
        self.graphicsViewLayout.setObjectName("graphicsViewLayout")
        self.horizontalLayout.addLayout(self.graphicsViewLayout)

        self.retranslateUi(CameraView)
        QtCore.QMetaObject.connectSlotsByName(CameraView)

    def retranslateUi(self, CameraView):
        _translate = QtCore.QCoreApplication.translate
        CameraView.setWindowTitle(_translate("CameraView", "Form"))
