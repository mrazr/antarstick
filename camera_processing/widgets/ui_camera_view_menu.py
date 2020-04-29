# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'camera_processing/ui/camera_view_menu.ui'
#
# Created by: PyQt5 UI code generator 5.14.2
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_CameraViewMenu(object):
    def setupUi(self, CameraViewMenu):
        CameraViewMenu.setObjectName("CameraViewMenu")
        CameraViewMenu.resize(866, 83)
        self.verticalLayout = QtWidgets.QVBoxLayout(CameraViewMenu)
        self.verticalLayout.setObjectName("verticalLayout")
        self.groupBox = QtWidgets.QGroupBox(CameraViewMenu)
        self.groupBox.setStyleSheet("QGroupBox {\n"
"    background: transparent;\n"
"    border: 2px solid gray;\n"
"}\n"
"\n"
"QGroupBox:title {\n"
"    subcontrol-position: top center;\n"
"}\n"
"\n"
"QToolButton {\n"
"    background-color: rgba(50, 50, 50, 50);\n"
"    border: 2px solid white;\n"
"    padding: 3px;\n"
"}\n"
"\n"
"QToolButton:hover {\n"
"    background-color: rgba(255, 125, 0, 100);\n"
"}")
        self.groupBox.setTitle("")
        self.groupBox.setObjectName("groupBox")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.groupBox)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.btnEditSticks = QtWidgets.QToolButton(self.groupBox)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.btnEditSticks.sizePolicy().hasHeightForWidth())
        self.btnEditSticks.setSizePolicy(sizePolicy)
        self.btnEditSticks.setObjectName("btnEditSticks")
        self.horizontalLayout.addWidget(self.btnEditSticks)
        self.btnShowLinkedCameras = QtWidgets.QToolButton(self.groupBox)
        self.btnShowLinkedCameras.setObjectName("btnShowLinkedCameras")
        self.horizontalLayout.addWidget(self.btnShowLinkedCameras)
        self.btnShowOverlay = QtWidgets.QToolButton(self.groupBox)
        self.btnShowOverlay.setObjectName("btnShowOverlay")
        self.horizontalLayout.addWidget(self.btnShowOverlay)
        self.verticalLayout.addWidget(self.groupBox)

        self.retranslateUi(CameraViewMenu)
        QtCore.QMetaObject.connectSlotsByName(CameraViewMenu)

    def retranslateUi(self, CameraViewMenu):
        _translate = QtCore.QCoreApplication.translate
        CameraViewMenu.setWindowTitle(_translate("CameraViewMenu", "Form"))
        self.btnEditSticks.setText(_translate("CameraViewMenu", "Edit sticks manually"))
        self.btnShowLinkedCameras.setText(_translate("CameraViewMenu", "Show linked cameras"))
        self.btnShowOverlay.setText(_translate("CameraViewMenu", "Show overlay"))
