# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mainwindow.ui'
#
# Created by: PyQt5 UI code generator 5.14.1
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1046, 671)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.stackedWidget = QtWidgets.QStackedWidget(self.centralwidget)
        self.stackedWidget.setObjectName("stackedWidget")
        self.page = QtWidgets.QWidget()
        self.page.setObjectName("page")
        self.stackedWidget.addWidget(self.page)
        self.page_2 = QtWidgets.QWidget()
        self.page_2.setObjectName("page_2")
        self.stackedWidget.addWidget(self.page_2)
        self.horizontalLayout.addWidget(self.stackedWidget)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1046, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.toolBar = QtWidgets.QToolBar(MainWindow)
        self.toolBar.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.toolBar.setObjectName("toolBar")
        MainWindow.addToolBar(QtCore.Qt.TopToolBarArea, self.toolBar)
        self.actionAdd_camera = QtWidgets.QAction(MainWindow)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/icons/camera-photo.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionAdd_camera.setIcon(icon)
        self.actionAdd_camera.setObjectName("actionAdd_camera")
        self.actionOpen_dataset = QtWidgets.QAction(MainWindow)
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/icons/document-open.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionOpen_dataset.setIcon(icon1)
        self.actionOpen_dataset.setObjectName("actionOpen_dataset")
        self.actionSave_dataset = QtWidgets.QAction(MainWindow)
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap(":/icons/document-save.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionSave_dataset.setIcon(icon2)
        self.actionSave_dataset.setObjectName("actionSave_dataset")
        self.actionClose_dataset = QtWidgets.QAction(MainWindow)
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap(":/icons/dialog-close.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionClose_dataset.setIcon(icon3)
        self.actionClose_dataset.setObjectName("actionClose_dataset")
        self.actionExport_to_JSON = QtWidgets.QAction(MainWindow)
        self.actionExport_to_JSON.setObjectName("actionExport_to_JSON")
        self.toolBar.addAction(self.actionAdd_camera)
        self.toolBar.addAction(self.actionOpen_dataset)
        self.toolBar.addAction(self.actionSave_dataset)
        self.toolBar.addAction(self.actionClose_dataset)
        self.toolBar.addAction(self.actionExport_to_JSON)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Antarstick"))
        self.toolBar.setWindowTitle(_translate("MainWindow", "toolBar"))
        self.actionAdd_camera.setText(_translate("MainWindow", "Add camera"))
        self.actionAdd_camera.setShortcut(_translate("MainWindow", "Ctrl+A"))
        self.actionOpen_dataset.setText(_translate("MainWindow", "Open dataset"))
        self.actionOpen_dataset.setShortcut(_translate("MainWindow", "Ctrl+O"))
        self.actionSave_dataset.setText(_translate("MainWindow", "Save dataset"))
        self.actionSave_dataset.setShortcut(_translate("MainWindow", "Ctrl+S"))
        self.actionClose_dataset.setText(_translate("MainWindow", "Close dataset"))
        self.actionExport_to_JSON.setText(_translate("MainWindow", "Export to JSON"))
import resources_rc
