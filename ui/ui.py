# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(590, 442)
        MainWindow.setMaximumSize(QtCore.QSize(590, 442))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.frame)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.tabWidget = QtWidgets.QTabWidget(self.frame)
        self.tabWidget.setObjectName("tabWidget")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.tab)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.tabWidget_2 = QtWidgets.QTabWidget(self.tab)
        self.tabWidget_2.setObjectName("tabWidget_2")
        self.tab_3 = QtWidgets.QWidget()
        self.tab_3.setObjectName("tab_3")
        self.widget = QtWidgets.QWidget(self.tab_3)
        self.widget.setGeometry(QtCore.QRect(10, 10, 501, 251))
        self.widget.setObjectName("widget")
        self.groupBox = QtWidgets.QGroupBox(self.widget)
        self.groupBox.setGeometry(QtCore.QRect(0, 0, 271, 251))
        self.groupBox.setTitle("")
        self.groupBox.setObjectName("groupBox")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.groupBox)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.label_4 = QtWidgets.QLabel(self.groupBox)
        self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_4.setObjectName("label_4")
        self.gridLayout_4.addWidget(self.label_4, 3, 0, 1, 1)
        self.label = QtWidgets.QLabel(self.groupBox)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.gridLayout_4.addWidget(self.label, 0, 0, 1, 1)
        self.btn_simulate = QtWidgets.QPushButton(self.groupBox)
        self.btn_simulate.setObjectName("btn_simulate")
        self.gridLayout_4.addWidget(self.btn_simulate, 5, 0, 1, 3)
        self.txt_n = QtWidgets.QDoubleSpinBox(self.groupBox)
        self.txt_n.setDecimals(0)
        self.txt_n.setMaximum(1e+24)
        self.txt_n.setObjectName("txt_n")
        self.gridLayout_4.addWidget(self.txt_n, 0, 1, 1, 2)
        self.txt_n_r = QtWidgets.QDoubleSpinBox(self.groupBox)
        self.txt_n_r.setDecimals(0)
        self.txt_n_r.setMaximum(1e+22)
        self.txt_n_r.setObjectName("txt_n_r")
        self.gridLayout_4.addWidget(self.txt_n_r, 1, 1, 1, 2)
        self.txt_cb = QtWidgets.QDoubleSpinBox(self.groupBox)
        self.txt_cb.setDecimals(0)
        self.txt_cb.setMaximum(1e+21)
        self.txt_cb.setObjectName("txt_cb")
        self.gridLayout_4.addWidget(self.txt_cb, 2, 1, 1, 2)
        self.label_2 = QtWidgets.QLabel(self.groupBox)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.gridLayout_4.addWidget(self.label_2, 1, 0, 1, 1)
        self.txt_n_grid_points = QtWidgets.QDoubleSpinBox(self.groupBox)
        self.txt_n_grid_points.setDecimals(0)
        self.txt_n_grid_points.setMaximum(1e+30)
        self.txt_n_grid_points.setObjectName("txt_n_grid_points")
        self.gridLayout_4.addWidget(self.txt_n_grid_points, 3, 1, 1, 2)
        self.label_3 = QtWidgets.QLabel(self.groupBox)
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")
        self.gridLayout_4.addWidget(self.label_3, 2, 0, 1, 1)
        self.label_5 = QtWidgets.QLabel(self.groupBox)
        self.label_5.setAlignment(QtCore.Qt.AlignCenter)
        self.label_5.setObjectName("label_5")
        self.gridLayout_4.addWidget(self.label_5, 4, 0, 1, 1)
        self.txt_n_states = QtWidgets.QDoubleSpinBox(self.groupBox)
        self.txt_n_states.setDecimals(0)
        self.txt_n_states.setMaximum(1e+31)
        self.txt_n_states.setObjectName("txt_n_states")
        self.gridLayout_4.addWidget(self.txt_n_states, 4, 1, 1, 2)
        self.groupBox_5 = QtWidgets.QGroupBox(self.widget)
        self.groupBox_5.setGeometry(QtCore.QRect(280, 10, 221, 231))
        self.groupBox_5.setTitle("")
        self.groupBox_5.setObjectName("groupBox_5")
        self.gridLayout_9 = QtWidgets.QGridLayout(self.groupBox_5)
        self.gridLayout_9.setObjectName("gridLayout_9")
        self.simul_result = QtWidgets.QPlainTextEdit(self.groupBox_5)
        self.simul_result.setObjectName("simul_result")
        self.gridLayout_9.addWidget(self.simul_result, 0, 0, 1, 1)
        self.tabWidget_2.addTab(self.tab_3, "")
        self.gridLayout_3.addWidget(self.tabWidget_2, 0, 0, 1, 1)
        self.tabWidget.addTab(self.tab, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.tab_2)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.groupBox_2 = QtWidgets.QGroupBox(self.tab_2)
        self.groupBox_2.setTitle("")
        self.groupBox_2.setObjectName("groupBox_2")
        self.gridLayout_8 = QtWidgets.QGridLayout(self.groupBox_2)
        self.gridLayout_8.setObjectName("gridLayout_8")
        self.groupBox_3 = QtWidgets.QGroupBox(self.groupBox_2)
        self.groupBox_3.setTitle("")
        self.groupBox_3.setObjectName("groupBox_3")
        self.gridLayout_6 = QtWidgets.QGridLayout(self.groupBox_3)
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.label_6 = QtWidgets.QLabel(self.groupBox_3)
        self.label_6.setObjectName("label_6")
        self.gridLayout_6.addWidget(self.label_6, 0, 0, 1, 1)
        self.txt_dp_n = QtWidgets.QSpinBox(self.groupBox_3)
        self.txt_dp_n.setObjectName("txt_dp_n")
        self.gridLayout_6.addWidget(self.txt_dp_n, 0, 1, 1, 3)
        self.label_7 = QtWidgets.QLabel(self.groupBox_3)
        self.label_7.setObjectName("label_7")
        self.gridLayout_6.addWidget(self.label_7, 1, 0, 1, 1)
        self.txt_dp_T = QtWidgets.QSpinBox(self.groupBox_3)
        self.txt_dp_T.setObjectName("txt_dp_T")
        self.gridLayout_6.addWidget(self.txt_dp_T, 1, 1, 1, 3)
        self.lbl_dp_nu = QtWidgets.QLabel(self.groupBox_3)
        self.lbl_dp_nu.setObjectName("lbl_dp_nu")
        self.gridLayout_6.addWidget(self.lbl_dp_nu, 2, 0, 1, 1)
        self.txt_dp_nu = QtWidgets.QSpinBox(self.groupBox_3)
        self.txt_dp_nu.setObjectName("txt_dp_nu")
        self.gridLayout_6.addWidget(self.txt_dp_nu, 2, 1, 1, 3)
        self.lbl_dp_nq = QtWidgets.QLabel(self.groupBox_3)
        self.lbl_dp_nq.setObjectName("lbl_dp_nq")
        self.gridLayout_6.addWidget(self.lbl_dp_nq, 3, 0, 1, 1)
        self.txt_dp_nq = QtWidgets.QSpinBox(self.groupBox_3)
        self.txt_dp_nq.setObjectName("txt_dp_nq")
        self.gridLayout_6.addWidget(self.txt_dp_nq, 3, 1, 1, 3)
        self.lbl_dp_npoints = QtWidgets.QLabel(self.groupBox_3)
        self.lbl_dp_npoints.setObjectName("lbl_dp_npoints")
        self.gridLayout_6.addWidget(self.lbl_dp_npoints, 4, 0, 1, 1)
        self.txt_dp_npoints = QtWidgets.QSpinBox(self.groupBox_3)
        self.txt_dp_npoints.setObjectName("txt_dp_npoints")
        self.gridLayout_6.addWidget(self.txt_dp_npoints, 4, 1, 1, 3)
        self.label_8 = QtWidgets.QLabel(self.groupBox_3)
        self.label_8.setObjectName("label_8")
        self.gridLayout_6.addWidget(self.label_8, 5, 0, 1, 1)
        self.btn_B = QtWidgets.QToolButton(self.groupBox_3)
        self.btn_B.setObjectName("btn_B")
        self.gridLayout_6.addWidget(self.btn_B, 5, 1, 1, 1)
        self.lbl_B_path = QtWidgets.QLabel(self.groupBox_3)
        self.lbl_B_path.setText("")
        self.lbl_B_path.setObjectName("lbl_B_path")
        self.gridLayout_6.addWidget(self.lbl_B_path, 5, 2, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(50, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_6.addItem(spacerItem, 5, 3, 1, 1)
        self.gridLayout_8.addWidget(self.groupBox_3, 0, 0, 1, 1)
        self.groupBox_4 = QtWidgets.QGroupBox(self.groupBox_2)
        self.groupBox_4.setTitle("")
        self.groupBox_4.setObjectName("groupBox_4")
        self.gridLayout_7 = QtWidgets.QGridLayout(self.groupBox_4)
        self.gridLayout_7.setObjectName("gridLayout_7")
        self.label_9 = QtWidgets.QLabel(self.groupBox_4)
        self.label_9.setObjectName("label_9")
        self.gridLayout_7.addWidget(self.label_9, 0, 0, 1, 1)
        self.dp_result = QtWidgets.QPlainTextEdit(self.groupBox_4)
        self.dp_result.setObjectName("dp_result")
        self.gridLayout_7.addWidget(self.dp_result, 1, 0, 1, 3)
        self.dp_combo_met = QtWidgets.QComboBox(self.groupBox_4)
        self.dp_combo_met.setEditable(False)
        self.dp_combo_met.setObjectName("dp_combo_met")
        self.gridLayout_7.addWidget(self.dp_combo_met, 0, 1, 1, 2)
        self.gridLayout_8.addWidget(self.groupBox_4, 0, 1, 1, 1)
        self.btn_run_dp = QtWidgets.QPushButton(self.groupBox_2)
        self.btn_run_dp.setObjectName("btn_run_dp")
        self.gridLayout_8.addWidget(self.btn_run_dp, 1, 0, 1, 1)
        self.gridLayout_5.addWidget(self.groupBox_2, 0, 1, 1, 1)
        self.tabWidget.addTab(self.tab_2, "")
        self.gridLayout_2.addWidget(self.tabWidget, 0, 0, 1, 1)
        self.gridLayout.addWidget(self.frame, 0, 0, 1, 1)
        self.lbl_status = QtWidgets.QLabel(self.centralwidget)
        self.lbl_status.setText("")
        self.lbl_status.setObjectName("lbl_status")
        self.gridLayout.addWidget(self.lbl_status, 1, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 590, 22))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuEdit = QtWidgets.QMenu(self.menubar)
        self.menuEdit.setObjectName("menuEdit")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionexit = QtWidgets.QAction(MainWindow)
        self.actionexit.setObjectName("actionexit")
        self.menuFile.addAction(self.actionexit)
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuEdit.menuAction())

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        self.tabWidget_2.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label_4.setText(_translate("MainWindow", "n_grid_points"))
        self.label.setText(_translate("MainWindow", "n"))
        self.btn_simulate.setText(_translate("MainWindow", "Simulate"))
        self.label_2.setText(_translate("MainWindow", "n_r"))
        self.label_3.setText(_translate("MainWindow", "cb"))
        self.label_5.setText(_translate("MainWindow", "n_states"))
        self.tabWidget_2.setTabText(self.tabWidget_2.indexOf(self.tab_3), _translate("MainWindow", "Simulate interpo error mat"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("MainWindow", "Run Simulation"))
        self.label_6.setText(_translate("MainWindow", "n"))
        self.label_7.setText(_translate("MainWindow", "T"))
        self.lbl_dp_nu.setText(_translate("MainWindow", "n_u"))
        self.lbl_dp_nq.setText(_translate("MainWindow", "n_q"))
        self.lbl_dp_npoints.setText(_translate("MainWindow", "n_points"))
        self.label_8.setText(_translate("MainWindow", "B"))
        self.btn_B.setText(_translate("MainWindow", "..."))
        self.label_9.setText(_translate("MainWindow", "Met Type"))
        self.btn_run_dp.setText(_translate("MainWindow", "Run"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("MainWindow", "Dynamic Program"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.menuEdit.setTitle(_translate("MainWindow", "Edit"))
        self.actionexit.setText(_translate("MainWindow", "Exit"))
