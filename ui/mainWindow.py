import concurrent.futures
import time
from threading import Thread
from utils.functions import f_val, cobb

import numpy as np

from models.dp_mat import DP
from PyQt5.QtCore import pyqtSignal, QThread, QTimer
from PyQt5.QtWidgets import QApplication, QWidget, QMainWindow, QFileDialog
from ui.ui import Ui_MainWindow
import sys
from utils.random_sampling_mat import interErrSim, randData
import pandas as pd
from utils.random_sampling_mat import simul_data_res, interpo_err_mat, interErrSim
import matplotlib.pyplot as plt


class Window(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(Window, self).__init__()
        self.setupUi(self)
        self.timer = QTimer(self)
        self.dp_combo_met.addItems(['simp2'])
        # self.timer.timeout.connect(lambda: self.progressBar.setValue(self.progressBar.value() + 1))
        self.btn_simulate.clicked.connect(self.simulate_interpo_error_mat)

        self.btn_B.clicked.connect(self.read_B_Data)
        self.btn_run_dp.clicked.connect(self.simulate_dp)

    def simulate_interpo_error_mat(self):
        try:
            # self.lbl_status.clear()
            if int(self.txt_n.value()) > 0 and int(self.txt_n_r.value()) > 0 \
                    and int(self.txt_cb.value()) > 0 and int(self.txt_n_grid_points.value()) > 0:
                self.lbl_status.setText("Please Wait !!!")
                self.btn_simulate.setEnabled(True)
                result = interErrSim(int(self.txt_n.value()), int(self.txt_n_r.value()), int(self.txt_cb.value()),
                                     int(self.txt_n_grid_points.value()), int(self.txt_n_states.value()))
                self.simul_result.setPlainText(result)
                self.lbl_status.setText("Done")
        except:
            raise Exception

    def simulate_dp(self):
        try:
            if int(self.txt_dp_n.value()) > 0 and int(self.txt_dp_nu.value()) > 0 \
                    and int(self.txt_dp_nq.value()) > 0 and int(self.txt_dp_npoints.value()) > 0 \
                    and int(self.txt_dp_T.value()) > 0:
                self.btn_run_dp.setEnabled(False)
                self.lbl_status.clear()
                self.lbl_status.setText("Please Wait !!!")
                met_item = self.dp_combo_met.currentText()
                n = int(self.txt_dp_n.value())
                n_u = int(self.txt_dp_nu.value())
                n_q = int(self.txt_dp_nq.value())
                n_points = int(self.txt_dp_npoints.value()) * n
                T = int(self.txt_dp_T.value())
                B = self.B
                C = B

                A, s_min, s_max, u_max, alpha, beta, theta = simul_data_res(n)
                u_disc = np.array([np.array([0.0] * n) + (i / n_u) * (u_max) for i in range(n_u + 1)])
                f_va = np.array([f_val(u_disc[i], alpha, beta) for i in range(len(u_disc))])
                s_points = np.array([s_min + (i / n_points) * (s_max - s_min) for i in range(n_points + 1)])
                v_val = np.array([cobb(s_points[i], [1. / n for j in range(n)])[0] for i in range(len(s_points))])
                Q = np.random.uniform(low=500, high=3000., size=[n_q, n])

                myprobs, ti, error = DP().dp_mat(T, B, C, s_min, s_max, u_max, f_va, v_val, s_points,
                                                 u_disc, Q, met_item)  # met may be changed to met1,met3,met4 or simp2

                self.dp_result.setPlainText(f'Period 1 optimal value {myprobs[0].objVal}')
                plt.plot(error)
                plt.show()
                self.btn_run_dp.setEnabled(True)
                self.lbl_status.setText("Done")
        except:
            raise Exception

    def run_simulate_dp(self):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = executor.submit(self.simulate_dp)

    def run_simulate_interpo_error_mat(self):

        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = executor.submit(self.simulate_interpo_error_mat)

    def open_dialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        filename, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
                                                  "csv Files (*.csv)", options=options)
        if filename:
            with open(filename, 'r') as f:
                return f

    def read_B_Data(self):
        f = self.open_dialog()
        self.lbl_B_path.setText(f.name)
        f = pd.read_csv(f.name, delimiter=',', header=None, index_col=False)
        f.reset_index(drop=True, inplace=True)
        self.B = np.array(f)
