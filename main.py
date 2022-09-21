from utils.functions import dec2bin, cobb, f_val, Simplex
import numpy as np
from models.delta_simp import DeltaSimpML
from models.dp_mat import DP
from utils.random_sampling_mat import simul_data_res, interpo_err_mat, interErrSim
import matplotlib.pyplot as plt


# n = 3 # change to any dimension you want
# T = 5
# B = np.array([[1,0,-1],[0,1,-1],[0,0,1]]) # change to the B you want
# #
# #
# if np.shape(B)[0] != n or np.shape(B)[1]!= n:
#     raise Exception('Inconsistent B size (n)')
# C = B
# ##############################################################################
# n_u=15
# n_q = 15
# n_points = 50*n
# ##############################################################################
# #A,s_min,s_max,u_max,a,b,tet = sim.simul_data_res(sim,n)
# A, s_min, s_max, u_max, alpha, beta, theta = simul_data_res(n)
# u_disc = np.array([np.array([0.0]*n)+ (i/n_u)*(u_max) for i in range(n_u+1)])
# #print(u_disc)
# f_val = np.array([f_val(u_disc[i],alpha, beta) for i in range(len(u_disc))])
# s_points = np.array([s_min + (i/n_points)*(s_max-s_min) for i in range(n_points+1)])
#
# v_val= np.array([cobb(s_points[i], [1./n for j in range(n)])[0] for i in range(len(s_points))])
#
# Q = np.random.uniform(low=500, high=3000., size=[n_q,n])
# #############################################################################
# myprobs,ti,error=DP().dp_mat(T,B,C,s_min,s_max,u_max,f_val,v_val,s_points,
#              u_disc,Q,'simp2') # met may be changed to met1,met3,met4 or simp2
#
# print('Period 1 optimal value : {}'.format(myprobs[0].objVal))
#
# plt.plot(error)
# plt.show()

# n = 2
# n_r = 5
# cb=1
# n_grid_points=200
# n_states=100
#
# #res = statSimul(B, n_grid_points, n_states)
#
# results = interErrSim(n, n_r, cb, n_grid_points, n_states)
# n = 2
# n_r = 5
# cb=1
# n_grid_points=200
# n_states=100
#
# #res = statSimul(B, n_grid_points, n_states)
#
# results = interErrSim(n, n_r, cb, n_grid_points, n_states)
# print(results)
from PyQt5.QtWidgets import QApplication, QWidget, QMainWindow, QFileDialog
from ui.mainWindow import Window
import sys
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Window()
    window.show()
    app.exec_()