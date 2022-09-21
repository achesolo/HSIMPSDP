import os

from models.errorMat import ErrorMat
from models.piecewiseApproxMat import Piecewise_Approx
from models.dp_mat import DP
from utils.functions import Node, dec2bin, eval_point, f_val, cobb, simul_fval, interpo_err_mat, gridGen
from scipy.linalg import qr
import numpy as np
import time
import pandas as pd


def ortho(n):
    H = np.random.rand(n, n)
    Q, R = qr(H)
    H = np.matmul(Q, np.diag(np.sign(np.diag(R))))
    return H


def simul_data_res(n):
    H = ortho(n=n)
    D = np.diag(np.diag(np.random.rand(n, n)))
    A = np.matmul(H, np.matmul(D, H.transpose()))
    s_min = np.random.uniform(low=150., high=600., size=[1, n])
    s_max = np.random.uniform(low=800., high=7000., size=[1, n])
    u_max = np.random.uniform(low=0.05 * s_min, high=1.5 * s_max, size=[1, n])
    alpha = np.random.uniform(low=0.7, high=.9, size=[1, n])
    beta = np.random.uniform(low=0.9, high=1.5, size=[1, n])
    theta = np.random.uniform(low=.05, high=0.1, size=[1, n])

    return A, s_min[0], s_max[0], u_max[0], alpha[0], beta[0], theta[0]


def statSimul(B, n_grid_points, n_states, T=10, nsimul=5):
    n = len(B)
    n_q = 15
    n_u = 15
    C = B

    mets = ['met1', 'met2', 'met3', 'met4', 'simp2']
    # mets = ['simp2']
    results = [[] for i in range(len(mets))]
    for k in range(nsimul):
        A, s_min, s_max, u_max, a, b, tet = simul_data_res(n)
        u_disc = np.array([np.array([0.0] * n) + (i / n_u) * (u_max) for i in range(n_u + 1)])
        f_va = np.array([f_val(u_disc[i], a, b) for i in range(len(u_disc))])
        s_points = np.array([s_min + (i / n_grid_points) * (s_max - s_min) for i in range(n_grid_points + 1)])
        v_val = [cobb(s_points[i], [1. / n for j in range(n)])[0]
                 for i in range(len(s_points))]
        Q = np.random.uniform(low=500, high=3000., size=[n_q, n])
        Q0 = np.random.uniform(low=500, high=3000., size=[n_states, n])
        s0 = np.random.uniform(low=s_min, high=s_max, size=[n_states, n])

        for j in range(len(mets)):
            my_probs, time, error = DP().dp_mat(T, B, C, s_min, s_max, u_max, f_va,
                                                 v_val, s_points, u_disc, Q, mets[j])
            stat = simul_fval(my_probs[0], s0, Q0)

            results[j].append((time, stat))

    return reshapeRes(results)


def reshapeRes(results):
    n_met = len(results)
    n_sim = len(results[0])
    reshape_res = []

    for i in range(n_met):
        ti = 0.
        mi = 0.
        ma = 0.0
        av = 0.0
        std = 0.0
        for j in range(n_sim):
            ti += results[i][j][0]
            mi += results[i][j][1][0]
            ma += results[i][j][1][1]
            av += results[i][j][1][2]
            std += results[i][j][1][3]
        reshape_res.append((ti / n_sim, mi / n_sim, ma / n_sim, av / n_sim, std / n_sim))
    return reshape_res


def interErrSim(n, nr, cb, n_grid_points, n_states, m_batch=3, r_simp=5, conv=0):

    tm1 = []
    tm2 = []
    tm3 = []
    tm4 = []
    tsimp = []

    err_m1 = []
    err_m2 = []
    err_m3 = []
    err_m4 = []
    err_simp = []

    for k in range(nr):
        s_min, s_max, A = randData(n)
        s0 = np.random.uniform(low=s_min, high=s_max, size=[n_states, n])

        s_disc, v_disc, s_grad, t = met1_mat(s_min, s_max, n_grid_points, A, conv)
        tm1.append(t)
        E = interpo_err_mat(s_disc, v_disc, s0, s_max, A, conv, cb)
        err_m1.append(E)

        s_disc, v_disc, s_grad, l, t = met2_mat(s_min, s_max, n_grid_points, A, conv)
        tm2.append(t)
        E = interpo_err_mat(s_disc, v_disc, s0, s_max, A, conv, cb)
        err_m2.append(E)

        s_disc, v_disc, s_grad, l, t = met3_mat(s_min, s_max, n_grid_points, m_batch, A, 0)
        tm3.append(t)
        E = interpo_err_mat(s_disc, v_disc, s0, s_max, A, conv, cb)
        err_m3.append(E)

        s_disc, v_disc, s_grad, l, t = met4_mat(s_min, s_max, n_grid_points, r_simp, m_batch, A, 0)
        tm4.append(t)
        E = interpo_err_mat(s_disc, v_disc, s0, s_max, A, conv, cb)
        err_m4.append(E)

        s_disc,v_disc, deltaList, t = gridGen(s_min,s_max,A,n_grid_points,conv,2)
        tsimp.append(t)
        E = interpo_err_mat(np.array(s_disc),v_disc,s0,s_max,A,conv,cb)
        err_simp.append(E)

    d = {"Mean Time": [np.mean(tm1), np.mean(tm2), np.mean(tm3), np.mean(tm4), np.mean(tsimp)],
         "Error M1": np.mean(err_m1, axis=0),
         "Error M2": np.mean(err_m2, axis=0),
         "Error M3": np.mean(err_m3, axis=0),
         "Error M4": np.mean(err_m4, axis=0),
         "Error M5": np.mean(err_simp, axis=0)
         }
    data = pd.DataFrame.from_dict(d, orient='index', columns=None)
    data = data.T
    data.to_excel("simulate_interpo_error_mat.xlsx", sheet_name="error", index=False)
    return f'data written to simulate_interpo_error_mat.xlsx'
    # results = [np.concatenate(([np.mean(tm1)], np.mean(err_m1, axis=0))).tolist(),
    #            np.concatenate(([np.mean(tm2)], np.mean(err_m2, axis=0))).tolist(),
    #            np.concatenate(([np.mean(tm3)], np.mean(err_m3, axis=0))).tolist(),
    #            np.concatenate(([np.mean(tm4)], np.mean(err_m4, axis=0))).tolist(),
    #            np.concatenate(([np.mean(tsimp)],np.mean(err_simp, axis=0))).tolist()]
    # return results

def met1(s_min, s_max, n_max, A=[], conv=0):
    time_1 = time.time()
    nsom = 2 ** (len(s_min))
    n = len(s_max)
    s_disc = []
    v_disc = []
    s_grad = []
    points = [Node() for i in range(nsom)]
    for j in range(nsom):
        x = np.array(dec2bin(j, n))
        points[j].coordinates = np.array(s_min) + np.multiply((np.array(s_max) - \
                                                               np.array(s_min)), np.array(x))
        val, subgrad = eval_point(A, points[j].coordinates, s_max, conv)
        points[j].val = val
        points[j].subgrad = subgrad
        s_disc.append(points[j].coordinates)
        v_disc.append(val)
        s_grad.append(subgrad)
    for i in range(n_max - nsom):
        x_new = np.random.uniform(low=s_min, high=s_max, size=[1, n])[0]
        val, subgrad = eval_point(A, x_new, s_max, conv)
        new_point = Node()
        new_point.coordinates = x_new
        new_point.val = val
        new_point.subgrad = subgrad
        points.append(new_point)
        s_disc.append(x_new)
        v_disc.append(val)
        s_grad.append(subgrad)
    time_2 = time.time()
    return points, s_disc, v_disc, s_grad, time_2 - time_1


def met1_mat(s_min, s_max, n_max, A=[], conv=0):
    time_1 = time.time()
    s_disc, v_disc, s_grad = init_nodes(s_min, s_max, n_max, A, conv)
    n = n = len(s_max)
    nsom = len(s_disc)
    for i in range(n_max - nsom):
        x_new = np.random.uniform(low=s_min, high=s_max, size=[1, n])[0]
        val, subgrad = eval_point(A, x_new, s_max, conv)
        s_disc.append(x_new)
        v_disc.append(val)
        s_grad.append(subgrad)
    time_2 = time.time()
    return s_disc, v_disc, s_grad, time_2 - time_1


def met2_mat(s_min, s_max, n_max, A=[], conv=0):
    time_1 = time.time()
    s_disc, v_disc, s_grad = init_nodes(s_min, s_max, n_max, A, conv)
    n = n = len(s_max)
    nsom = len(s_disc)
    list_delta = []
    for j in range(n_max - nsom):
        x_new = np.random.uniform(low=s_min, high=s_max, size=[1, n])[0]
        objval, ld = Piecewise_Approx(s_disc, v_disc, x_new, conv).lamda()
        err, x, l = ErrorMat(ld, s_disc, v_disc, s_grad, conv).delta()
        val, subgrad = eval_point(A, x, s_max, conv)
        s_disc.append(x)
        v_disc.append(val)
        s_grad.append(subgrad)
        list_delta.append(err)
    time_2 = time.time()
    return s_disc, v_disc, s_grad, list_delta, time_2 - time_1


def met3_mat(s_min, s_max, n_max, m_batch=3, A=[], conv=0):
    time_1 = time.time()
    s_disc, v_disc, s_grad = init_nodes(s_min, s_max, n_max, A, conv)
    n = len(s_max)
    nsom = len(s_disc)
    list_delta = []
    for j in range(n_max - nsom):
        max_err = 0
        y_new = []
        for k in range(m_batch):
            x_new = np.random.uniform(low=s_min, high=s_max, size=[1, n])[0]
            objval, ld = Piecewise_Approx(s_disc, v_disc, x_new, conv).lamda()
            err, x, l = ErrorMat(ld, s_disc, v_disc, s_grad, conv).delta()
            if err > max_err:
                y_new = x
                max_err = err
        val, subgrad = eval_point(A, y_new, s_max, conv)
        list_delta.append(max_err)
        s_disc.append(y_new)
        v_disc.append(val)
        s_grad.append(subgrad)
        time_2 = time.time()
    return s_disc, v_disc, s_grad, list_delta, time_2 - time_1


def met4_mat(s_min, s_max, n_max, r_simp=5, m_batch=3, A=[], conv=0):
    time_1 = time.time()
    s_disc, v_disc, s_grad = init_nodes(s_min, s_max, n_max, A, conv)
    n = len(s_max)
    nsom = len(s_disc)
    list_delta = []
    queue = []
    for j in range(n_max - nsom):
        pool = [] + queue
        for k in range(m_batch):
            x_new = np.random.uniform(low=s_min, high=s_max, size=[1, n])[0]
            objval, ld = Piecewise_Approx(s_disc, v_disc, x_new, conv).lamda()
            err, x, l = ErrorMat(ld, s_disc, v_disc, s_grad, conv).delta()
            val, subgrad = eval_point(A, x, s_max, conv)
            new_point = Node()
            new_point.coordinates = x
            new_point.val = val
            new_point.subgrad = subgrad
            new_point.const = err
            n_candid = len(pool)
            if n_candid > 0:
                if err <= pool[n_candid - 1].const:
                    pool.insert(n_candid - 1, new_point)
                else:
                    for c in range(n_candid - 1):
                        if err >= pool[c].const:
                            pool.insert(c, new_point)
                            break
            else:
                pool.append(new_point)
        s_disc.append(pool[0].coordinates)
        v_disc.append(pool[0].val)
        s_grad.append(pool[0].subgrad)
        list_delta.append(pool[0].const)
        if len(pool) > r_simp:
            queue = pool[1:r_simp]
        else:
            queue = pool[1:]
        time_2 = time.time()
    return s_disc, v_disc, s_grad, list_delta, time_2 - time_1

def randData(n):
    s_min = np.random.uniform(low=150., high=600., size=[1,n])
    s_max = np.random.uniform(low=800., high=7000., size=[1,n])
    A = np.random.rand(n,n)
    H = ortho(n)
    D = np.diag(np.diag(np.random.rand(n,n)))
    A = np.matmul(H, np.matmul(D,H.transpose()))
    return s_min[0],s_max[0],A


def init_nodes(s_min, s_max, n_max, A=[], conv=0):
    nsom = 2 ** (len(s_min))
    if n_max < nsom:
        raise Exception('# of grids points should be>={}'.format(nsom))
    n = len(s_max)
    s_disc = [(np.array(s_min) + np.multiply((np.array(s_max) - np.array(s_min)), np.array(np.array(dec2bin(j, n)))))for j in range(nsom)]
    v_disc = [eval_point(A, (np.array(s_min) + np.multiply((np.array(s_max) - np.array(s_min)),
                                                           np.array(np.array(dec2bin(j, n))))), s_max, conv)[0] for j in range(nsom)]
    s_grad = [eval_point(A, (np.array(s_min) + np.multiply((np.array(s_max) - np.array(s_min)),
                                                           np.array(np.array(dec2bin(j, n))))), s_max, conv)[1] for j in range(nsom)]
    return s_disc, v_disc, s_grad
