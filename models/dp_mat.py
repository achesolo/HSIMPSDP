import time
import numpy as np
from copy import deepcopy
from models.model_mat import Model_mat
from models.piecewiseApproxMat import Piecewise_Approx
from models.errorMat import ErrorMat
from utils.functions import dec2bin, esper_q, Node, gridGenDyn
import math


class DP:
    def dp_mat(self, T, B, C, s_min, s_max, u_max, f_val, v_val, s_points, u_disc, Q, meth, m_batch=3, r_simp=5,
                 conv=0):
        n = len(B)
        nsom = math.pow(2, n)
        n_points = len(s_points)
        if n_points < nsom:
            raise Exception("number of grid points should be {}".format(nsom))
        my_probs = [0.0 for t in range(T)]
        time_1 = time.time()
        if meth == "met2" or meth == "met3" or meth == "met4":
            # states_simp = [Node() for i in range(nsom)]
            points_met = []
            for j in range(nsom):
                x = np.array(dec2bin(j, n))
                y = np.array(s_min) + \
                    np.multiply((np.array(s_max) - np.array(s_min)), np.array(x))
                # states_simp[j].coordinates = y
                points_met.append(y)
                # print(y)

        # DP recursion
        for t in range(T, 0, -1):
            my_probs[t - 1] = Model_mat(u_disc, s_points, f_val, v_val,
                                        u_max, s_min, s_max, B, C).Model

            if meth == "met1":
                s_points = np.random.uniform(low=s_min, high=s_max,
                                             size=[n_points, n])
                val_val = []
                for i in range(n_points):
                    val, subgrad = esper_q(my_probs[t - 1], s_points[i], Q)
                    val_val.append(val)
            err_list = []
            if meth == "met2" or meth == "met3" or meth == "met4":
                val_val = []
                sub_grad = []
                s_points = deepcopy(points_met)

                for k in range(nsom):
                    val, subgrad = esper_q(my_probs[t - 1], s_points[k], Q)

                    val_val.append(val)
                    sub_grad.append(subgrad)
            if meth == "met2":
                err_list = []
                for j in range(n_points - nsom):
                    x_new = np.random.uniform(low=s_min, high=s_max, size=[1, n])[0]
                    ld = Piecewise_Approx(s_points, val_val, x_new, conv).lamda()['X']
                    # print(l)
                    err, x, l = ErrorMat(ld, s_points, val_val, sub_grad, conv).delta()
                    # print(x)
                    err_list.append(err)
                    val, subgrad = esper_q(my_probs[t - 1], x, Q)

                    s_points.append(x)
                    val_val.append(val)
                    sub_grad.append(subgrad)
            if meth == "met3":
                err_list = []
                for j in range(n_points - nsom):
                    max_err = 0
                    y_new = []
                    for k in range(m_batch):
                        x_new = np.random.uniform(low=s_min, high=s_max, size=[1, n])[0]
                        ld = Piecewise_Approx(s_points, val_val, x_new, conv).lamda()['X']
                        err, x, l = ErrorMat(ld, s_points, val_val, sub_grad, conv).delta()
                        if abs(err) >= max_err:
                            y_new = x
                            max_err = err
                    val, subgrad = esper_q(my_probs[t - 1], y_new, Q)
                    s_points.append(y_new)
                    val_val.append(val)
                    sub_grad.append(subgrad)
                    err_list.append(max_err)
            if meth == "met4":
                err_list = []
                queue = []
                for j in range(n_points - nsom):
                    pool = [] + queue
                    for k in range(m_batch):
                        x_new = np.random.uniform(low=s_min, high=s_max, size=[1, n])[0]
                        ld = Piecewise_Approx(s_points, val_val, x_new, conv).lamda()['X']
                        err, x, l = ErrorMat(ld, s_points, val_val, sub_grad, conv).delta()
                        val, subgrad = esper_q(my_probs[t - 1], x, Q)
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

                    err_list.append(pool[0].const)
                    s_points.append(pool[0].coordinates)
                    val_val.append(pool[0].val)
                    sub_grad.append(pool[0].subgrad)
                    if len(pool) > r_simp:
                        queue = pool[1:r_simp]
                    else:
                        queue = pool[1:]
            if meth == "simp1":
                s_points, val_val, err_list = gridGenDyn(my_probs[t - 1],
                                                      s_min, s_max, n_points, Q, conv, 1)
            if meth == "simp2":
                s_points, val_val, err_list = gridGenDyn(my_probs[t - 1],
                                                      s_min, s_max, n_points, Q, conv, 2)

            v_val = val_val
        time_2 = time.time()

        return my_probs, time_2 - time_1, err_list