import numpy as np
from gurobipy import *
from collections import OrderedDict


class Model_mat:

    def __init__(self, u_disc, s_disc, f_val, v_val, u_max, s_min, s_max, B, C):
        try:
            if isinstance(u_disc, np.ndarray):
                self.Model = Model()
                self.Model.setParam('OutputFlag', 0)

                dec_vars = self.__add_vars(u_disc, s_disc, u_max, s_min, s_max)
                self.__objective(dec_vars, u_disc, f_val, v_val)
                self.__constraints(dec_vars, u_disc, s_disc, B, C)
                self.Model.update()

            else:
                raise Exception("{} should be in ndarray format".format((u_disc, s_disc, f_val, B, C)))

        except:
            raise Exception("Something went wrong!")

    def __add_vars(self, u_disc, s_disc, u_max, s_min, s_max):
        dec_vars = OrderedDict()
        u = self.Model.addMVar(np.shape(u_disc)[1], ub=u_max, name='u')  # release
        y = self.Model.addMVar(np.shape(u_disc)[1], name='y')  # spillage
        s = self.Model.addMVar(np.shape(u_disc)[1], lb=s_min, ub=s_max, name='s')  # storage
        m = self.Model.addMVar(np.shape(s_disc)[0], ub=1.0, name='m')  # convex combination coefficient w.r.t storage
        l = self.Model.addMVar(np.shape(u_disc), ub=1.0, name='l')  # convex combination coefficient w.r.t release

        dec_vars['u'] = u
        dec_vars['y'] = y
        dec_vars['s'] = s
        dec_vars['m'] = m
        dec_vars['l'] = l

        return dec_vars

    def __objective(self, dec_vars, u_disc, f_val, v_val):
        self.Model.setObjective(quicksum(dec_vars['l'][:, j] @ f_val[:, j]
                                         for j in range(np.shape(u_disc)[1]))
                                + dec_vars['m'] @ np.array(v_val),
                                GRB.MAXIMIZE
                                )

    def __constraints(self, dec_vars, u_disc, s_disc, B, C):
        self.Model.addConstrs((quicksum(dec_vars['l'][:, j]) == 1 for j in range(np.shape(u_disc)[1])),
                              name='lConv')  # Convexity constraint on l
        self.Model.addConstr(dec_vars['m'].sum() == 1, name='mConv')  # Convexity constraint on m
        self.Model.addConstr(dec_vars['s'] - np.array(s_disc).T @ dec_vars['m'] == 0,
                             name='sInterpo')  # storage interpolation

        self.Model.addConstrs(
            (dec_vars['u'][j] - u_disc[i, j] * dec_vars['l'][i, j] == 0
             for i in range(np.shape(u_disc)[0])
             for j in range(np.shape(u_disc)[1])),
            name='uInterpo')  # release interpolation
        self.Model.addConstr(dec_vars['s'] + B @ dec_vars['u'] + C @ dec_vars['y'] == 0, name='sto')

#
#
