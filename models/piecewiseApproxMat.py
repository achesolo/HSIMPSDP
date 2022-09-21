import numpy as np
import gurobipy as gb
import pandas as pd


class Piecewise_Approx:
    def __init__(self, s_disc, v_disc, x, conv=0):

        model = gb.Model()
        model.Params.LogToConsole = 0
        # n_points, n = np.shape(s_disc)
        rhs = np.append([1], x)
        A_eq = np.append(np.ones([1, np.shape(s_disc)[0]]), np.transpose(s_disc), axis=0)
        l = self.__variables(model, np.shape(s_disc)[0])
        self.__objective(model, l, v_disc, conv)
        self.__constraints(model, rhs, l, A_eq)
        self.data = self.__optimize(model, l)

    @staticmethod
    def __variables(model, n_points):
        return model.addMVar(n_points, ub=1.0)

    @staticmethod
    def __objective(model, decision_var, v_disc, conv):
        model.update()
        if conv == 0:
            model.setObjective(decision_var @ np.array(v_disc), gb.GRB.MAXIMIZE)
        elif conv == 1:
            model.setObjective(decision_var @ np.array(v_disc), gb.GRB.MINIMIZE)
        else:
            raise Exception('Check Convexity Option')

    @staticmethod
    def __constraints(model, rhs, decision_var, A_eq):
        model.addConstr(A_eq @ decision_var == rhs)
        model.update()

    @staticmethod
    def __optimize(model, decision_var):
        model.optimize()
        tupleD = (model.ObjVal, [i for i, x in enumerate(decision_var.X) if x > 0])
        # d = {'ObjVal': [model.ObjVal],
        #      "X": [i for i, x in enumerate(decision_var.X) if x > 0]
        #      }
        # data = pd.DataFrame.from_dict(d, orient='index', columns=None)
        # data = data.T
        # data.to_excel("error_mat.xlsx", sheet_name="Piecewise_news", index=False)
        return tupleD

    def lamda(self):
        return self.data
