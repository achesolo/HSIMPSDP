import numpy as np
import gurobipy as gb
import pandas as pd


class ErrorMat:
    def __init__(self, l, s_disc, v_disc, s_grad, conv=0):
        model = gb.Model()
        model.Params.LogToConsole = 0
        simp = np.array(list(map(s_disc.__getitem__, l)))
        val = np.array(list(map(v_disc.__getitem__, l)))
        sougrad = np.array(list(map(s_grad.__getitem__, l)))

        ns = len(val)
        n = len(sougrad[0])
        phi, ld, x = self.__variables(model, n, ns)
        self.__objective(model, phi, ld, val, conv)
        self.__constraints(model, simp, phi, sougrad, val, ld, x, ns,conv)
        self.data = self.__optimize(model, x, ld)

    @staticmethod
    def __variables(model, n, ns):
        phi = model.addMVar(1, name='Phi')
        ld = model.addMVar(ns, ub=1.0, name='l')
        x = model.addMVar(n, name='x')
        return phi, ld, x

    @staticmethod
    def __objective(model, phi, ld, val, conv):

        if conv == 0:
            model.setObjective(phi @ np.ones((1, 1)) - ld @ val, gb.GRB.MAXIMIZE)
        elif conv == 1:
            model.setObjective(- phi @ np.ones((1, 1)) + ld @ val, gb.GRB.MAXIMIZE)
        else:
            raise Exception("Check Convexity option")

    @staticmethod
    def __constraints(model, simp, phi, sougrad, val, ld, x, ns, conv):
        if conv == 0:
            model.addConstrs(np.ones((1, 1)) @ phi - sougrad[i] @ x <= val[i] - \
                             np.inner(sougrad[i], simp[i]) for i in range(ns))
        else:
            model.addConstrs(np.ones((1, 1)) @ phi - sougrad[i] @ x >= val[i] - \
                             np.inner(sougrad[i], simp[i]) for i in range(ns))

        model.addConstr(x - np.transpose(simp) @ ld == 0)
        model.addConstr(ld.sum() == 1)
        model.update()

    @staticmethod
    def __optimize(model, x, ld):
        model.optimize()
        tupleD = (model.ObjVal, x.X, ld.X)
        # d = {"ObjVal": [model.ObjVal],
        #      "x": x.X,
        #      "ls": ld.X
        #      }
        # data = pd.DataFrame.from_dict(d, orient='index', columns=None)
        # data = data.T
        # data.to_excel("error.xlsx", sheet_name="error", index=False)
        return tupleD #d

    def delta(self):
        return self.data
