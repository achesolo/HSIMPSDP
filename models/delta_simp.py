import numpy as np
from gurobipy import *
import pandas as pd


class DeltaSimpML:
    def __init__(self, simp, nodes_list, s_min, s_max, conv):
        self.__model = Model()
        self.__model.setParam('OutputFlag', 0)
        self.__obj = LinExpr()
        self.__constL = LinExpr()
        self.__c0 = LinExpr()
        self.__c1 = LinExpr()
        self.__simp = simp
        self.__node_list = nodes_list
        self.__s_min = s_min
        self.__s_max = s_max
        self.__conv = conv
        x, ld, mu = self.__variables(self.__model, self.__s_min, self.__s_max)
        self.__objective(self.__model, self.__obj, self.__simp, self.__node_list, ld, mu, self.__s_min, self.__conv)
        self.__constraints(self.__model, self.__constL, self.__c0, self.__c1, self.__simp,
                           self.__node_list, ld, x, mu, self.__s_min, self.__conv)
        self.__data = self.__optimize(self.__model, self.__s_min)

    @staticmethod
    def __variables(model, s_min, s_max):
        mu = [model.addVar(name='mu')]
        ld = [model.addVar(lb=0, ub=1, name='l' + str(i)) for i in range(len(s_min) + 1)]
        x = [model.addVar(lb=s_min[i], ub=s_max[i], name='x' + str(i)) for i in range(len(s_min))]
        return x, ld, mu

    @staticmethod
    def __objective(model, obj, simp, nodes_list, ld, mu, s_min, conv):
        if conv == 0:
            obj += mu[0]
        elif conv == 1:
            obj -= mu[0]
        for i in range(len(s_min) + 1):
            if conv == 0:
                obj -= nodes_list[simp.nodes[i]].val * ld[i]
            elif conv == 1:
                obj += [simp.nodes[i]].val * ld[i]
        model.setObjective(obj, GRB.MAXIMIZE)
        model.update()

    @staticmethod
    def __constraints(model, constL, c0, c1, simp, nodes_list, ld, x, mu, s_min, conv):
        for i in range(len(s_min) + 1):
            constL += ld[i]
            for j in range(len(s_min)):
                c0 += nodes_list[simp.nodes[i]].subgrad[j] * x[j]
            if conv == 0:
                model.addConstr(mu[0] - c0 <= nodes_list[simp.nodes[i]].val - \
                                np.inner(nodes_list[simp.nodes[i]].coordinates,
                                         nodes_list[simp.nodes[i]].subgrad))
            elif conv == 1:
                model.addConstr(mu[0] - c0 >= nodes_list[simp.nodes[i]].val - \
                                np.inner(nodes_list[simp.nodes[i]].coordinates,
                                         nodes_list[simp.nodes[i]].subgrad))

        model.addConstr(constL == 1)
        for i in range(len(s_min)):
            for j in range(len(s_min) + 1):
                c1 += nodes_list[simp.nodes[j]].coordinates[i] * ld[j]
            model.addConstr(x[i] - c1 == 0)
        # constL += quicksum(ld[i] for i in range(len(s_min) + 1))
        # c0 = quicksum((nodes_list[simp.nodes[i]].subgrad[j] * x[j]
        #                         for i in range(len(s_min) + 1)
        #                         for j in range(len(s_min))))
        # c1 = quicksum((nodes_list[simp.nodes[j]].coordinates[i] * ld[j]
        #                         for i in range(len(s_min))
        #                         for j in range(len(s_min) + 1)))
        #
        # model.addConstr(
        #     mu[0] - c0 <= quicksum(nodes_list[simp.nodes[i]].val -
        #                            np.inner(nodes_list[simp.nodes[i]].coordinates,
        #                                     nodes_list[simp.nodes[i]].subgrad)
        #                            for i in range(len(s_min) + 1) if conv == 0)
        # )
        # model.addConstr(
        #     mu[0] - c0 >= quicksum(nodes_list[simp.nodes[i]].val -
        #                            np.inner(nodes_list[simp.nodes[i]].coordinates,
        #                                     nodes_list[simp.nodes[i]].subgrad)
        #                            for i in range(len(s_min) + 1) if conv == 1)
        # )
        # model.addConstr(constL == 1)
        # model.addConstr(quicksum(x[i] for i in range(len(s_min))) - c1 == 0)

    @staticmethod
    def __optimize(model, s_min):
        model.optimize()
        x = [v.x for v in model.getVars() for i in range(len(s_min) + 1) if v.varName == 'x' + str(i)]
        ld = [v.x for v in model.getVars() for i in range(len(s_min) + 1) if v.varName == 'l' + str(i)]

        tupleD = (model.ObjVal, x, ld)
        d = {"ObjVal": [model.ObjVal],
             "x": x,
             "l": ld
             }
        data = pd.DataFrame.from_dict(d, orient='index', columns=None)
        data = data.T
        data.to_excel("error.xlsx", sheet_name="error", index=False)
        return tupleD  # d

    def deltaML(self):
        return self.__data
