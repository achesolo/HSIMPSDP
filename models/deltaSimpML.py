
import numpy as np
from gurobipy import *


def deltaML(simp,nodesList,sMin,sMax,conv):

    n = len(sMin)
    
    # Model
    model = Model("deltaS")
    model.Params.LogToConsole = 0
    
    # Decision variables and objective
    x = []
    l = []
    mu = []
    obj = LinExpr()
    
    mu.append(model.addVar(name = 'mu'))
    if conv == 0:
        obj += mu[0]
    elif conv == 1:
        obj -= mu[0]
    
    for i in range(n):
        x.append(model.addVar(lb = sMin[i], ub = sMax[i], 
                              name ='x' + str(i)))
              
    constL = LinExpr() # for constraint on the lambdas
    for i in range(n + 1):
        l.append(model.addVar(lb = 0., ub = 1., name ='l' + str(i)))
        if conv == 0:            
            obj -= nodesList[simp.nodes[i]].val*l[i]
        elif conv == 1:
            obj += [simp.nodes[i]].val*l[i]
        constL += l[i]
    
    model.setObjective(obj, GRB.MAXIMIZE)
    
    # constraints
    for i in range(n + 1):
        c0 = LinExpr()
        for j in range(n):
            c0 += nodesList[simp.nodes[i]].subgrad[j]*x[j]
        if conv == 0:
            model.addConstr(mu[0] - c0 <= nodesList[simp.nodes[i]].val - \
                        np.inner(nodesList[simp.nodes[i]].coordinates, 
                                 nodesList[simp.nodes[i]].subgrad))
        elif conv == 1:
            model.addConstr(mu[0] - c0 >= nodesList[simp.nodes[i]].val - \
                    np.inner(nodesList[simp.nodes[i]].coordinates, 
                                nodesList[simp.nodes[i]].subgrad))
    
    model.addConstr(constL == 1)    
    for i in range(n):
        c1 = LinExpr()        
        for j in range(n + 1):
            c1 += nodesList[simp.nodes[j]].coordinates[i]*l[j]
        model.addConstr(x[i] - c1 == 0)

    model.optimize()

    x = []
    l = []
    for v in model.getVars():   
        for i in range(n + 1):
            if v.varName == 'x' + str(i):
                x.append(v.x)
            elif v.varName == 'l' + str(i):
                l.append(v.x)   
    return model.objVal, x, l
        
        