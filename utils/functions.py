import numpy as np
from collections import deque
import math
from copy import deepcopy
from itertools import permutations
import pandas as pd
import numpy as np
from collections import OrderedDict
import time
#from models.delta_simp import DeltaSimpML
from models.deltaSimpML import deltaML


def dec2bin(d, n):
    dbin = deque()
    for i in range(n):
        dbin.append(d % 2)
        d = int(d / 2)
    return list(dbin)


def update(prob, state, Q):
    for j, el in enumerate(state):
        sto = prob.getConstrByName("sto" + "[" + str(j) + "]")
        sto.rhs = state[j] + Q[j]
        prob.update()


def esper_q(prob, state, Q):
    val = 0.0
    sub_grad = [0.0 for i, el_state in enumerate(state)]
    for q, el_Q in enumerate(Q):
        update(prob, state, Q[q])
        prob.optimize()
        val += prob.objVal / len(Q)
        sg = [prob.getConstrByName("sto" + "[" + str(i) + "]").getAttr("Pi") for i, el_State in enumerate(state)]
        sub_grad = np.add(sub_grad, sg)
    for i, ei in enumerate(state):
        sub_grad[i] = sub_grad[i] / len(Q)
    return val, sub_grad


def cobb(x, a):
    val = 1.0
    sub_grad = [a[i] * math.pow(x[i], (a[i] - 1)) for i, el in enumerate(a)]
    for i, el in enumerate(zip(x, a)):
        val *= math.pow(x[i], a[i])
        for j, ex in enumerate(zip(x, a)):
            if j != i:
                sub_grad[i] *= math.pow(x[j], a[j])
    return val, sub_grad


def eval_point(A, point, s_max, conv, cb=1):
    match conv:
        case 1:
            val = 0.5 * np.inner(s_max, s_max) + \
                  0.5 * np.inner(np.matmul(np.add(point, -s_max), A), np.add(point, -s_max))
            sub_grad = np.inner(A, np.add((point, -s_max)))
        case 0:
            if cb == 1:
                if len(s_max) == 1:
                    val, sub_grad = cobb(point, [0.5])
                else:
                    val, sub_grad = cobb(point, [1. / len(s_max) for i, el in enumerate(s_max)])

            else:
                val = 0.5 * np.inner(s_max, s_max) - \
                      0.5 * np.inner(np.matmul(np.add(point, -s_max), A), np.add(point, -s_max))
                sub_grad = -np.inner(A, np.add(point, -s_max))
    return val, sub_grad


def f_val(u, alpha, beta):
    return [beta[i] * (math.pow(u[i], 0.5) + np.random.uniform(low=125 * u[i], high=170 * u[i])) for i, el in
            enumerate(u)]


def simul_fval(my_prob,s0,Q0):
    f_simul = []
    for i in range(len(Q0)):
        val, subgrad = esper_q(my_prob,s0[i],[Q0[i]])
        f_simul.append(val)
    return np.min(f_simul), np.max(f_simul), np.mean(f_simul), np.std(f_simul)


class Node(object):
    def __init__(self, node_num='-', coordinates=np.array([]), val=0.0, sub_grad=np.array([]), const=0.0):
        self.node_num = node_num
        self.coordinates = coordinates
        self.val = val
        self.sub_grad = sub_grad
        self.const = const


class Simplex(object):
    def __init__(self, simpNum=None, nodes=None, divPoint=None, delta=None,
                 lamb=None, vol=None):
        if simpNum is None:
            simpNum = 0  # simplex number
        self.simpNum = simpNum

        if nodes is None:
            nodes = []  # list of node numbers
        self.nodes = nodes

        if divPoint is None:
            divPoint = Node()  # division point
        self.divPoint = divPoint

        if delta is None:
            delta = 0.0  # approximation error on simplex
        self.delta = delta

        if lamb is None:
            lamb = np.array([])
        self.lamb = lamb

        if vol is None:
            vol = 0.0
        self.lamb = vol

    def divSimp(self, n_node, n_simplices):
        sub_simps = []
        k = 0
        for j, el in enumerate(self.nodes):
            nodes = deepcopy(self.nodes)
            if round(self.lamb[j], 5) > 0.:
                nodes[j] = n_node
                sub_simps.append(Simplex(n_simplices + k, nodes))
                k += 1

        return sub_simps

    def div_simp_edge(self, n_node, n_simplices, edge):
        return [Simplex(n_simplices + i + 1, n_node)
                for j, el in enumerate(self.nodes)
                for i, ex in enumerate(range(2))
                if deepcopy(self.nodes)[j] == edge[i]
                ]


def kuhn(n):
    nsom = int(math.pow(2, n))
    soms = (np.zeros((nsom, n))).astype(int)
    simps = (np.zeros((math.factorial(n), n + 1))).astype(int)
    for j in range(nsom):
        soms[j, :] = np.array(dec2bin(j, n))
    perm = list(permutations(range(n)))
    for i, el in enumerate(perm):
        pos = 0
        n_points = 0
        for j in range(nsom):
            x_prev = 0
            test_som = False
            for k in range(n):
                if soms[j, perm[i][k]] >= x_prev:
                    test_som = True
                    x_prev = soms[j, perm[i][k]]
                else:
                    test_som = False
                    break
            if test_som:
                simps[i, pos] = j
                pos += 1
                n_points += 1
            if n_points == n + 1:
                break
    return simps, soms


def init_simps(sMin, sMax):
    n = len(sMin)
    nodesList = [Node(i) for i in range(int(math.pow(2, n)))]  # set first 2^n nodes
    simp, som = kuhn(n)
    simpsInit = [Simplex(i, simp[i, :]) for i, el in enumerate(simp)]
    s_points = []
    for i in range(2 ** n):
        x = np.array(sMin) + np.multiply((np.array(sMax) - np.array(sMin)),
                                         np.array(som[i, :]))
        y = np.ndarray.tolist(x)
        nodesList[i].coordinates = y
        s_points.append(y)
    return simpsInit, nodesList, s_points


def init_simp(A, sMin, sMax, conv):
    s_points = []
    v_val = []
    n = len(sMin)
    nodesList = [Node(i) for i in range(n + 1)]  # set first n + 1 node #
    nodesList[0].coordinates = sMin

    # temporary
    val, subgrad = eval_point(A, sMin, sMax, conv)
    nodesList[0].val = val
    nodesList[0].subgrad = subgrad

    idMat = np.eye(n)
    for i in range(n):
        x = np.add(np.array(sMin), np.multiply(n * (sMax[i] - sMin[i]), idMat[i]))

        nodesList[i + 1].coordinates = np.ndarray.tolist(x)
        s_points.append(nodesList[i + 1].coordinates)
        # temporary
        val, subgrad = eval_point(A, nodesList[i + 1].coordinates, sMax, conv)
        nodesList[i + 1].val = val
        nodesList[i + 1].subgrad = subgrad
        v_val.append(val)
    simpInit = Simplex(0, [i for i in range(n + 1)])

    return simpInit, nodesList, s_points, v_val


class ReadFiles:
    @staticmethod
    def read_data(file, sheet_name):
        dict_params = OrderedDict()
        if sheet_name is None:
            input("Input vector here: ")
        else:
            data = pd.read_excel(file, sheet_name=sheet_name)
        dict_params[sheet_name] = np.array(data)
        return dict_params

    # for csv
    def create_vector(self, column_name, dict_name):
        col = self.data[column_name]
        count = 0
        splitter = [[] for _ in range(len(col)) if _ % self.vec_size == 0]
        for j, value in enumerate(col):
            if j > 0 and j % self.vec_size == 0:
                count = count + 1
                splitter[count].append(value)
            else:
                splitter[count].append(value)

        self.vec_data[dict_name] = np.array(splitter)

    def readtxt(self, filename):
        with open(filename, 'r') as f:
            return [txt for txt in f.readlines]


def gridGen(sMin, sMax, A, n_points, conv=0, opt_init_simp=2):
    time_1 = time.time()
    # Calculation of approximation errors on the first n! simplices
    v_disc = []
    s_disc = []
    if opt_init_simp == 2:
        simpsInit, nodesList, s_points = init_simps(sMin, sMax)
        maxdeltaInit = 0.
        for j in range(len(nodesList)):
            val, subgrad = eval_point(A, nodesList[j].coordinates, sMax, conv)
            nodesList[j].val = val
            v_disc.append(val)
            s_disc.append(nodesList[j].coordinates)
            nodesList[j].subgrad = subgrad
        for i in range(len(simpsInit)):
            delta, x, l = deltaML(simpsInit[i], nodesList, sMin, sMax, conv)
            simpsInit[i].delta = delta
            simpsInit[i].lamb = l
            simpsInit[i].divPoint.coordinates = x
            val, subgrad = eval_point(A, x, sMax, conv)
            simpsInit[i].divPoint.val = val
            simpsInit[i].divPoint.subgrad = subgrad
            if simpsInit[i].delta > maxdeltaInit:
                maxdeltaInit = simpsInit[i].delta
        simpList = simpsInit
        activeSimpList = simpsInit
    elif opt_init_simp == 1:
        simpInit, nodesList, s_points, v_val = init_simp(A, sMin, sMax, conv)
        delta, x, l = deltaML(simpInit, nodesList, sMin, sMax, conv)
        maxdeltaInit = delta
        simpInit.delta = delta
        simpInit.lamb = l
        simpInit.divPoint.coordinates = x
        # s_disc.append(x)
        val, subgrad = eval_point(A, x, sMax, conv)
        v_disc.append(val)
        simpInit.divPoint.val = val
        simpInit.divPoint.subgrad = subgrad
        simpList = [simpInit]
        activeSimpList = [simpInit]

    # simplicial decomposition
    deltaList = []
    relativeDelta = float('inf')
    iterat = 0
    while iterat <= n_points:  # relativeDelta >= tolDel:
        iterat += 1
        # looking up the non-divided simplex with max error
        maxdelta = 0.
        for k in range(len(activeSimpList)):
            divSom = False
            l = activeSimpList[k].lamb
            for j in range(len(l)):
                if round(abs(l[j] - 1), 2) == 0:
                    divSom = True
                    break
            if activeSimpList[k].delta > maxdelta and divSom == False:
                maxdelta = activeSimpList[k].delta
                # print(maxdelta)
                indexSimp = k

        relativeDelta = (activeSimpList[indexSimp].delta) / maxdeltaInit
        deltaList.append(activeSimpList[indexSimp].delta)
        # division of the simplex with max error
        numNode = len(nodesList)
        activeSimpList[indexSimp].divPoint.nodeNum = numNode
        nodesList.append(activeSimpList[indexSimp].divPoint)
        subSimps = activeSimpList[indexSimp].divSimp(numNode, len(activeSimpList))
        v_disc.append(activeSimpList[indexSimp].divPoint.val)
        s_disc.append(activeSimpList[indexSimp].divPoint.coordinates)
        # calculate error, division point, etc. for each subsimplex
        for i in range(len(subSimps)):
            delta, x, l = deltaML(subSimps[i], nodesList, sMin, sMax, conv)
            subSimps[i].delta = delta
            subSimps[i].lamb = l
            subSimps[i].divPoint.coordinates = x
            val, subgrad = eval_point(A, x, sMax, conv)
            subSimps[i].divPoint.val = val
            subSimps[i].divPoint.subgrad = subgrad
        activeSimpList.pop(indexSimp)  # remove divided simp to active list
        activeSimpList = activeSimpList + subSimps  # add subsimps to AL
        simpList = simpList + subSimps  # add subsimps to simp list
    time_2 = time.time()
    return s_disc, v_disc, deltaList, time_2 - time_1  # [len(simpList), len(nodesList), len(deltaList), (time_2-time_1)]


def gridGenDyn(prob, sMin, sMax, n_points, Q, conv, opt_init_simp=1): # gridGen Dyn
    time_1 = time.time()

    if opt_init_simp == 2:
        v_val = []
        simpsInit, nodesList, s_points = init_simps(sMin, sMax)
        maxdeltaInit = 0.
        for j in range(len(nodesList)):
            val, subgrad = esper_q(prob, nodesList[j].coordinates, Q)
            nodesList[j].val = val
            v_val.append(val)
            nodesList[j].subgrad = subgrad
        for i in range(len(simpsInit)):
            delta, x, l = deltaML(simpsInit[i], nodesList, sMin, sMax, conv)
            simpsInit[i].delta = delta
            simpsInit[i].lamb = l
            simpsInit[i].divPoint.coordinates = x
            val, subgrad = esper_q(prob, x, Q)
            simpsInit[i].divPoint.val = val
            simpsInit[i].divPoint.subgrad = subgrad
            if simpsInit[i].delta > maxdeltaInit:
                maxdeltaInit = simpsInit[i].delta
        simpList = simpsInit
        activeSimpList = simpsInit

    elif opt_init_simp == 1:
        simpInit, nodesList, s_points, v_val = init_simp(prob, sMin, sMax, Q)
        delta, x, l = deltaML(simpInit, nodesList, sMin, sMax, conv)
        maxdeltaInit = delta
        simpInit.delta = delta
        simpInit.lamb = l
        simpInit.divPoint.coordinates = x
        val, subgrad = esper_q(prob, x, Q)
        simpInit.divPoint.val = val
        simpInit.divPoint.subgrad = subgrad
        simpList = [simpInit]
        activeSimpList = [simpInit]

    n_new_points = n_points - len(nodesList)

    deltaList = []
    # # relativeDelta = float('inf')
    iterat = 0
    while iterat < n_new_points and len(activeSimpList) >= 0:
        iterat += 1
        # looking up the non-divided simplex with max error
        maxdelta = 0.0

        for k in range(len(activeSimpList)):
            divSom = False
            l = activeSimpList[k].lamb
            for j in range(len(l)):
                if round(abs(l[j] - 1), 2) == 0:
                    divSom = True
                    break
            if abs(activeSimpList[k].delta) >= maxdelta and divSom == False:
                maxdelta = activeSimpList[k].delta
                indexSimp = k

        relativeDelta = activeSimpList[indexSimp].delta  # /simpInit.delta
        deltaList.append(relativeDelta)
        # division of the simplex with max error
        numNode = len(nodesList)
        activeSimpList[indexSimp].divPoint.nodeNum = numNode
        nodesList.append(activeSimpList[indexSimp].divPoint)
        s_points.append(activeSimpList[indexSimp].divPoint.coordinates)
        v_val.append(activeSimpList[indexSimp].divPoint.val)

        subSimps = activeSimpList[indexSimp].divSimp(numNode, len(simpList))

        # calculate error, division point, etc. for each subsimplex
        for i in range(len(subSimps)):
            delta, x, l = deltaML(subSimps[i], nodesList, sMin, sMax, conv)
            subSimps[i].delta = delta
            subSimps[i].lamb = l
            subSimps[i].divPoint.coordinates = x
            val, subgrad = esper_q(prob, x, Q)
            subSimps[i].divPoint.val = val
            subSimps[i].divPoint.subgrad = subgrad
        activeSimpList.pop(indexSimp)  # remove divided simp to active list
        activeSimpList = activeSimpList + subSimps  # add subsimps to AL
        simpList = simpList + subSimps  # add subsimps to simp list
    time_2 = time.time()
    return s_points, v_val, deltaList #
