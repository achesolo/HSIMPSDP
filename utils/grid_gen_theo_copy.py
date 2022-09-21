# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 14:34:29 2018

im@author: lzephyr
"""
from  init_simps import initSimps
from delta_simp import deltaML
from  initSimp import initSimp
from eval_state import evalPoint
import time

#from write_div_simp import writeDivSimp

def gridGen(sMin, sMax,A,n_points,conv=0,opt_init_simp=2):
    """    
    Parameters
    ----------
    sMin : TYPE
        DESCRIPTION.
    sMax : TYPE
        DESCRIPTION.
    A : semi-definite matrix
        DESCRIPTION.
    n_points : TYPE
        DESCRIPTION.
    conv : 0: concave functions
           1: convex functions
        DESCRIPTION.
    opt_init_simp : 
        1: 1 initial simplex
        2: n! initial simplices
        DESCRIPTION.

    Returns
    -------
    nodes and list of error bounds

    """    
    
    time_1 = time.time()
    # Calculation of approximation errors on the first n! simplices
    v_disc = []
    s_disc = []
    if opt_init_simp == 2:
        simpsInit, nodesList = initSimps(sMin, sMax)
        maxdeltaInit = 0.
        for j in range(len(nodesList)):
            val, subgrad = evalPoint(A,nodesList[j].coordinates,sMax,conv)
            nodesList[j].val = val
            v_disc.append(val)
            s_disc.append(nodesList[j].coordinates)
            nodesList[j].subgrad = subgrad
        for i in range(len(simpsInit)):            
            delta, x, l = deltaML(simpsInit[i], nodesList, sMin, sMax,conv) 
            simpsInit[i].delta = delta
            simpsInit[i].lamb = l
            simpsInit[i].divPoint.coordinates = x 
            val, subgrad = evalPoint(A,x,sMax,conv)
            simpsInit[i].divPoint.val = val            
            simpsInit[i].divPoint.subgrad = subgrad
            if simpsInit[i].delta > maxdeltaInit:
                maxdeltaInit = simpsInit[i].delta                 
        simpList = simpsInit
        activeSimpList = simpsInit
    elif opt_init_simp==1:
        simpInit, nodesList = initSimp(A,sMin,sMax,conv)
        delta, x, l = deltaML(simpInit,nodesList,sMin,sMax, conv) 
        maxdeltaInit=delta
        simpInit.delta = delta
        simpInit.lamb = l
        simpInit.divPoint.coordinates = x 
        #s_disc.append(x)
        val, subgrad = evalPoint(A,x,sMax,conv)  
        v_disc.append(val)
        simpInit.divPoint.val = val
        simpInit.divPoint.subgrad = subgrad    
        simpList = [simpInit]
        activeSimpList = [simpInit]       
        
    ##########################################################################
    ##########################################################################
    # simplicial decomposition
    deltaList = []
    relativeDelta = float('inf')    
    iterat = 0
    while iterat <=n_points: #relativeDelta >= tolDel:
        iterat += 1
        # looking up the non-divided simplex with max error
        maxdelta = 0.
        for k in range(len(activeSimpList)):
            divSom = False
            l = activeSimpList[k].lamb
            for j in range(len(l)):
                if round(abs(l[j] - 1),2) == 0:
                    divSom = True
                    break
            if activeSimpList[k].delta > maxdelta and divSom == False:
                maxdelta = activeSimpList[k].delta
                #print(maxdelta)
                indexSimp = k
            
        relativeDelta = (activeSimpList[indexSimp].delta)/maxdeltaInit
        deltaList.append(activeSimpList[indexSimp].delta)
        # division of the simplex with max error    
        numNode = len(nodesList)
        activeSimpList[indexSimp].divPoint.nodeNum = numNode
        nodesList.append(activeSimpList[indexSimp].divPoint)
        subSimps = activeSimpList[indexSimp].divSimp(numNode,len(activeSimpList))
        v_disc.append(activeSimpList[indexSimp].divPoint.val)     
        s_disc.append(activeSimpList[indexSimp].divPoint.coordinates)
        # calculate error, division point, etc. for each subsimplex
        for i in range(len(subSimps)):
            delta, x, l = deltaML(subSimps[i],nodesList,sMin,sMax,conv)
            subSimps[i].delta = delta
            subSimps[i].lamb = l
            subSimps[i].divPoint.coordinates = x                 
            val, subgrad = evalPoint(A,x,sMax,conv)                                      
            subSimps[i].divPoint.val = val
            subSimps[i].divPoint.subgrad = subgrad                
        activeSimpList.pop(indexSimp) # remove divided simp to active list
        activeSimpList = activeSimpList + subSimps # add subsimps to AL
        simpList = simpList + subSimps # add subsimps to simp list
    time_2 = time.time()
    return s_disc,v_disc,deltaList,time_2 - time_1  #[len(simpList), len(nodesList), len(deltaList), (time_2-time_1)]