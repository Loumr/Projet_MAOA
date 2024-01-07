from gurobipy import *
import numpy as np
import shutil
import sys
import os
from p_median import *
from data_analysis import *
from heuristic import *
from itertools import combinations
import matplotlib.pyplot as plt


def m_anneau(file_path, p, alpha):
    '''
    Defines model in Gurobi for the anneau étoile problem.
    -------------
    Input: p: int number of stations
           file_path: path to the tsp file for the data
    Output: Gurobi model to be optimized.
    '''
    # Get data
    n, cities_coord, cities = read_tsplib_file(file_path)

    # Create a new Gurobi model
    m = Model()

    # Create an array of the distances between points
    dist = np.linalg.norm(cities_coord[:, np.newaxis, :] - cities_coord, axis=2)

    # Create variables
    x = np.empty((n,n), dtype=Var)  
    y = np.empty((n,n), dtype=Var)
    for i in range(n):
        for j in range(n):
            x[i,j] = m.addVar(vtype=GRB.INTEGER, name=f"x_{i}{j}")
            y[i,j] = m.addVar(vtype=GRB.INTEGER, name=f"y_{i}{j}")
    
    # Set objective function
    m.setObjective(quicksum(dist[i,j] * (x[i,j] + y[i,j]) for i in range(n) for j in range(n)), sense=GRB.MINIMIZE)

    # Add constraints
    m.addConstr(np.trace(y) == p)  # contrainte (1)
    for i in range(n):
        m.addConstr(np.sum(y[i,:]) == 1)  # contraintes (2)
        m.addConstr(np.sum(x[np.arange(n)!=i,j]) == 2*y[i,i])  # contraintes (4)
    for j in range(n):
        if j != 0:
            m.addConstr(y[0,j] == 0) 
        for i in range(n):  
            if i != j:
                m.addConstr(y[i,j] <= y[j,j])   # contraintes (3)
    for subset_size in range(n):
        print(subset_size)
        sub = combinations(set(np.arange(n)), subset_size)
        for s in sub:
            if 0 not in s:
                ns = set(np.arange(n)) - set(s) # complémentaire de s
                sum = quicksum(x[i,j] for i in s for j in ns)
                for i in s:
                    m.addConstr(sum >= 2*y[i,i])
    m.addConstr(y[0,0] == 1) 
   
    return m, n, cities_coord, cities