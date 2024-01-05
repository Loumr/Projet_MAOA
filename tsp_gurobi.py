from gurobipy import *
import numpy as np
import shutil
import sys
import os
from p_median import *
from data_analysis import *


def model_tspMTZ(file_path):
    '''
    Defines model in Gurobi for the TSP problem.
    -------------
    Input: p: int number of stations
           file_path: path to the tsp file for the data
    Output: Gurobi model to be optimized.
    '''
    # Get data
    n, cities_coord, cities = read_tsplib_file(file_path)

    # Create a new Gurobi model
    m_tsp = Model()

    # Create an array of the distances between points
    dist = np.linalg.norm(cities_coord[:, np.newaxis, :] - cities_coord, axis=2)

    # Create variables
    x = np.empty((n,n), dtype=Var)  
    for i in range(n):
        for j in range(n):
            x[i,j] = m_tsp.addVar(vtype=GRB.INTEGER, name=f"x_{i}{j}")
    # MTZ variables
    u = np.empty(n, dtype=Var)
    for i in range(n):
        u[i] = m_tsp.addVar(vtype=GRB.CONTINUOUS, name=f"u_{i}")

    # Set objective function
    m_tsp.setObjective(quicksum(dist[i,j] * x[i,j] for i in range(n) for j in range(n)), sense=GRB.MINIMIZE)

    # Add constraints
    for i in range(n):
        m_tsp.addConstr(np.sum(x[i,:]) == 1) 
    for j in range(n):
        m_tsp.addConstr(np.sum(x[:,j]) == 1)
    for i in range(n):
        for j in range(n):
            m_tsp.addConstr(x[i,j] + x[j,i] <= 1)
    # MTZ constraints
    m_tsp.addConstr(u[0] == 1)
    for i in range(1,n):
        m_tsp.addConstr(u[i] <= n)
        m_tsp.addConstr(u[i] >= 2)
        for j in range(1,n):
            if j != i:
                m_tsp.addConstr(u[i] - u[j] + 1 <= n*(1 - x[i,j]))
            
    return m_tsp, n, cities_coord, cities

def main():
    # Get file and p from command-line arguments
    # Check if correct number of command-line arguments
    if len(sys.argv) != 3:
        print("Usage: python script.py file_path p")
        sys.exit(1)

    file_path = sys.argv[1]
    p = int(sys.argv[2])

    # Create destination path
    l = os.path.basename(file_path).split('.')
    name, ext = '.'.join(l[:-1]), '.' + l[-1]
    destination_path = os.path.dirname(file_path) + f'/{name}_modif/'
    if not os.path.exists(destination_path): # Check if the directory exists, create it if not
        os.makedirs(destination_path)
    destination_path = destination_path + f"/{name}_to_{p}" + ext

    # Create the model
    model_pmedian, n, cities, cities_coord = model_pmed(p, file_path)

    # Optimize the model
    model_pmedian.optimize()

    # Get the results
    stations = []
    stations_plot = []
    if model_pmedian.status == GRB.OPTIMAL:
        for i in range(n):
            y_ii = model_pmedian.getVarByName(f"y_{i}{i}")
            if y_ii.getAttr('X'):
                stations.append(cities[i])
                stations_plot.append(i)

    plot_pmedian(stations_plot, cities_coord, show=True, save=False, plot_name="Solution p-median", file_name="stations")
    # Create new file with the resulting TSP instance
    copy_and_modify_file(file_path, destination_path, file_modification, stations)
    #print("stations : \n", stations) 

    # Apply TSP on chosen stations
    m_tsp, n2, cities_coord2, cities2 = model_tspMTZ(destination_path)

    # Optimize the model
    m_tsp.optimize()

    # Get the results
    tour = {}  
    if m_tsp.status == GRB.OPTIMAL:
        for i in range(n2):
            u_i = m_tsp.getVarByName(f"u_{i}")
            tour[i] = (cities_coord2[i][0], cities_coord2[i][1]), u_i.x
    print("tour", tour)
    tour = [tour[k][0] for k in sorted(tour, key=lambda x: tour[x][1])]
    instance =  parse_instance(file_path)
    plotTSP([tour], instance, save=True)

if __name__ == "__main__":
    main()