from gurobipy import *
import numpy as np
import shutil
import sys
import os
from itertools import combinations
import matplotlib.pyplot as plt
from data_analysis import *
from p_median import *
from tsp_gurobi import model_tspMTZ
from heuristic import *
from dbscan import *

def model_kcl(k, n, cities_coord, cities):
    '''
    Defines model in Gurobi for the k-clustering problem. Here, we choose clusters by solving the p-median problem
    and attributing other points to the cluster in which the nearest p-median is in.
    -------------
    Input: k: int number of clusters
           n : int (number of cities)
           cities_coord : np.array with each city coordinates
           cities : dict with keys: index of node in cities_coord, value: value of node
    Output: Gurobi model to be optimized.
    '''
    # Create a new Gurobi model
    model_kcluster = Model()

    # Create an array of the distances between points
    dist = np.linalg.norm(cities_coord[:, np.newaxis, :] - cities_coord, axis=2)

    # Create variables
    y = np.empty((n,n), dtype=Var)  # initialize array of integer variables
    for i in range(n):
        for j in range(n):
            y[i,j] = model_kcluster.addVar(vtype=GRB.INTEGER, name=f"y_{i}{j}")

    # Set objective function
    model_kcluster.setObjective(quicksum(dist[i,j] * y[i,j] for i in range(n) for j in range(n)), sense=GRB.MINIMIZE)

    # Add constraints
    model_kcluster.addConstr(np.trace(y) == k, "c0")  # contrainte (1)
    for i in range(n):
        model_kcluster.addConstr(np.sum(y[i,:]) == 1, f"c_{i+1}")  # contraintes (2)
        for j in range(n):  #  contraintes (3)
            model_kcluster.addConstr(y[i,j] <= y[j,j], f"c_{n*(i+1)+j+1}")
            model_kcluster.addConstr(y[i,j] >= 0, f"c_{n*n + n+1 +j}")
    return model_kcluster

def heur_med(k, n, cities_coord):
    dist = np.linalg.norm(cities_coord[:, np.newaxis, :] - cities_coord, axis=2)
    medians = np.zeros(n)
    dist_med = np.ones(n)*np.inf
    mean_dist = np.mean(dist, axis=1)
    idx = np.argmax(mean_dist)
    medians[idx] = 1
    for i in range(n):
            dist_med[i] = np.minimum(dist_med[i], dist[idx, i])

    for i in range(k-1):
        idx = np.argmax(dist_med)
        medians[idx] = 1
        for i in range(n):
            dist_med[i] = np.minimum(dist_med[i], dist[idx, i])
    return medians

def model_med(k, n, cities_coord, cities):
    '''
    Defines model in Gurobi for the k-clustering problem. Here, we choose clusters by choosing clusters' "medians" which are as 
     far from each other as possible and attributing other points to the cluster in which the nearest median is in.
    -------------
    Input: k: int number of clusters
           n : int (number of cities)
           cities_coord : np.array with each city coordinates
           cities : dict with keys: index of node in cities_coord, value: value of node
    Output: Gurobi model to be optimized.
    '''
    # Create a new Gurobi model
    model_med = Model()

    # Create an array of the distances between points
    dist = np.linalg.norm(cities_coord[:, np.newaxis, :] - cities_coord, axis=2)
    

    # Create variables
    x = np.empty(n, dtype=Var)
    for i in range(n):
        x[i] = model_med.addVar(vtype=GRB.BINARY, name=f"x_{i}")

    # Set objective function
    #model_kcluster.setObjective(quicksum(dist[i,j] * y[i,j] for i in range(n) for j in range(n)), sense=GRB.MINIMIZE)
    model_med.setObjective(quicksum(dist[i,j] * x[i] * x[j] for i in range(n) for j in range(n)), sense=GRB.MAXIMIZE)

    # Add constraints
    model_med.addConstr(quicksum(x[i] for i in range(n)) == k)

    return model_med

def model_kcl2(medians, n, cities_coord, cities):
    '''
    Defines model in Gurobi for the k-clustering problem. Here, we choose clusters according to clusters' "medians" which are as 
     far from each other as possible and attributing other points to the cluster in which the nearest median is in.
    -------------
    Input: k: int number of clusters
           n : int (number of cities)
           cities_coord : np.array with each city coordinates
           cities : dict with keys: index of node in cities_coord, value: value of node
    Output: Gurobi model to be optimized.
    '''
    # Create a new Gurobi model
    m = Model()
    # Create an array of the distances between points
    dist = np.linalg.norm(cities_coord[:, np.newaxis, :] - cities_coord, axis=2)
    y = np.empty((n,n), dtype=Var) 
    for i in range(n):
        for j in range(n):
            y[i,j] = m.addVar(vtype=GRB.INTEGER, name=f"y_{i}{j}")
    m.setObjective(quicksum(dist[i,j] * y[i,j] for i in range(n) for j in range(n)), sense=GRB.MINIMIZE)
    for i in range(n):
        m.addConstr(y[i,i] == medians[i]) 
        m.addConstr(np.sum(y[i,:]) == 1)  # contraintes (2)
        for j in range(n):  #  contraintes (3)
            m.addConstr(y[i,j] <= y[j,j])
            m.addConstr(y[i,j] >= 0)
    return m


def model_TSPG(clusters, n, cities_coord, cities):
    '''
    Defines model in Gurobi for the TSPG problem.
    -------------
    Input: clusters: dict: key: index of cluster ; value: list of points in that cluster (indices of points in cities_coord array)
           n : int (number of cities)
           cities_coord : np.array with each city coordinates
           cities : dict with keys: index of node in cities_coord, value: value of node
    Output: Gurobi model to be optimized.
    '''
    # Create a new Gurobi model
    m_tspg = Model()

    # Create an array of the distances between points
    dist = np.linalg.norm(cities_coord[:, np.newaxis, :] - cities_coord, axis=2)

    # Create variables
    # x[i,j]=1 if edge(i,j) chosen, else x[i,j]=0
    x = np.empty((n,n), dtype=Var)  # initialize array of binary variables
    # y[i]=1 if node i visited, else y[i]=0
    y = np.empty(n, dtype=Var)
    for i in range(n):
        y[i] = m_tspg.addVar(vtype=GRB.BINARY, name=f"y_{i}")
        for j in range(n):
            x[i,j] = m_tspg.addVar(vtype=GRB.BINARY, name=f"x_{i}{j}")

    # Set objective function
    m_tspg.setObjective(quicksum(dist[i,j] * x[i,j] for i in range(n) for j in range(n)), sense=GRB.MINIMIZE)

    # Add constraints
    for i in range(n):
        m_tspg.addConstr(np.sum(x[i,:]) - x[i,i] == 2*y[i], f"c_{i}")  # contraintes (1.2)
    for h in clusters.keys():
        m_tspg.addConstr(quicksum(y[i] for i in clusters[h]) == 1, f"c_{n+h}")  # contraintes (1.3)
    # Generate all subsets of points of size >=2 and <=n-2
    k = len(clusters)
    n_c = n+k
    for subset_size in range(2,n-2):
        print(subset_size)
        sub = combinations(set(np.arange(n)), subset_size)
        for s in sub:
            ns = set(np.arange(n)) - set(s) # complémentaire de s
            sum = quicksum(x[i,j] for i in s for j in ns)
            for i in s:
                for j in ns:
                    m_tspg.addConstr(sum >= 2*(y[i] + y[j] - 1), f"c_{n_c}")
                    n_c += 1
    return m_tspg

def plot_clusters(clusters, cities_coord, show=True, save=False, plot_name="Solution clusters", file_name="clusters"):
    """
    clusters: List of lists of coordinates of the stations 
    cities_coord: numpy array of coordinates for the different nodes
    """
    min_x, min_y = np.min(cities_coord[:,0]), np.min(cities_coord[:,1])
    max_x, max_y = np.max(cities_coord[:,0]), np.max(cities_coord[:,1])
    for l, cluster in clusters.items():
        x, y = [], []
        for pt in cluster:
            x.append(cities_coord[pt,0])
            y.append(cities_coord[pt,1])
        plt.scatter(x, y, marker='o',label=f'Cluster {l}')

    s_x = 1
    s_y = 1
    plt.xlim(min_x - s_x, max_x + s_x)
    plt.ylim(min_y - s_y, max_y + s_y)
    plt.title(plot_name)
    if save:
        plt.tight_layout()
        plt.savefig('outputs/'+file_name+'.png', bbox_inches='tight')
    if show:
        plt.show()
    plt.close()

def main():
    # Get file and p from command-line arguments
    # Check if correct number of command-line arguments
    if len(sys.argv) != 4:
        print("Usage: python script.py file_path cluster_method k")
        sys.exit(1)

    file_path = sys.argv[1]
    p = int(sys.argv[3])
    cluster_method = sys.argv[2]

    # Create destination path
    l = os.path.basename(file_path).split('.')
    name, ext = '.'.join(l[:-1]), '.' + l[-1]
    destination_path = os.path.dirname(file_path) + f'/{name}_modif/'
    if not os.path.exists(destination_path): # Check if the directory exists, create it if not
        os.makedirs(destination_path)
    destination_path = destination_path + f"/{name}_to_{p}" + ext

    # Get data
    n, cities_coord, cities = read_tsplib_file(file_path)

    # Create the model for k_clustering
    if cluster_method == 'p_median' or cluster_method == 'eloignes':
        if cluster_method == 'p_median':
            m = model_med(p, n, cities_coord, cities)
            # Optimize the model
            m.optimize()
            medians = np.empty(n)
            med = []
            if m.status == GRB.OPTIMAL:
                for i in range(n):
                    x_i = m.getVarByName(f"x_{i}")
                    medians[i] = x_i.x
                    if x_i.x == 1:
                        med.append(i)

        #medians = heur_med(p, n, cities_coord)
        #plot_pmedian(medians, cities_coord, save=True, file_name=f'med_{name}_{p}')
        elif cluster_method == 'eloignes':
            medians = heur_med(p, n, cities_coord)
        model_kcluster = model_kcl2(medians, n, cities_coord, cities)
        model_kcluster.optimize()

        # Get the results
        clusters = {}
        if model_kcluster.status == GRB.OPTIMAL:
            l = -1  # initialize index of cluster
            for j in range(n):
                y_jj = model_kcluster.getVarByName(f"y_{j}{j}")
                if y_jj.getAttr('X'):  # if yjj != 0
                    l += 1
                    clusters[l] = []
                    for i in range(n):
                        y_ij = model_kcluster.getVarByName(f"y_{i}{j}") 
                        if y_ij.getAttr('X'):  # if yij != 0
                            clusters[l].append(i)    # each cluster is a list of indices of points in the coord_cities array
    #plot_clusters(clusters, cities_coord, save=True, file_name=f'clusters_{name}_{p}')  
    elif cluster_method == 'dbscan':
        clusters, p = dbscan(cities_coord)
        print("clusters", p)

    m_med, n, cities, cities_coord = model_pmed(p, file_path, clusters=clusters)

    # Optimize the model
    m_med.optimize()

    # Get the results
    stations = []
    stations_plot = []
    if m_med.status == GRB.OPTIMAL:
        for i in range(n):
            y_ii = m_med.getVarByName(f"y_{i}{i}")
            if y_ii.getAttr('X'):
                stations.append(cities[i])
                stations_plot.append(i)

    #plot_pmedian(stations_plot, cities_coord, show=True, save=False, plot_name="Solution p-median", file_name="stations")
    # Create new file with the resulting TSP instance
    copy_and_modify_file(file_path, destination_path, file_modification, stations)

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
    tour = [tour[k][0] for k in sorted(tour, key=lambda x: tour[x][1])]
    instance =  parse_instance(file_path)
    plotTSP([tour], instance, save=True, file_name=f"metro_circ_gurobi_{name}_{p}", clusters=clusters)
    solution_val = calculate_solution_value(tour, instance)
    print(solution_val)
    
    
    '''
    # Create the model for generalized TSP
    m_tspg = model_TSPG(clusters, n, cities_coord, cities)
   
    # Optimize the model
    m_tspg.optimize()

    # Get the results
    stations = []
    tour = []
    if m_tspg.status == GRB.OPTIMAL:
        for i in range(n):
            y_i = m_tspg.getVarByName(f"y_{i}")
            if y_i.getAttr('X'):  # if yii != 0
                stations.append(i)
                for j in range(n):
                    x_ij = m_tspg.getVarByName(f"x_{i}{j}")
                    if x_ij.x == 1:
                        tour.append([cities_coord[i], cities_coord[j]])
    print("stations : \n", stations) 
    instance =  parse_instance(file_path)
    plotTSP([tour], instance, show=True, save=False, plot_name="Solution Métro Circulaire", file_name=f"tspg_{name}_{p}", mode="edges")
    '''
    
if __name__ == "__main__":
    main()