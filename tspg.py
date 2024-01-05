from gurobipy import *
import numpy as np
import shutil
import sys
import os
from itertools import combinations
import matplotlib.pyplot as plt

# Get data from TSP file
def read_tsplib_file(file_path):
    '''
    Reads a tsp file.
    Returns : num_cities : int (number of cities)
              cities_coord : np.array with each city coordinates
              cities : dict with keys: index of node in cities_coord, value: value of node
    '''
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Parse relevant information
    for line in lines:
        if line.startswith("DIMENSION"):
            num_cities = int(line.split(":")[1].strip())
        '''
        elif line.startswith("NODE_COORD_SECTION"):
            break  # Start reading coordinates
        '''
    # Parse city coordinates
    cities_coord = np.zeros((num_cities,2))
    cities = {}
    for i, line in enumerate(lines):
        if line.strip().startswith("NODE_COORD_SECTION"):
            index = i
            break
    for i, line in enumerate(lines[index + 1:]):
        #if line.startswith("EOF"):
        if line.strip() == "EOF":
            break  # End of coordinates
        city_info = line.split()
        number = int(city_info[0])
        x, y = map(float, city_info[1:3])
        cities[i] = number
        cities_coord[i,:] = np.array([x,y])
        i += 1 

    return num_cities, cities_coord, cities


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


def model_TSPG(clusters, n, cities_coord, cities):
    '''
    Defines model in Gurobi for the TSPG problem.
    -------------
    Input: clusters: dict: key: index of cluster ; value: list of points in that cluster
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
    x = np.empty((n,n), dtype=Var)  # initialize array of binary variables
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
        m_tspg.addConstr(quicksum(y[i] for i in clusters[h]) >= 1, f"c_{n+h}")  # contraintes (1.3)
    # Generate all subsets of points of size >=2 and <=n-2
    k = len(clusters)
    n_c = n+k
    for subset_size in range(2,n-2):
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
    if len(sys.argv) != 3:
        print("Usage: python script.py file_path k")
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

    # Get data
    n, cities_coord, cities = read_tsplib_file(file_path)

    # Create the model for k_clustering
    model_kcluster = model_kcl(p, n, cities_coord, cities)

    # Optimize the model
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
                       clusters[l].append(i)

    plot_clusters(clusters, cities_coord)
    '''
    # Create the model for generalized TSP
    m_tspg = model_TSPG(clusters, n, cities_coord, cities)
   
    # Optimize the model
    m_tspg.optimize()

    # Get the results
    stations = []
    if m_tspg.status == GRB.OPTIMAL:
        for i in range(n):
            y_i = m_tspg.getVarByName(f"y_{i}")
            if y_i.getAttr('X'):  # if yii != 0
                stations.append(i)

    print("stations : \n", stations) 
    '''

if __name__ == "__main__":
    main()