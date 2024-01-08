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
import networkx as nx
import logging

class Callback:
    """Callback class implementing lazy constraints.  At MIPSOL
    callbacks, solutions are checked for cycles without the first point and cycles elimination
    constraints are added if needed."""

    def __init__(self, cities_coord, x_var, y_var):
        self.cities_coord = cities_coord
        self.x_var = x_var
        self.y_var = y_var

    def __call__(self, model, where):
        """Callback entry point: call lazy constraints routine when new
        solutions are found. Stop the optimization if there is an exception in
        user code."""
        if where == GRB.Callback.MIPSOL:
            try:
                # Récupérer la solution courante
                x_values = model.cbGetSolution(self.x_var)
                y_values = model.cbGetSolution(self.y_var)
                n = int(np.sqrt(len(x_values)))
                x = np.array(x_values).reshape((n, n))
                y = np.array(y_values).reshape((n, n))

                # Perform the separation of inequalities
                self.separation_inequalities_lazy( x, y, model)
            except Exception:
                logging.exception("Exception occurred in MIPSOL callback")
                model.terminate()

    def separation_inequalities_lazy(self, x, y, model):
        """Extract the current solution, adds lazy constraints in Gurobi model for the anneau étoile problem
        for all cycles without the first node.
        Assumes we are at MIPSOL.
        ------------
        Input: x: values of variables x_ij in current solution (np.array n*n)
            y: values of variables y_ij in current solution (np.array n*n)
            model: Gurobi model
        """
        n = len(x)
        G = nx.Graph()

        # Ajouter les points médians (yii = 1) comme nœuds dans le graphe
        median_points = [i for i in range(n) if y[i,i] == 1]
        G.add_nodes_from(median_points)

        # Ajouter les arêtes formées par xij = 1
        for i in range(n):
            for j in range(i+1, n):
                if x[i][j] == 1:
                    G.add_edge(i, j)

        # Trouver les cycles dans le graphe
        cycles = list(nx.connected_components(G))

        # Vérifier si tous les cycles passent par 1
        for cycle in cycles:
            if 0 not in cycle:
                # les cycles ne passant pas par 1
                S = set(cycle)
                nS = set(np.arange(n)) - S

                # Ajouter une contrainte pour exclure cet ensemble S
                model.cbLazy(quicksum(x[i,j] for i in S for j in nS) >= 2*y[i,i])
    
 
def solve_model(file_path, p, alpha):
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
            x[i,j] = m.addVar(vtype=GRB.BINARY, name=f"x_{i}{j}")
            y[i,j] = m.addVar(vtype=GRB.BINARY, name=f"y_{i}{j}")
    
    # Set objective function
    m.setObjective(quicksum(dist[i,j] * (x[i,j] + alpha*y[i,j]) for i in range(n) for j in range(n)), sense=GRB.MINIMIZE)

    # Add constraints
    m.addConstr(np.sum(y[i,i]) == p, name='c1')  # contrainte (1)
    for i in range(n):
        m.addConstr(np.sum(y[i,:]) == 1, name='c2')  # contraintes (2)
        m.addConstr(np.sum(x[i,np.arange(n)!=i]) == 2*y[i,i], name='c4')  # contraintes (4)
        #m.addConstr(np.sum(x[np.arange(n), i]) == 2*y[i,i], name='c4')
    for j in range(n):
        if j != 0:
            m.addConstr(y[0,j] == 0, name='c10') 
        for i in range(n):  
            if i != j:
                m.addConstr(y[i,j] <= y[j,j], name='c3')   # contraintes (3)
    '''
    for subset_size in range(1, n):   # contraintes (8)
        print(subset_size)
        sub = combinations(set(np.arange(n)), subset_size)
        for s in sub:
            if 0 not in s:
                ns = set(np.arange(n)) - set(s) # complémentaire de s
                sum = quicksum(x[i,j] for i in s for j in ns)
                for i in s:
                    m.addConstr(sum >= 2*y[i,i], name='c8')
    '''
    m.addConstr(y[0,0] == 1, name='c9') 

    # Optimize model using lazy constraints
    m.Params.LazyConstraints = 1
    cb = Callback(cities_coord, x.flatten().tolist(), y.flatten().tolist())
    m.optimize(cb)
   
    return m, n, cities_coord, cities


'''
def main():
    # Get file and p from command-line arguments
    # Check if correct number of command-line arguments
    if len(sys.argv) != 3:
        print("Usage: python script.py file_path p")
        sys.exit(1)

    file_path = sys.argv[1]
    p = int(sys.argv[2])
'''
script_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_directory)
file_path = "instances/ulysses16.tsp"
p = 4

#current_directory = os.getcwd()
#print(current_directory)
#file_path = os.path.join(current_directory, file_path)
# Create destination path
l = os.path.basename(file_path).split('.')
name, ext = '.'.join(l[:-1]), '.' + l[-1]
'''
destination_path = os.path.dirname(file_path) + f'/{name}_modif/'
if not os.path.exists(destination_path): # Check if the directory exists, create it if not
    os.makedirs(destination_path)
destination_path = destination_path + f"/{name}_to_{p}" + ext
'''
# Create the model
m, n, cities_coord, cities = solve_model(file_path, p, 10)


# Get the results
stations = []
stations_plot = []
tour = []
x_sol, y_sol = np.empty((n,n)), np.empty((n,n))
if m.status == GRB.OPTIMAL:
    for i in range(n):
        y_ii = m.getVarByName(f"y_{i}{i}")
        if y_ii.getAttr('X'):
            stations.append(cities_coord[i])
            stations_plot.append(i) 
            #
            for j in range(n):
                x_ij = m.getVarByName(f"x_{i}{j}")
                if x_ij.getAttr('X'):
                    tour.append([cities_coord[i], cities_coord[j]])
            #
        for j in range(n):
            x_ij = m.getVarByName(f"x_{i}{j}")
            x_sol[i,j] = x_ij.X
            y_ij = m.getVarByName(f"y_{i}{j}")
            y_sol[i,j] = y_ij.X
    print("y_sol =", y_sol)
    print("x_sol =", x_sol)
    instance =  parse_instance(file_path) #
    plotTSP([tour], instance, save=True, file_name=f"metro_circ_anneau_{name}_{p}", mode="edges")
    i = stations_plot[0]
    print(stations_plot)
    #print(stations)
    tour = [] #
    tour.append((cities_coord[i][0], cities_coord[i][1]))
    ct = 0
    stations_in_tour = []
    stations_in_tour.append(i)
    while len(tour) < p and ct:
        for j in stations_plot:
            if j not in stations_in_tour:
                x_ij = m.getVarByName(f"x_{i}{j}")
                if x_ij.x == 1:
                    tour.append((cities_coord[j][0], cities_coord[j][1]))
                    stations_in_tour.append(j)
                    i = j
                    break
        ct += 1
        
if m.Status == GRB.INFEASIBLE:
    print("The model is infeasible.")
    m.computeIIS()
    print("IIS Constraints:")
    for constr in m.getConstrs():
        if constr.IISConstr:
            print(constr.ConstrName)
            
instance =  parse_instance(file_path)
print("len_tour", len(tour))
#plotTSP([tour], instance, save=True, file_name=f"metro_circ_anneau_{name}_{p}")
evaluation = evaluate_solution(tour, instance)
print(evaluation)

'''
if __name__ == "__main__":
    main()
'''