from gurobipy import *
import numpy as np
import shutil
import sys
import os
import matplotlib.pyplot as plt

# Get data from TSP file
def read_tsplib_file(file_path):
    '''
    Reads a tsp file.
    Returns : num_cities : int (number of cities)
              cities_coord : np.array with each city coordinates
              cities : dict with keys: index of node in cities_coord, value: value of node (string)
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
        if line.strip() =="EOF":
            break  # End of coordinates
        city_info = line.split()
        #number, x, y = map(int, city_info[0:3])
        number = str(city_info[0])
        x, y = map(float, city_info[1:3])
        cities[i] = number
        cities_coord[i,:] = np.array([x,y])
        i += 1 

    return num_cities, cities_coord, cities


def model_pmed(p, file_path, clusters=None):
    '''
    Defines model in Gurobi for the p-median problem.
    -------------
    Input: p: int number of stations
           file_path: path to the tsp file for the data
    Output: Gurobi model to be optimized.
    '''
    # Get data
    n, cities_coord, cities = read_tsplib_file(file_path)

    # Create a new Gurobi model
    model_pmedian = Model()

    # Create an array of the distances between points
    dist = np.linalg.norm(cities_coord[:, np.newaxis, :] - cities_coord, axis=2)

    # Create variables
    y = np.empty((n,n), dtype=Var)  # initialize array of binary variables
    for i in range(n):
        for j in range(n):
            y[i,j] = model_pmedian.addVar(vtype=GRB.INTEGER, name=f"y_{i}{j}")

    # Set objective function
    model_pmedian.setObjective(quicksum(dist[i,j] * y[i,j] for i in range(n) for j in range(n)), sense=GRB.MINIMIZE)

    # Add constraints
    model_pmedian.addConstr(np.trace(y) == p)  # contrainte (1)
    for i in range(n):
        model_pmedian.addConstr(np.sum(y[i,:]) == 1)  # contraintes (2)
        for j in range(n):  #  contraintes (3)
            model_pmedian.addConstr(y[i,j] <= y[j,j])
            model_pmedian.addConstr(y[i,j] >= 0)
    if clusters != None:
        for h in clusters.keys():
            model_pmedian.addConstr(quicksum(y[i,i] for i in clusters[h]) == 1)
    return model_pmedian, n, cities, cities_coord


def file_modification(content, stations):
    """
    Modification function: Only keep the cities that are stations in TSPLIB file.

    Parameters:
    - content (str): Original content of the file
    - stations (set) : Set of number of node in original TSPLIB file for all nodes chosen as a station

    Returns:
    - str: Modified content
    """
    n = len(stations)
    new_content = []
    for i, line in enumerate(content):
        new_content.append(line)
        if line.strip().startswith("NAME"):
            new_content[i] = ''.join(list(new_content[i])[:-1] + [f"_to_{n}\n"])
        if line.strip().startswith("DIMENSION"):
            new_content[i] = f"DIMENSION : {n}\n"
        if line.strip().startswith("NODE_COORD_SECTION"):
            break

    for number in stations:
        for i, line in enumerate(content):
            if line.strip().startswith(str(number)):
                new_content.append(content[i])
                break
            
    return ''.join(new_content)
    

def copy_and_modify_file(source_path, destination_path, modification_function, stations):
    """
    Copy a file from the source path to the destination path and apply a modification function.

    Parameters:
    - source_path (str): Path to the source file.
    - destination_path (str): Path to the destination file.
    - modification_function (function): A function that takes a file content as input and returns the modified content.

    Returns:
    - None
    """
    # Copy the file
    shutil.copy2(source_path, destination_path)

    # Read the content of the copied file
    with open(destination_path, 'r') as file:
        content = file.readlines()

    # Apply the modification function
    modified_content = modification_function(content, stations)

    # Write the modified content back to the file
    with open(destination_path, 'w') as file:
        file.write(modified_content)



def plot_pmedian(stations, cities_coord, show=True, save=False, plot_name="Solution p-median", file_name="stations"):
    """
    stations: List of indices of the stations 
    cities_coord: numpy array of coordinates for the different nodes
    """
    x = cities_coord[:,0].tolist()
    y = cities_coord[:,1].tolist()
    
    plt.scatter(x, y, color='blue', marker='o')

    s_x, s_y = [], []
    for s in stations:
        s = int(s)
        s_x.append(cities_coord[s,0])
        s_y.append(cities_coord[s,1])
    plt.scatter(s_x, s_y, color='red', marker='o')

    s_x = 1
    s_y = 1
    plt.xlim(min(x) - s_x, max(x) + s_x)
    plt.ylim(min(y) - s_y, max(y) + s_y)
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
            print(f"y_{i}", y_ii.x)
            if y_ii.getAttr('X'):
                stations.append(cities[i])
                stations_plot.append(i)

    # Create new file with the resulting TSP instance
    copy_and_modify_file(file_path, destination_path, file_modification, stations)
    print("stations : \n", stations) 
    plot_pmedian(stations_plot, cities_coord, show=True, save=False, plot_name="Solution p-median", file_name="stations")


if __name__ == "__main__":
    main()