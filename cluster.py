from gurobipy import *
import numpy as np
import shutil
import sys
import os
from p_median import *
from data_analysis import *
from heuristic import *
from tspg import *
from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt

def dbscan(file_path, eps):
    pass
# Get data
file_path = 'instances/rd100.tsp'
file_path = "C:/Users/loumo/OneDrive/Documents/Academique/ANDROIDE/M2/MAOA/Projet/Projet_MAOA/instances/rd100.tsp"
n, cities_coord, cities = read_tsplib_file(file_path)


# Applying DBSCAN
dbscan = DBSCAN(eps=110, min_samples=3)
labels = dbscan.fit_predict(cities_coord)
print("num_clusters", len(np.unique(labels)))

# Plotting the results
unique_labels = np.unique(labels)
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

for label, color in zip(unique_labels, colors):
    class_member_mask = (labels == label)
    xy = cities_coord[class_member_mask]
    plt.scatter(xy[:, 0], xy[:, 1], c=[color], label=f'Cluster {label}')

plt.title('DBSCAN Clustering')
#plt.legend()
plt.show()