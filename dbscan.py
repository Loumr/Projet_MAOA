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

def dbscan(cities_coord, eps=110, min_samples=3):
    # Applying DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(cities_coord)
    num_clusters = len(np.unique(labels))

    '''
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
    '''
    label_dict = {}
    
    for i, label in enumerate(labels):
        if label not in label_dict:
            label_dict[label] = [i]
        else:
            label_dict[label].append(i)
    return label_dict, num_clusters