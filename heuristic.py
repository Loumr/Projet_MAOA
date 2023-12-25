import numpy as np
import random
from scipy.optimize import minimize


def solve_heuristic(points, number_of_stations):
    stations = choose_stations(points, number_of_stations)
    #print("stations = ", stations)
    tsp = traveling_salesman_path(stations, points)
    #print("\ntsp = ", tsp)

def calculate_total_distance(config, points):
    total_distance = 0
    for i in range(len(config) - 1):
        index1, index2 = config[i], config[i + 1]
        point1, point2 = points[index1], points[index2]
        distance = ((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)**0.5
        total_distance += distance
    return total_distance

def choose_stations(points, number_of_stations, nb_of_iterations = 1000):
    # heuristique randomisée
    best_config = []
    best_config_value = float('inf') 
    for _ in range(nb_of_iterations):
        config = random.sample(range(len(points)), number_of_stations)
        config_value = calculate_total_distance(config, points)
        if config_value < best_config_value:
            best_config = config
            best_config_value = config_value
    return best_config

def calculate_total_distance_order(order, points):
    total_distance = 0
    for i in range(len(order) - 1):
        index1, index2 = order[i], order[i + 1]
        point1, point2 = points[int(index1)], points[int(index2)]
        distance = ((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)**0.5
        total_distance += distance
    return total_distance

def traveling_salesman_path(station_indices, points):
    num_stations = len(station_indices)
    initial_order = np.argsort(station_indices)  # Start with the order of station indices
    bounds = [(0, num_stations - 1) for _ in range(num_stations)]  # Bounds for each index

    result = minimize(
        calculate_total_distance_order,
        initial_order,
        args=(points,),
        bounds=bounds,
        #method='TSP',
        options={'disp': True}
    )

    best_order = result.x.astype(int)
    best_path = [station_indices[i] for i in best_order]  # Convert indices back to original order
    return best_path

def local_search():
    pass


