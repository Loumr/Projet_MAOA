import numpy as np
import math
import random
from scipy.optimize import minimize


def solve_heuristic(points, number_of_stations):
    stations = choose_stations(points, number_of_stations)
    station_points = []
    for s in stations:
        station_points.append(points[s])    
    tsp = heuristic_TSP(station_points)
    return tsp

def calculate_total_distance(config, points):
    total_distance = 0
    for i in range(len(config) - 1):
        index1, index2 = config[i], config[i + 1]
        point1, point2 = points[index1], points[index2]
        distance = ((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)**0.5
        total_distance += distance
    return total_distance

def choose_stations(points, number_of_stations, nb_of_iterations = 10000):
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



"""def calculate_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def heuristic_TSP(points):
    num_points = len(points)
    visited = [False] * num_points
    tour = [points[0]]  # Starting from the first point

    for _ in range(num_points - 1):
        current_point = tour[-1]
        min_distance = float('inf')
        nearest_point = None

        for i, point in enumerate(points):
            if not visited[i]:
                distance = calculate_distance(current_point, point)
                if distance < min_distance:
                    min_distance = distance
                    nearest_point = i

        tour.append(points[nearest_point])
        visited[nearest_point] = True

    # Return to the starting point to complete the tour
    tour.append(tour[0])

    return tour"""
    

def calculate_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def total_distance(tour, points):
    distance = 0
    for i in range(len(tour) - 1):
        distance += calculate_distance(points[tour[i]], points[tour[i+1]])
    return distance

def heuristic_TSP(points, temperature=10000, cooling_rate=0.995, iterations=10000):
    # simulated annealing approach
    num_points = len(points)
    current_tour = list(range(num_points))
    random.shuffle(current_tour)
    current_distance = total_distance(current_tour, points)

    best_tour = current_tour.copy()
    best_distance = current_distance

    for it in range(iterations):
        next_tour = current_tour.copy()

        # Perform a random swap to generate a neighboring solution
        i, j = sorted(random.sample(range(num_points), 2))
        next_tour[i:j+1] = reversed(next_tour[i:j+1])

        next_distance = total_distance(next_tour, points)

        # Accept the new solution with a certain probability
        if random.random() < np.exp((current_distance - next_distance) / temperature):
            current_tour = next_tour
            current_distance = next_distance

        # Update the best solution if needed
        if current_distance < best_distance:
            best_tour = current_tour.copy()
            best_distance = current_distance

        # Cool down the temperature
        temperature *= cooling_rate
    
    best_tour_points = []
    for k in best_tour:
        best_tour_points.append(points[k])

    return best_tour_points

def local_search():
    pass


