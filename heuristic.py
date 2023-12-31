import numpy as np
import math
import random
from scipy.optimize import minimize

def solve_heuristic(points, number_of_stations, iterations=1000, nb_children=5, nb_parents=4):
    stations = choose_stations(points, number_of_stations)
    station_points = []
    for s in stations:
        station_points.append(points[s])
    solutions = []
    solutions.append(heuristic_TSP(station_points))
    best_solution = None

    for iter in range(iterations):
        if iter%50 == 0:
            print("iteration ", iter)
        sorted_solutions = rank_solutions(solutions, points)
        if best_solution == None or solution_value(sorted_solutions[0]) < solution_value(best_solution):
            best_solution = sorted_solutions[0]
            print("best value:", solution_value(best_solution), " at iteration ", iter)
        solutions = []
        for i in range(nb_parents):
            if i >= len(sorted_solutions):
                break
            for j in range(nb_children):
                solutions.append(generate_child_solution(sorted_solutions[i], points))
    
    return best_solution

def generate_child_solution(solution_comp, points, number_of_possible_points=10):
    solution, sol_dict = solution_comp
    point_to_replace = random.choice(solution)
    
    nearby_points = [point for point in points if point not in solution]
    distances = [(point, calculate_distance(point_to_replace, point)) for point in nearby_points]
    
    sorted_distances = sorted(distances, key=lambda x: x[1])
    closest_points = [point for point, _ in sorted_distances[:number_of_possible_points]]
    new_point = random.choice(closest_points)

    child_solution = [new_point if point == point_to_replace else point for point in solution]
    return child_solution

def solution_value(item):
    solution, s_dict = item
    avg_time = s_dict["average_time"]
    foot_metro_ratio = s_dict["average_metro_dist"] / (s_dict["average_foot_dist"]*10.0)
    return (avg_time + foot_metro_ratio * 1000.0)

def rank_solutions(solutions, points):
    solution_dicts = []
    for s in solutions:
        s_dict = evaluate_solution(s, points)
        solution_dicts.append( (s, s_dict) )
    sorted_solutions = sorted(solution_dicts, key=solution_value)
    return sorted_solutions

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

def calculate_distance(point1, point2):
    p1x, p1y = point1
    p2x, p2y = point2
    return math.sqrt((p1x - p2x)**2 + (p1y - p2y)**2)

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

def find_closest_point(target_point, points):
    closest_point = None
    min_distance = float('inf')

    for i in range(len(points)):
        point = points[i]
        distance = calculate_distance(target_point, point)
        if distance < min_distance:
            min_distance = distance
            closest_point = i

    return closest_point, min_distance

def distance_between_stations(s1, s2, forward, backward):
    f = 0
    b = 0
    cs = s1
    while cs != s2:
        cs += 1
        if cs >= len(forward):
            cs = 0
        f += forward[cs]
    cs = s1
    while cs != s2:
        cs -= 1
        if cs < 0:
            cs = len(backward)-1
        b += backward[cs]
    return min(f, b)

def calculate_total_metro_dist(path):
    forward = [calculate_distance(path[len(path)-1], path[0])]
    for i in range(len(path)-1):
        forward.append(calculate_distance(path[i], path[i+1]))
    backward = [calculate_distance(path[1], path[0])]
    for i in range(1, len(path)):
        backward.append(calculate_distance(path[i-1], path[i]))
    return (forward, backward)

def evaluate_solution(solution, points):
    average_time = 0
    average_foot_dist = 0.0
    average_metro_dist = 0.0
    worst_station = 0
    best_station = 0
    worst_station_time = 0.0
    best_station_time = float('inf') 

    solution_index = []
    for sp in solution:
        solution_index.append(points.index(sp))

    closest_station = []
    for p in points:
        closest, min_dist = find_closest_point(p, solution) # with fucked up indexes
        #true_closest = solution_index[closest] # with correct indexes
        closest_station.append((closest, min_dist))
    forward, backward = calculate_total_metro_dist(solution) # with fucked up indexes still

    for p1 in range(len(points)):
        station_metro_time = 0.0
        station_foot_time = 0.0
        station_time = 0.0
        for p2 in range(len(points)):
            if p1 != p2:
                closest_p1, dist_p1 = closest_station[p1]
                closest_p2, dist_p2 = closest_station[p2]
                on_foot = dist_p1 + dist_p2
                in_metro = distance_between_stations(closest_p1, closest_p2, forward, backward)
                station_time += on_foot*10 + in_metro
                station_metro_time += in_metro
                station_foot_time += on_foot
        station_time /= float(len(points))
        average_time += station_time
        average_foot_dist += station_foot_time / float(len(points))
        average_metro_dist += station_metro_time / float(len(points))
        if p1 in solution_index:
            if (average_time > worst_station_time):
                worst_station_time = average_time
                worst_station = p1
            if (average_time < best_station_time):
                best_station_time = average_time
                best_station = p1

    average_metro_dist /= float(len(points))
    average_foot_dist /= float(len(points))
    average_time /= float(len(points))
    return {
        "average_time" : average_time,
        "worst_station" : worst_station,
        "best_station" : best_station,
        "average_foot_dist" : average_foot_dist,
        "average_metro_dist" : average_metro_dist 
    }



