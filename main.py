from data_analysis import *
from heuristic import *
from formulation_compacte import *
from gif_maker import *
import time

file = "rd100"
NB_OF_TESTS = 10

# remove the PNG files, only keep the gif
delete_all_temp_outputs('outputs')

print("parsing instance...")
instance = parse_instance("instances/"+file+".tsp")
print("instance parsed!")

heuristic_sol = [] 
average_execution_time = 0.0
average_time = 0.0
average_ratio = 0.0
average_cost = 0.0


for _ in range(NB_OF_TESTS):
    start_time = time.time()
    s_path = solve_heuristic(instance, int(len(instance)/3), save_img=True)
    end_time = time.time()

    s_dict = evaluate_solution(s_path, instance)
    average_time = s_dict["average_time"]
    average_ratio = float(float(s_dict["average_foot_dist"]) / float(s_dict["average_metro_dist"])) / float(NB_OF_TESTS)
    average_execution_time += (end_time - start_time) / float(NB_OF_TESTS)
    average_cost += calculate_metro_cost(s_path) / float(NB_OF_TESTS)

create_gif('outputs', 'heuristic_iter', file+'_heuristic_gif')

print("FINAL RESULTS:")
print("execution_time : ", average_execution_time)
print("average_time : ", average_time)
print("average_ratio : ", average_ratio)
print("average_cost : ", average_cost)

