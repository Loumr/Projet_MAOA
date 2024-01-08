from data_analysis import *
from heuristic import *
from formulation_compacte import *
from gif_maker import *

file = "rd100"

# remove the PNG files, only keep the gif
delete_all_temp_outputs('outputs')

print("parsing instance...")
instance = parse_instance("instances/"+file+".tsp")
print("instance parsed!")

heuristic_sol = [] 
for i in range(1):
    heuristic_sol.append(solve_heuristic(instance, int(len(instance)/3), save_img=True))

print("solution:", evaluate_solution(heuristic_sol[0], instance))

plotTSP(heuristic_sol, instance)

create_gif('outputs', 'heuristic_iter', file+'_heuristic_gif')


