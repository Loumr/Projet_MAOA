from data_analysis import *
from heuristic import *
from formulation_compacte import *

file = "rd100"

print("parsing instance...")
instance = parse_instance("instances/"+file+".tsp")
print("instance parsed!")
#solution = parse_solution("instances/"+file+".opt.tour", instance)

heuristic_sol = [] 
for i in range(1):
    heuristic_sol.append(solve_heuristic(instance, int(len(instance)/3)))

#print("instance =", instance)
#print("heuristic_sol =", heuristic_sol)

plotTSP(heuristic_sol, instance)


