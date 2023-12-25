from data_analysis import *
from heuristic import *
from formulation_compacte import *

file = "pla33810"

instance = parse_instance("instances/"+file+".tsp")
solution = parse_solution("instances/"+file+".opt.tour")

solve_heuristic(instance, 1000)
