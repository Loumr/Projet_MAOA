import matplotlib.pyplot as plt


def evaluate_solution():
    pass 
# critère du ratio temps métro/temps à pieds
# critère de temps moyen
# => budget max fixé

def compare_solutions():
    pass
# maybe with pandas

def parse_instance(file_path):
    file = open(file_path, 'r')
    Lines = file.readlines()
    file.close()

    instance = []
    
    reading_points = False
    for line in Lines:
        if "NODE_COORD_SECTION" in line:
            reading_points = True
        elif reading_points:
            if "EOF" in line:
                break
            parts = line.split()
            if len(parts) >= 3:
                coordinates = (float(parts[1]), float(parts[2]))
                instance.append(coordinates)

    return instance

def parse_solution(file_path):
    file = open(file_path, 'r')
    Lines = file.readlines()
    file.close()
    solution = []
    
    reading_points = False
    for line in Lines:
        if "TOUR_SECTION" in line:
            reading_points = True
        elif reading_points:
            if (line == "") or ("EOF" in line):
                break
            parts = line.split()
            for p in parts:
                if int(p) == -1:
                    break
                else:
                    solution.append(int(p))

    return solution


# Source:
# https://gist.github.com/payoung/6087046
def plotTSP(paths, points):
    """
    paths: List of lists with the different orders in which the nodes are visited
    points: coordinates for the different nodes
    """

    x = []
    y = []

    for x_p, y_p in points:
        x.append(x_p)
        y.append(y_p) 
    plt.plot(x, y, 'co')

    k = len(paths)-1
    while k>= 0:
        path = paths[k]
        for i in range(0,len(path)-1):
            plt.arrow(x[path[i]-1], y[path[i]-1], (x[path[i+1]-1] - x[path[i]-1]), (y[path[i+1]-1] - y[path[i]-1]), 
                      head_width = 0, color = 'g', length_includes_head = True)
        plt.arrow(x[path[len(path)-1]-1], y[path[len(path)-1]-1], (x[path[0]-1] - x[path[len(path)-1]-1]), 
                  (y[path[0]-1] - y[path[len(path)-1]-1]), head_width = 0, color = 'g', length_includes_head = True)
        k -= 1

    #Set axis too slitghtly larger than the set of x and y
    s_x = 1   #abs(max(x)) - abs(min(x)) * 0.0001
    s_y = 1   #abs(max(y)) - abs(min(y)) * 0.0001
    plt.xlim(min(x) - s_x, max(x) + s_x)
    plt.ylim(min(y) - s_y, max(y) + s_y)
    plt.show()

##################################################
##################################################
# EXAMPLE:

if __name__ == '__main__':
    instance = parse_instance("instances/ulysses16.tsp")
    solution = parse_solution("instances/ulysses16.opt.tour")
    print("Instance:")
    print(instance)
    print("Solution:")
    print(solution)
    print("\n")
    
    # Run the function
    plotTSP([solution], instance)