import matplotlib.pyplot as plt

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

def parse_solution(file_path, points):
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
                    solution.append(points[int(p)-1])

    return solution


# Source:
# https://gist.github.com/payoung/6087046
def plotTSP(paths, points):
    """
    paths: List of lists with the different orders in which the nodes are visited
    points: coordinates for the different nodes
    """
    colors = ['k', 'g', 'r', 'b', 'c', 'm']

    x = []
    y = []

    for x_p, y_p in points:
        x.append(x_p)
        y.append(y_p) 
    plt.plot(x, y, 'co')

    k = len(paths)-1
    col = 0
    while k>= 0:
        path = paths[k]
        startx, starty = path[len(path)-1]
        endx = (0, 0)
        endy = (0, 0)
        for i in range(0,len(path)):
            endx, endy = path[i]
            plt.arrow(startx, starty, endx - startx, endy - starty,
                    head_width = 0, color = colors[col%len(colors)], length_includes_head = True)
            startx = endx
            starty = endy
        #plt.arrow(startx, starty, endx - startx, endy - starty,
        #        head_width = 0, color = colors[col%len(colors)], length_includes_head = True)
        k -= 1
        col += 1

    s_x = 1
    s_y = 1
    plt.xlim(min(x) - s_x, max(x) + s_x)
    plt.ylim(min(y) - s_y, max(y) + s_y)
    plt.show()

##################################################
##################################################
# EXAMPLE:

if __name__ == '__main__':
    instance = parse_instance("instances/ts225.tsp")
    solution = parse_solution("instances/tsp225.opt.tour", instance)
    print("Instance:")
    print(instance)
    print("Solution:")
    print(solution)
    print("\n")
    
    # Run the function
    plotTSP([solution], instance)