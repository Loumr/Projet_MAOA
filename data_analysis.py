import pandas
import matplotlib.pyplot as plt

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

def display_instance():
    pass

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

    # Set a scale for the arrow heads
    a_scale = float(max(x))/float(100)
    k = len(paths)-1
    while k>= 0:
        path = paths[k]
        print("len x=" + str(len(x)) + ", len y= " + str(len(y)) + ", len path=" + str(len(path)))
        for i in range(0,len(path)-1):
            print("it:" + str(i) + ", point:" + str(path[i]) + ", next_point:" + str(path[i+1]))
            plt.arrow(x[path[i]-1], y[path[i]-1], (x[path[i+1]-1] - x[path[i]-1]), (y[path[i+1]-1] - y[path[i]-1]), 
                      head_width = a_scale, color = 'g', length_includes_head = True)
        k -= 1

    #Set axis too slitghtly larger than the set of x and y
    plt.xlim(0, max(x)*1.1)
    plt.ylim(0, max(y)*1.1)
    plt.show()

##################################################
##################################################

if __name__ == '__main__':
    # Run an example
    instance = parse_instance("instances/ulysses16.tsp")
    solution = parse_solution("instances/ulysses16.opt.tour")
    print("Instance:")
    print(instance)
    print("Solution:")
    print(solution)
    print("\n")
    
    # Run the function
    plotTSP([solution], instance)