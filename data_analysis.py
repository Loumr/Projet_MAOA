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
        if reading_points:
            if "EOF" in line:
                break
            parts = line.split()
            if len(parts) >= 3:
                coordinates = (float(parts[1]), float(parts[2]))
                instance.append(coordinates)
    
    return instance

def display_instance():
    pass

def parse_solution():
    pass


# Source:
# https://gist.github.com/payoung/6087046
def plotTSP(paths, points, num_iters=1):

    """
    paths: List of lists with the different orders in which the nodes are visited
    points: coordinates for the different nodes
    num_iters: number of paths that are in the path list
    
    """

    # Unpack the primary TSP path and transform it into a list of ordered 
    # coordinates

    x = []; y = []

    
    for x_p, y_p in points:
        x.append(x_p)
        y.append(y_p)
        
    plt.plot(x, y, 'co')

    # Set a scale for the arrow heads
    a_scale = float(max(x))/float(100)

    # Draw the older paths, if provided
    if num_iters > 1 and len(paths) > 1:

        for i in range(1, min(len(paths), num_iters)):

            # Transform the old paths into a list of coordinates
            xi = []; yi = [];
            for j in paths[i]:
                xi.append(points[j][0])
                yi.append(points[j][1])

            plt.arrow(xi[-1], yi[-1], (xi[0] - xi[-1]), (yi[0] - yi[-1]), 
                    head_width = a_scale, color = 'r', 
                    length_includes_head = True, ls = 'dashed',
                    width = 0.001/float(num_iters))
            for i in range(0, len(x) - 1):
                plt.arrow(xi[i], yi[i], (xi[i+1] - xi[i]), (yi[i+1] - yi[i]),
                        head_width = a_scale, color = 'r', length_includes_head = True,
                        ls = 'dashed', width = 0.001/float(num_iters))

    if len(paths) > 0:
        # Draw the primary path for the TSP problem
        plt.arrow(x[-1], y[-1], (x[0] - x[-1]), (y[0] - y[-1]), head_width = a_scale, 
                color ='g', length_includes_head=True)
        for i in range(0,len(x)-1):
            plt.arrow(x[i], y[i], (x[i+1] - x[i]), (y[i+1] - y[i]), head_width = a_scale,
                    color = 'g', length_includes_head = True)

    #Set axis too slitghtly larger than the set of x and y
    plt.xlim(0, max(x)*1.1)
    plt.ylim(0, max(y)*1.1)
    plt.show()

if __name__ == '__main__':
    # Run an example
    
    # Create a randomn list of coordinates, pack them into a list
    x_cor = [1, 8, 4, 9, 2, 1, 8]
    y_cor = [1, 2, 3, 4, 9, 5, 7]
    points = []
    for i in range(0, len(x_cor)):
        points.append((x_cor[i], y_cor[i]))

    # Create two paths, teh second with two values swapped to simulate a 2-OPT
    # Local Search operation
    path4 = [0, 1, 2, 3, 4, 5, 6]
    path3 = [0, 2, 1, 3, 4, 5, 6]
    path2 = [0, 2, 1, 3, 6, 5, 4]
    path1 = [0, 2, 1, 3, 6, 4, 5]

    # Pack the paths into a list
    paths = [path1]
    
    # Run the function
    plotTSP(paths, points, 1)