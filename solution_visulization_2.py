import matplotlib.pyplot as plt


# Source:
# https://stackoverflow.com/questions/46506375/creating-graphics-for-euclidean-instances-of-tsp


# A 2d-array positions of shape (n_points, n_dimension) 
"""
[[  4.17022005e-01   7.20324493e-01]
 [  1.14374817e-04   3.02332573e-01]
 [  1.46755891e-01   9.23385948e-02]
 [  1.86260211e-01   3.45560727e-01]
 [  3.96767474e-01   5.38816734e-01]]
"""
# A 2d-array x_sol which is our MIP-solution marking ~1 when node x is followed by y in our solution-tour, like:
"""
[[  0.00000000e+00   1.00000000e+00  -3.01195977e-11   2.00760084e-11
    2.41495095e-11]
 [ -2.32741108e-11   1.00000000e+00   1.00000000e+00   5.31351363e-12
   -6.12644932e-12]
 [  1.18655962e-11   6.52816609e-12   0.00000000e+00   1.00000000e+00
    1.42473796e-11]
 [ -4.19937042e-12   3.40039727e-11   2.47921345e-12   0.00000000e+00
    1.00000000e+00]
 [  1.00000000e+00  -2.65096995e-11   3.55630808e-12   7.24755899e-12
    1.00000000e+00]]
"""

fig, ax = plt.subplots(2, sharex=True, sharey=True)         # Prepare 2 plots
ax[0].set_title('Raw nodes')
ax[1].set_title('Optimized tour')
ax[0].scatter(positions[:, 0], positions[:, 1])             # plot A
ax[1].scatter(positions[:, 0], positions[:, 1])             # plot B
start_node = 0
distance = 0.
for i in range(N):
    start_pos = positions[start_node]
    next_node = np.argmax(x_sol[start_node]) # needed because of MIP-approach used for TSP
    end_pos = positions[next_node]
    ax[1].annotate("",
            xy=start_pos, xycoords='data',
            xytext=end_pos, textcoords='data',
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3"))
    distance += np.linalg.norm(end_pos - start_pos)
    start_node = next_node

textstr = "N nodes: %d\nTotal length: %.3f" % (N, distance)
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax[1].text(0.05, 0.95, textstr, transform=ax[1].transAxes, fontsize=14, # Textbox
        verticalalignment='top', bbox=props)

plt.tight_layout()
plt.show()