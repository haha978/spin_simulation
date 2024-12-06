import matplotlib.pyplot as plt
import os

def save_graph(positions, output_path, fname = "positions.png"):
    """
    Generate 3D graph that shows different positions in output_path as fname
    positions: 3D Numpy array with dimensions (N,3) where N indicate

    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c='r', marker='o')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('z [m]')
    plt.locator_params(axis = "both", nbins = 5)
    plt.savefig(os.path.join(output_path, fname))