import matplotlib.pyplot as plt
import os
import numpy as np

def save_poisitons_graph(positions, output_path, fname = "positions.png"):
    """
    Generate 3D graph that shows different positions in output_path as fname
    positions: 3D Numpy array with dimensions (N,3) where N indicate
    
    Arguments:
    positions: numpy array with dimensions (N,3). 
               0th index is index of the 13C and 1st index indicate position of that 13C
    output_path: output path where the position graph is stored
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c='r', marker='o')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('z [m]')
    plt.locator_params(axis = "both", nbins = 5)
    plt.savefig(os.path.join(output_path, fname))

def plot_eigenenergy(H, output_path):
    fig, ax = plt.subplots(nrows = 1, ncols = 1)
    eigen_energies = H.eigenenergies()
    x = np.arange(len(eigen_energies))
    ax.bar(x, eigen_energies, rasterized = True, color = 'b')
    plt.ylabel('Energy [au]')
    plt.title('Eigenenergy distribution \n' + f'mean energy: {np.mean(eigen_energies)}')
    plt.savefig(os.path.join(output_path, "energy_distribution.png"))

def plot_I(t_list, exp_Ix_l, exp_Iy_l, exp_Iz_l, data_dir):
    fig, ax = plt.subplots(nrows = 1, ncols = 3, figsize = (12, 4), layout = "constrained")
    ax[0].scatter(t_list, exp_Ix_l, s = 0.5, rasterized= True, color = 'r', label = 'Ix')
    ax[1].scatter(t_list, exp_Iy_l, s = 0.5, rasterized = True, color = 'b', label = 'Iy')
    ax[2].scatter(t_list, exp_Iz_l, s = 0.5, rasterized = True, color = 'g', label = 'Iz')
    for i in range(3):
        ax[i].set_xlabel("time [s]")
        ax[i].set_ylabel("Signal [au]")
        ax[i].legend(fontsize = 12, markerscale = 5)
    plt.savefig(os.path.join(data_dir, "I_plot.png"))