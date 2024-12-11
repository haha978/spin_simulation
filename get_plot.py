import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def get_args(parser):
    parser.add_argument('--input_path', type = str, required = True, help = "PATH to where the data is stored.")
    args = parser.parse_args()
    return args


def plot_every_4(t_list, Ii_l):
    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (4, 4), layout = "constrained")
    ax.scatter(t_list[::4], Ii_l[::4], color = 'r', rasterized = True, s = 0.5)
    ax.scatter(t_list[1::4], Ii_l[1::4], color = 'm', rasterized = True, s = 0.5)
    ax.scatter(t_list[2::4], Ii_l[2::4], color = 'c', rasterized = True, s = 0.5)
    ax.scatter(t_list[3::4], Ii_l[3::4], color = 'b', rasterized = True, s = 0.5)
    plt.show()

def main():
    parser = argparse.ArgumentParser(description = "Train image model")
    args = get_args(parser)
    INPUT_PATH = args.input_path
    t_list_p = os.path.join(INPUT_PATH, "t_list.npy")
    Ix_l_p = os.path.join(INPUT_PATH, "exp_Ix_l.npy")
    Iy_l_p = os.path.join(INPUT_PATH, "exp_Iy_l.npy")
    Iz_l_p = os.path.join(INPUT_PATH, "exp_Iz_l.npy")
    
    t_list_b = os.path.exists(t_list_p)
    Ix_l_b = os.path.exists(Ix_l_p)
    Iy_l_b = os.path.exists(Iy_l_p)
    Iz_l_b = os.path.exists(Iz_l_p)
    if (t_list_b and Ix_l_b and Iy_l_b and Iz_l_b):
        t_list = np.load(t_list_p)
        Ix_l = np.load(Ix_l_p)
        Iy_l = np.load(Iy_l_p)
        Iz_l = np.load(Iz_l_p)
    else:
        raise ValueError("Files missing in input path")
    fig, ax = plt.subplots(nrows = 1, ncols = 3, figsize = (12, 4), layout = "constrained")
    ax[0].scatter(t_list, Ix_l, s = 0.5, rasterized= True, color = 'r', label = 'Ix')
    ax[1].scatter(t_list, Iy_l, s = 0.5, rasterized = True, color = 'b', label = 'Iy')
    ax[2].scatter(t_list, Iz_l, s = 0.5, rasterized = True, color = 'g', label = 'Iz')
    for i in range(3):
        ax[i].set_xlabel("time [s]")
        ax[i].set_ylabel("Signal [au]")
        ax[i].legend(fontsize = 12, markerscale = 5)
    plt.show()

    plot_every_4(t_list, Ix_l)


if __name__ == '__main__':
    main()