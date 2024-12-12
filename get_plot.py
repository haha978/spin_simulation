import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from utils import load_parameters_from_yaml

def get_args(parser):
    parser.add_argument('--input_path', type = str, required = True, help = "PATH to where the data is stored.")
    args = parser.parse_args()
    return args


def plot_every_4(t_list, pulse_length, spacing, t_step, Ii_l):
    T = pulse_length + spacing
    plen = int((pulse_length + t_step/100) // t_step)
    slen = int((spacing + t_step/100) / t_step)
    floquet_len = int((T + t_step/100) // t_step)
    
    idx0_l, idx1_l = [], []
    idx2_l, idx3_l = [], []
    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (4, 4), layout = "constrained")
    for t_idx, t_val in enumerate(t_list):
        micromotion_idx = t_idx % floquet_len
        tf_idx = t_idx // floquet_len
        if plen <= micromotion_idx <= floquet_len:
            if tf_idx % 4 == 0:
                idx0_l.append(t_idx)
            elif tf_idx % 4 == 1:
                idx1_l.append(t_idx)
            elif tf_idx % 4 == 2:
                idx2_l.append(t_idx)
            else:
                idx3_l.append(t_idx)
    idx0_l, idx1_l = np.array(idx0_l), np.array(idx1_l)
    idx2_l, idx3_l = np.array(idx2_l), np.array(idx3_l)
    ax.scatter(t_list[idx0_l], Ii_l[idx0_l], color = 'r', rasterized = True, s = 0.5)
    ax.scatter(t_list[idx1_l], Ii_l[idx1_l], color = 'g', rasterized = True, s = 0.5)
    ax.scatter(t_list[idx2_l], Ii_l[idx2_l], color = 'b', rasterized = True, s = 0.5)
    ax.scatter(t_list[idx3_l], Ii_l[idx3_l], color = 'c', rasterized = True, s = 0.5)
    max_Ii = max(Ii_l)
    ax.set_ylim(0, max_Ii + max_Ii/10)
    plt.show()

def main():
    parser = argparse.ArgumentParser(description = "Train image model")
    args = get_args(parser)
    INPUT_PATH = args.input_path
    t_list_p = os.path.join(INPUT_PATH, "t_list.npy")
    Ix_l_p = os.path.join(INPUT_PATH, "exp_Ix_l.npy")
    Iy_l_p = os.path.join(INPUT_PATH, "exp_Iy_l.npy")
    Iz_l_p = os.path.join(INPUT_PATH, "exp_Iz_l.npy")
    
    # get param_dict
    param_dict_p = os.path.join(INPUT_PATH, "param_dict.yaml")
    if os.path.exists(param_dict_p):
        param_dict = load_parameters_from_yaml(param_dict_p)
    else:
        raise ValueError("parameter dictionary missing")

    t_list_b, Ix_l_b = os.path.exists(t_list_p), os.path.exists(Ix_l_p)
    Iy_l_b, Iz_l_b = os.path.exists(Iy_l_p), os.path.exists(Iz_l_p)
    if (t_list_b and Ix_l_b and Iy_l_b and Iz_l_b):
        t_list, Ix_l = np.load(t_list_p), np.load(Ix_l_p) 
        Iy_l, Iz_l = np.load(Iy_l_p), np.load(Iz_l_p)
    else:
        raise ValueError("Files missing in input path")
    
    pulse_length = param_dict["pulse_length"]
    spacing = param_dict["spacing"]
    t_step = param_dict["t_step"]

    # fig, ax = plt.subplots(nrows = 1, ncols = 3, figsize = (12, 4), layout = "constrained")
    # ax[0].scatter(t_list, Ix_l, s = 0.5, rasterized= True, color = 'r', label = 'Ix')
    # ax[1].scatter(t_list, Iy_l, s = 0.5, rasterized = True, color = 'b', label = 'Iy')
    # ax[2].scatter(t_list, Iz_l, s = 0.5, rasterized = True, color = 'g', label = 'Iz')
    # for i in range(3):
    #     ax[i].set_xlabel("time [s]")
    #     ax[i].set_ylabel("Signal [au]")
    #     ax[i].legend(fontsize = 12, markerscale = 5)
    # plt.show()

    plot_every_4(t_list, pulse_length, spacing, t_step, Ix_l)


if __name__ == '__main__':
    main()