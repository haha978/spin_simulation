import numpy as np
import qutip as qt
from utils import generate_random_graph, get_interaction_matrix
from utils import get_dipolar_interaction, get_Hp, get_I_component
from utils_plot import save_poisitons_graph
from qutip_qip.operations import ry
import matplotlib.pyplot as plt
import argparse
import os

def get_args(parser):
    parser.add_argument('--N', type = int, default = 10, help = "number of spins (N=10).")
    parser.add_argument('--output_path', type = str, required = True, help = "PATH to output directory.")
    args = parser.parse_args()
    return args

def main():
    parser = argparse.ArgumentParser(description = "Train image model")
    args = get_args(parser)
    N = args.N
    OUTPUT_PATH = args.output_path
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    positions_path = os.path.join(OUTPUT_PATH,"positions.npy")
    if os.path.exists(positions_path):
        print("Loading positions from output_path")
        positions = np.load(positions_path, allow_pickle=True)
    else:
        # This is hard-coded for now
        print("generating new positions")
        diamond_inter_dist = 1.54e-10
        r_min, r_max = diamond_inter_dist * 1, diamond_inter_dist * 7

        # generate random graph
        positions = generate_random_graph(N = N, r_min = r_min, r_max = r_max, seed = 10)
        
        # save positions and plots 
        np.save(os.path.join(OUTPUT_PATH, "positions.npy"), positions, allow_pickle = True)
        save_poisitons_graph(positions = positions, output_path = OUTPUT_PATH, fname = "positions.png")
        
    bij_M = get_interaction_matrix(positions)
    B_field = 100e-6 #in Tesla
    # print(f"This is mean of dipolar coupling: {np.mean(np.abs(bij_M))/(2*np.pi)}Hz")
    Hdd = get_dipolar_interaction(bij_M)
    Hp = get_Hp(B_field, N, type = 'y')
    
    #initialize all state x polarized state
    init_state = qt.tensor([ry(np.pi/2)*qt.basis(2, 0)]*10)
    t_step = 1e-5
    t_list = np.arange(0, 5e-2-t_step/2, t_step)
    result = qt.sesolve(Hp, init_state, t_list)

    # measure x-polarization
    Ix, Iy, Iz = get_I_component(N, 'x'), get_I_component(N, 'y'), get_I_component(N, 'z')
    exp_Ix_l = qt.expect(Ix, result.states)
    exp_Iy_l = qt.expect(Iy, result.states)
    exp_Iz_l = qt.expect(Iz, result.states)

    # now plot the expectation value
    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (6, 6))
    ax.plot(t_list, exp_Ix_l, linewidth = 1, rasterized= True, color = 'r', label = 'Ix')
    ax.plot(t_list, exp_Iy_l, linewidth = 1, rasterized = True, color = 'b', label = 'Iy')

    fft_values = np.fft.fft(exp_Ix_l)
    fft_magnitudes = np.abs(fft_values)

    freqs = np.fft.fftfreq(len(t_list), t_step)
    max_idx = np.argmax(fft_magnitudes)
    max_freq = freqs[max_idx]
    print(f"This is max frequency {max_freq}")
    # ax.plot(t_list, exp_Ix_l, s = 1, rasterized= True, color = 'g', label = 'Iz')
    ax.legend(fontsize = 12)
    plt.show()
if __name__ == '__main__':
    main()