import numpy as np
import qutip as qt
from utils import generate_random_graph, get_interaction_matrix
from utils import get_I_component, get_B_field
from utils_plot import save_poisitons_graph, plot_I
from qutip_qip.operations import ry
from pulse_sequence import spin_lock_Hamiltonian
import matplotlib.pyplot as plt
import argparse
import os

def get_args(parser):
    parser.add_argument('--N', type = int, default = 10, help = "number of spins (N=10).")
    parser.add_argument('--output_path', type = str, required = True, help = "PATH to output directory.")
    parser.add_argument('--t_total', type = float, required = True, help = "Total simulation time")
    parser.add_argument('--t_step', type = float, default = 5e-6, help = "time step for simulation[default: 5e-6]")
    args = parser.parse_args()
    return args

def main():
    parser = argparse.ArgumentParser(description = "Train image model")
    args = get_args(parser)
    N = args.N
    OUTPUT_PATH = args.output_path
    t_total = args.t_total
    t_step = args.t_step
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
    bij_M_path = os.path.join(OUTPUT_PATH, "bij_M.npy")
    if os.path.exists(bij_M_path):
        print("Loading bij matrix")
        bij_M = np.load(bij_M_path, allow_pickle = True)
    else:
        print("generating bij matrix")
        bij_M = get_interaction_matrix(positions)
        np.save(bij_M_path, bij_M, allow_pickle = True)
    
    #initialize all state x polarized state
    init_state = qt.tensor([ry(np.pi/2)*qt.basis(2, 0)]*10)
    t_list = np.arange(0, t_total-t_step/2, t_step)
    pulse_len, theta = 55e-6, 90
    spacing = 100e-6
    
    # theta is in degrees
    B_field = get_B_field(pulse_len, theta)

    Ham = spin_lock_Hamiltonian(N, B_field, pulse_len, spacing, p_type = 'x', bij_M = bij_M)
    
    DATA_DIR = os.path.join(OUTPUT_PATH, f't_total_{t_total}_t_step_{t_step}')
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    exp_Ix_p = os.path.join(DATA_DIR, "exp_Ix_l.npy")
    exp_Iy_p = os.path.join(DATA_DIR, "exp_Iy_l.npy")
    exp_Iz_p = os.path.join(DATA_DIR, "exp_Iz_l.npy")
    
    b1 = os.path.exists(exp_Ix_p)
    b2 = os.path.exists(exp_Iy_p)
    b3 = os.path.exists(exp_Iz_p)

    if not (b1 and b2 and b3):
        Ix, Iy, Iz = get_I_component(N, 'x'), get_I_component(N, 'y'), get_I_component(N, 'z')
        e_ops = [Ix, Iy, Iz]
        # Probably will need to make this shorter/longer etc.
        result = qt.sesolve(Ham, init_state, t_list, e_ops = e_ops, options = {"store_states": False})
        exp_Ix_l, exp_Iy_l, exp_Iz_l = result.expect[0], result.expect[1], result.expect[2]
        np.save(exp_Ix_p, exp_Ix_l)
        np.save(exp_Iy_p, exp_Iy_l)
        np.save(exp_Iz_p, exp_Iz_l)
    else:
        exp_Ix_l = np.load(exp_Ix_p, allow_pickle= True)
        exp_Iy_l = np.load(exp_Iy_p, allow_pickle= True)
        exp_Iz_l = np.load(exp_Iz_p, allow_pickle= True)
    # now plot the expectation value
    plot_I(t_list, exp_Ix_l, exp_Iy_l, exp_Iz_l, DATA_DIR)
if __name__ == '__main__':
    main()