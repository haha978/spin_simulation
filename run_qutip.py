import numpy as np
import qutip as qt
from utils import generate_random_graph, get_interaction_matrix
from utils import get_I_component, get_B_field, load_parameters_from_yaml
from utils_plot import save_poisitons_graph, plot_I
from qutip_qip.operations import ry
import pulse_sequence
import matplotlib.pyplot as plt
import argparse
import os
import yaml

def get_args(parser):
    parser.add_argument('--N', type = int, default = 10, help = "number of spins (N=10).")
    parser.add_argument('--output_path', type = str, required = True, help = "PATH to output directory.")
    parser.add_argument('--t_total', type = float, required = True, help = "Total simulation time")
    parser.add_argument('--t_step', type = float, default = 5e-6, help = "time step for simulation[default: 5e-6]")
    parser.add_argument('--yaml', type = str, required = True, help = "PATH to yaml file that defines the pulse sequence")
    parser.add_argument('--data_dir', type = str, help = "Directory name that stores data. If not provided, defaults to default directory name")
    args = parser.parse_args()
    return args

def main():
    parser = argparse.ArgumentParser(description = "spin simulation")
    args = get_args(parser)
    N = args.N
    OUTPUT_PATH = args.output_path
    t_total = args.t_total
    t_step = args.t_step
    YAML_PATH = args.yaml
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    # GENERATING positions in a random graph
    positions_path = os.path.join(OUTPUT_PATH,"positions.npy")
    if os.path.exists(positions_path):
        print("Loading positions from output_path")
        positions = np.load(positions_path, allow_pickle=True)
    else:
        # This is hard-coded for now
        print("generating new positions")
        diamond_inter_dist = 1.54e-10
        r_min, r_max = diamond_inter_dist * 5, diamond_inter_dist * 8

        # generate random graph
        positions = generate_random_graph(N = N, r_min = r_min, r_max = r_max, seed = 10)
        
        # save positions and plots 
        np.save(os.path.join(OUTPUT_PATH, "positions.npy"), positions, allow_pickle = True)
        save_poisitons_graph(positions = positions, output_path = OUTPUT_PATH, fname = "positions.png")
    
    # CALCULATING/LOADING bij matrix 
    bij_M_path = os.path.join(OUTPUT_PATH, "bij_M.npy")
    if os.path.exists(bij_M_path):
        print("Loading bij matrix")
        bij_M = np.load(bij_M_path, allow_pickle = True)
    else:
        print("generating bij matrix")
        bij_M = get_interaction_matrix(positions)
        np.save(bij_M_path, bij_M, allow_pickle = True)
    
    # initialize state to x polarized state
    init_state = qt.tensor([ry(np.pi/2)*qt.basis(2, 0)]*10)
    t_list = np.arange(0, t_total-t_step/2, t_step)

    # LOAD PULSE SEQUENCE PARAMETERS 
    param_dict = load_parameters_from_yaml(YAML_PATH)
    sequence_name = param_dict["sequence_name"]
    if sequence_name == 'spin_lock':
        seq = pulse_sequence.SpinLock(param_dict)
        Ham = seq.get_Hamiltonian(bij_M)
    elif sequence_name == 'spin_lock_square_AC':
        seq = pulse_sequence.SpinLock_Square_AC(param_dict)
        Ham = seq.get_Hamiltonian(bij_M)
    else:
        ValueError("pulse sequence not supported")

    # MAKE DATA DIRECTORY THAT STORES ALL DATA AND HYPERPARAMETERS
    DATA_DIR = args.data_dir
    
    if DATA_DIR == None:
        DATA_DIR_PATH = os.path.join(OUTPUT_PATH, f't_total_{t_total}_t_step_{t_step}_sn_{sequence_name}')
    else:
        DATA_DIR_PATH = os.path.join(OUTPUT_PATH, DATA_DIR)
        if not os.path.exists(DATA_DIR_PATH):
            os.makedirs(DATA_DIR_PATH)

    param_dict['t_total'], param_dict['t_step'] = args.t_total, args.t_step
    
    # GENERATE PATH NAMES WHERE THE DATA WILL BE STORED
    param_dict_p = os.path.join(DATA_DIR_PATH, "param_dict.npy")
    t_list_p = os.path.join(DATA_DIR_PATH, "t_list.npy")
    exp_Ix_p = os.path.join(DATA_DIR_PATH, "exp_Ix_l.npy")
    exp_Iy_p = os.path.join(DATA_DIR_PATH, "exp_Iy_l.npy")
    exp_Iz_p = os.path.join(DATA_DIR_PATH, "exp_Iz_l.npy")
    
    # GENERATE OBSERVABLES
    Ix, Iy, Iz = get_I_component(N, 'x'), get_I_component(N, 'y'), get_I_component(N, 'z')
    e_ops = [Ix, Iy, Iz]

    # TIME EVOLUTION
    result = qt.sesolve(Ham, init_state, t_list, e_ops = e_ops, options = {"store_states": False})
    exp_Ix_l, exp_Iy_l, exp_Iz_l = result.expect[0], result.expect[1], result.expect[2]
    
    # SAVE ALL DATA / HYPERPARMETERS
    np.save(t_list_p, t_list)
    np.save(exp_Ix_p, exp_Ix_l)
    np.save(exp_Iy_p, exp_Iy_l)
    np.save(exp_Iz_p, exp_Iz_l)
    with open(os.path.join(DATA_DIR_PATH, "param_dict.yaml"), "w") as f:
        yaml.dump(param_dict, f, default_flow_style=False, sort_keys=False)
    
    # PLOT AND SAVE the expectation value
    plot_I(t_list, exp_Ix_l, exp_Iy_l, exp_Iz_l, DATA_DIR_PATH)
if __name__ == '__main__':
    main()