import numpy as np
import qutip as qt
from utils import dipolar_interaction, generate_random_graph, get_constants, get_interaction_matrix
from utils_plot import save_poisitons_graph
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
    print(f"This is mean of dipolar coupling: {np.mean(np.abs(bij_M))/(2*np.pi)}Hz")
    Hdd = 0
    for i in range(N):
        for j in range(i+1, N):
            bij = bij_M[i][j]
            Hdd += dipolar_interaction(i, j, bij_M[i][j], N)



    

if __name__ == '__main__':
    main()