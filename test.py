import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utils import generate_random_graph


def test_generate_random_graph():
    N = 10
    diamond_inter_dist = 1.54e-10
    r_min = diamond_inter_dist * 5
    r_max = diamond_inter_dist * 8
    graph = generate_random_graph(N, r_min, r_max, seed = 10)

    all_pts_valid = True
    for p_idx, point in enumerate(graph):    
        condition1, condition2 = True, False
        for idx, pt2 in enumerate(graph):
            if idx != p_idx:
                if np.linalg.norm(pt2 - point) < r_min:
                    condition1 = False
                if np.linalg.norm(pt2 - point) < r_max:
                    condition2 = True
        valid = condition1 and condition2
        all_pts_valid = all_pts_valid and valid
    assert all_pts_valid, "This satisfies the random graph condition"

    # Plotting the points in 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(graph[:, 0], graph[:, 1], graph[:, 2], c='r', marker='o')

    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('z [m]')
    plt.show()


def main():
    # init_state = qt.tensor([ry(np.pi/2)*qt.basis(2, 0)]*10)
    # print(init_state)
    # N = 10
    # ans1= get_I_component(10, 'y')
    # params = load_parameters_from_yaml("/home/jm2239/spin_simulation/pulse_sequence/spinlock.yaml")
    # print(params['pulse_length'])
    # print(type(params['pulse_length']))
    # print(params)
    # arr = np.arange(1, 99+0.1, 1)
    # index_arr = np.tile(np.concatenate((np.full(6, False), np.full(3, True))), reps = 11)
    # print(index_arr)
    # print(arr[index_arr])
    test_generate_random_graph()

if __name__ == '__main__':
    main()