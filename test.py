import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utils import generate_random_graph, get_list_k_v_in_dict


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

def get_list_in_key_test():
    arr = list(range(0, 5, 1))
    param_dict = {"key2": "Hello", "AC_field": 1}
    for (k, v) in param_dict.items():
        print(k)
        print(v)
    k, v = get_list_k_v_in_dict(param_dict)
    print(k, v)


def main():
    if True:
        get_list_in_key_test()
    else:
        test_generate_random_graph()

if __name__ == '__main__':
    main()