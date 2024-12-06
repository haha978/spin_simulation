import numpy as np
import qutip as qt

def tensor_product(pauli, i, n):
    op_list = [qt.qeye(2)] * n
    op_list[i] = pauli
    return qt.tensor(op_list)

def dipolar_interaction(i, j, bij, n):
    sx, sy, sz = qt.sigmax(), qt.sigmay(), qt.sigmaz()
    return bij * (1/4) *(2* tensor_product(sz, i, n)*tensor_product(sz, j, n) \
                    - tensor_product(sx, i, n)*tensor_product(sx, j, n) \
                    - tensor_product(sy, i, n)*tensor_product(sy, j, n))

def generate_random_graph(N, r_min, r_max, seed, max_attempts = 1000):
    """
    Return numpy array with size (N, 3), where each index indicate position

    Parameter:
    N: number of spins
    r_min: minimum distance between the vertices
    r_max: maximum distance between the vertices
    seed: random seed
    """
    # initialize spin 1 in to 0, 0, 0 position
    vertices = []
    vertices.append(np.zeros(3))
    
    while len(vertices) < N:
        attempts = 0
        while attempts < max_attempts:
            new_point = np.random.randn(3) * r_max
            valid = False
            for vertex in vertices:
                if (r_min < np.linalg.norm(new_point - vertex) < r_max):
                    valid = True
                    break
            if valid:
                vertices.append(new_point)
                break
            attempts += 1
        if attempts == max_attempts:
            raise ValueError("Failed to generate a valid point within the maximum number of attempts.")
    return np.array(vertices)

def get_constants():
    mu0 = 1.25663706127 * 10**(-6) #[N][s]^2/[C]^2
    gamma = 10.705 * 2*np.pi * 10**(6) # ([C][m][rad])/([s]^2[N])
    hbar = 1.054571817 * 10**(-34) #[N][m][s]/[rad]
    return mu0, gamma, hbar

def get_interaction_matrix(positions):
    # Create bij_M matrix
    mu0, gamma, hbar = get_constants()
    N = positions.shape[0]
    bij_M = np.zeros((N, N))
    for i in range(N):
        for j in range(i+1, N):
            ri, rj = positions[i], positions[j]
            rij = ri - rj
            cos_Bij = np.linalg.norm(rij[2])/np.linalg.norm(rij)
            # factor does not consider 
            factor = mu0/(4*np.pi)*(gamma**2)*hbar*(3*cos_Bij**2 - 1)/np.linalg.norm(rij)**3
            # now need some secular approximation factors.
            bij_M[i][j] = factor
            bij_M[j][i] = factor
    return bij_M

