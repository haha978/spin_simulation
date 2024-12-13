import numpy as np
import qutip as qt
import yaml

def tensor_product(pauli, i, n):
    op_list = [qt.qeye(2)] * n
    op_list[i] = pauli
    return qt.tensor(op_list)

def dipolar_interaction(i, j, bij, n):
    sx, sy, sz = qt.sigmax(), qt.sigmay(), qt.sigmaz()
    return bij * (1/4) *(2* tensor_product(sz, i, n)*tensor_product(sz, j, n) \
                    - tensor_product(sx, i, n)*tensor_product(sx, j, n) \
                    - tensor_product(sy, i, n)*tensor_product(sy, j, n))

def get_I_component(N, type):
    assert type == 'x' or type == 'y' or type == 'z', "type should be x, y, or z"
    sigma_i = 0
    if type == 'x':
        sigma_i = 1/2*qt.sigmax()
    elif type == 'y':
        sigma_i = 1/2*qt.sigmay()
    elif type == 'z':
        sigma_i = 1/2*qt.sigmaz()
    else:
        ValueError("type must be x, y, or z")
    I_comp = 0
    for j in range(N):
        I_comp += tensor_product(sigma_i, j, N)
    return I_comp

def get_Hp(B_field, N, type):
    """
    Get pulse Hamiltonian
    """
    gamma = get_constants()[1]
    assert type == 'x' or type == 'y' or type == 'z', "pulse type should be x or y"
    Hp = 0
    if type == 'x':
        sigma_p = -gamma/2*B_field*qt.sigmax() 
    elif type == 'y':
        sigma_p = -gamma/2*B_field*qt.sigmay()
    elif type == 'z':
        sigma_p = -gamma/2*B_field*qt.sigmaz()
    Hp = 0
    for i in range(N):
        Hp += tensor_product(sigma_p,i,N)
    # Here B is in Tesla
    return Hp

def get_dipolar_interaction(bij_M):
    Hdd = 0
    N = bij_M.shape[0]
    for i in range(N):
        for j in range(i+1, N):
            bij = bij_M[i][j]
            Hdd += dipolar_interaction(i, j, bij_M[i][j], N)
    return Hdd

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
            new_point = vertices[-1] + np.random.randn(3) * r_max
            condition1, condition2 = True, False
            for vertex in vertices:
                if (np.linalg.norm(new_point - vertex) < r_min):
                    condition1 = False
                if np.linalg.norm(new_point - vertex) < r_max:
                    condition2 = True
            valid = condition1 and condition2
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

def get_B_field(pulse_len, theta):
    mu0, gamma, hbar = get_constants()
    freq = 2 * np.pi/(pulse_len * 360/theta) #[rad/s]
    B_field = freq/gamma
    return B_field

def load_parameters_from_yaml(file_path):
    with open(file_path, 'r') as file:
        parameters = yaml.safe_load(file)
    return parameters

