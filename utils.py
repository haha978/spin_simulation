import numpy as np
import qutip as qt

def tensor_product(pauli, i, n):
    op_list = [qt.qeye(2)] * n
    op_list[i] = pauli
    return qt.tensor(op_list)

def dipolar_interaction(i, j, bij, n):
    sx, sy, sz = qt.sigmax(), qt.sigmay(), qt.sigmaz()
    return bij * (3* tensor_product(sz, i, n)*tensor_product(sz, j, n) - tensor_product(sx, i, n)*tensor_product(sx, j, n) \
                  - tensor_product(sy, i, n)*tensor_product(sy, j, n) - tensor_product(sz, i, n)*tensor_product(sz, j, n))

