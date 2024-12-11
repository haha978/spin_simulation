import numpy as np
import qutip as qt
from utils import get_Hp, get_dipolar_interaction

def spin_lock_Hamiltonian(N, B_field, pulse_len, spacing, p_type, bij_M):
    T = pulse_len + spacing
    Hp = get_Hp(B_field, N, p_type)
    Hdd = get_dipolar_interaction(bij_M)
    def spin_lock(t):
        tf = t - t//T*T
        if 0 <= tf <= pulse_len:
            return Hp+Hdd
        else:
            return Hdd
    return spin_lock

def spin_lock_AC_field_Hamiltonian(N, B_field, pulse_len, spacing, p_type, bij_M):
    pass


def DTC_seq():
    pass