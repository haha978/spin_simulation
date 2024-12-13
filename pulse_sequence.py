import numpy as np
import qutip as qt
from utils import get_Hp, get_dipolar_interaction
from utils import get_B_field
from abc import ABC, abstractmethod


class PulseSequence(ABC):
    @abstractmethod
    def check_param_dict(self, param_dict):
        pass

    @abstractmethod
    def get_Hamiltonian(self):
        pass

class SpinLock(PulseSequence):
    def __init__(self, param_dict):
        self.check_param_dict(param_dict)
        self.sequence_name = param_dict['sequence_name']
        self.pulse_length = param_dict['pulse_length']
        self.theta = param_dict['theta']
        self.spacing = param_dict['spacing']
        self.inter_pulse_dipole = param_dict['inter_pulse_dipole']

    def check_param_dict(self, param_dict):
        required_keys = {'sequence_name', 'pulse_length', 'theta', 'spacing', 'inter_pulse_dipole'}
        assert required_keys.issubset(param_dict.keys()), f"required keys dont exists in the param_dict/yamlfile {required_keys}"

    def get_Hamiltonian(self, bij_M):
        N = bij_M.shape[0]
        T = self.pulse_length + self.spacing
        B_field = get_B_field(self.pulse_length, self.theta)
        Hp = get_Hp(B_field, N, 'x')
        Hdd = get_dipolar_interaction(bij_M)
        def spin_lock_Hamiltonian(t):
            tf = t - t//T*T
            if 0 <= tf <= self.pulse_length:
                if self.inter_pulse_dipole:
                    return Hp + Hdd
                else:
                    return Hp
            else:
                return Hdd
        return spin_lock_Hamiltonian

class SpinLock_Square_AC(PulseSequence):
    def __init__(self, param_dict):
        self.check_param_dict(param_dict)
        self.sequence_name = param_dict['sequence_name']
        self.pulse_length = param_dict['pulse_length']
        self.theta = param_dict['theta']
        self.spacing = param_dict['spacing']
        self.AC_field = param_dict['AC_field']
        self.inter_pulse_dipole = param_dict['inter_pulse_dipole']
    
    def check_param_dict(self, param_dict):
        required_keys = {"sequence_name", "pulse_length", "theta", "spacing", "inter_pulse_dipole", "AC_field"}
        assert required_keys.issubset(param_dict.keys()), f"required keys dont exists in the param_dict/yamlfile {required_keys}"

    def get_Hamiltonian(self, bij_M):
        N = bij_M.shape[0]
        T = 4 * (self.pulse_length + self.spacing)
        B_field = get_B_field(self.pulse_length, self.theta)
        Hp = get_Hp(B_field, N, 'x')
        Hac = get_Hp(self.AC_field, N, 'z')
        Hdd = get_dipolar_interaction(bij_M)
        def AC_Ham(t):
            tf = t - t//T*T
            if 0 <= tf <= self.pulse_length/2 or \
                (T + self.pulse_length)/2 <= tf <= T:
                return Hac
            else:
                return -Hac
        def spin_lock_AC_square_Hamiltonian(t):
            tf = t - t//T*T
            Ham = 0
            if (0 <= tf < self.pulse_length) or (T/4 <= tf < T/4 + self.pulse_length) \
                or (T/2 <= tf < T/2+self.pulse_length) or (3*T/4 <= tf < 3*T/4 + self.pulse_length):
                if self.inter_pulse_dipole:
                    Ham = Hp + Hdd
                else:
                    Ham = Hp
            else:
                Ham = Hdd
            
            Ham = Ham + AC_Ham(t)
            return Ham
        return spin_lock_AC_square_Hamiltonian