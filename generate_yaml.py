import os
import yaml
import numpy as np
from utils import load_parameters_from_yaml

"""
sequence_name: "spin_lock_square_AC"
pulse_length: 60.e-6
theta: 90
spacing: 100.e-6
inter_pulse_dipole: False
AC_field: 20.e-6 
"""

def generate_AC_square_yaml():
    DATA_DIR_PATH = "/home/jm2239/spin_simulation/pulse_sequence_yaml"
    param_dict = {"sequence_name": "spin_lock_square_AC", "pulse_length": 60.e-6, "theta": 90, "spacing": 100.e-6, \
                  "inter_pulse_dipole": False, "AC_field": []}
    AC_step = 5.e-6
    AC_field_arr = np.round(np.arange(0, 60.e-6 + AC_step/2, AC_step), 9)
    for AC_field in AC_field_arr:
        param_dict["AC_field"].append(AC_field.item())
    with open(os.path.join(DATA_DIR_PATH, "AC_param.yaml"), "w") as f:
        yaml.dump(param_dict, f, default_flow_style=False, sort_keys=False)


def main():
    generate_AC_square_yaml()

if __name__ == '__main__':
    main()