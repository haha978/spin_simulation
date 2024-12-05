import numpy as np
import qutip as qt
from utils import dipolar_interaction




def main():
    N = 10
    Hdd = 0
    for i in range(N):
        for j in range(i+1, N):
            print( f"This is {i} and {j}")
            Hdd += dipolar_interaction(i, j, 1, N)
    print(Hdd)

if __name__ == '__main__':
    main()