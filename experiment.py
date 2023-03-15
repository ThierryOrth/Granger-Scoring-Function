import numpy as np
from functional_dependencies import *


# TODO: assign p-value matrices only for those cells in the PCDAG where m[i,j]==1

# TODO: construct p-value matrix for all (V_i, V_j) s.t. m[i,j]==1

# TODO: for different densities, simply build up the p-value matrix from what we have before

# TODO: for a different number of variables, we use a new p-value matrix

# TODO: save data in folders to save time

if __name__ == "__main__":
    N = 100
    time_params = np.arange(10, N+1, 10)
    coeff_params = np.concatenate((np.arange(-0.1, -1.0, -0.1),\
                                            np.arange(0.1,1,0.1)))
    var_params = np.arange(3, 20)
    density_params = [[np.arange(1, n+1, 1)] for n in var_params]