import itertools, iteration_utilities, indep_tests, random
import numpy as np
import random

def get_params() -> tuple:
    """
        :returns var_params:
            range with model dimensionality parameters
        :returns coeff_params:
            range with coefficients defining causal influence 
        :returns func_params:
            set of functional dependencies
        :returns tau_params:
            range of time lags
    """

    var_params = np.arange(2, 26)

    coeff_params = np.concatenate((np.arange(-0.1, -1.0, -0.2),\
                                        np.arange(0.1, 1.0, 0.2)))
    tau_params = np.array([1,2]) 

    sample_params = np.arange(20, 101, 20)

    func_params = [lambda x : x, 
                   lambda x : (1.0-4.0*np.e**(-x**2.0/2.0))*x, 
                   lambda x : (1.0-4.0*x**3.0 * np.e**(-x**2.0/2.0))*x]
    
    oracle_params = [indep_tests.PartialCorrelation(), 
                     indep_tests.GaussianProcessRegression(), 
                     indep_tests.GaussianProcessRegression()]

    return var_params, coeff_params, tau_params, sample_params, func_params, oracle_params

def get_random_config(d : int, coeff_params : np.ndarray, tau_params : np.ndarray) -> dict:
    effects = set(range(d))
    arcs = list(itertools.permutations(effects, 2))

    config = dict({effect: [((effect, -1), np.random.choice(coeff_params))] for effect in effects})

    max_links = d if 2 < d else 1

    n_of_instant = random.randint(0, max_links)
    n_of_lagged = random.randint(0, max_links)

    idxs = random.sample(range(len(arcs)), k = n_of_instant+n_of_lagged)

    instant_idxs, lagged_idxs = idxs[:n_of_instant], idxs[n_of_instant:]
    # print(d, instant_idxs, lagged_idxs)
    
    for instant_idx in instant_idxs:
        (cause, effect) = arcs[instant_idx]
        config[effect].append(((cause, 0), np.random.choice(coeff_params)))

    for lagged_idx in lagged_idxs:
        (cause, effect) = arcs[lagged_idx]
        config[effect].append(((cause, -np.random.choice(tau_params)), np.random.choice(coeff_params)))

    return config

    