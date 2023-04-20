import itertools
import iteration_utilities
import numpy as np
from data_recs import save_config
from indep_tests import PartialCorrelation, GaussianProcessRegression

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

    var_params = np.arange(3, 13)

    coeff_params = np.concatenate((np.arange(-0.1, -1.0, -0.2),\
                                        np.arange(0.1, 1.0, 0.2)))
    tau_params = np.array([1,2]) 

    sample_params = np.arange(20, 101, 20)

    func_params = [lambda x : x, lambda x : (1.0-4.0*np.e**(-x**2.0/2.0))*x, 
                        lambda x : (1.0-4.0*x**3.0 * np.e**(-x**2.0/2.0))*x]
    
    oracle_params = [PartialCorrelation(), GaussianProcessRegression(), GaussianProcessRegression()]

    return var_params, coeff_params, tau_params, sample_params, func_params, oracle_params

def save_var_configs(var_params : np.array) -> dict:
    """
            :param var_params:
                range with model dimensionality parameters
            :returns valid_param_configs:
    """
    for n_of_var in var_params:
        n_of_effects = 1 if n_of_var == 2 else int(np.floor(1.5*n_of_var)) 
        for n_of_instant in range(n_of_effects):
            for n_of_lagged in range(n_of_effects):
                config = dict({"n_of_var" : n_of_var, "n_of_instant" : n_of_instant, 
                                                        "n_of_lagged" : n_of_lagged})
                save_config(config = config, filename = f"{n_of_var}-{n_of_instant}-{n_of_lagged}", folder_name = "configs")

def get_random_config(var_param_config : dict, coeff_params : np.array, tau_params : np.array, 
                                                                        random_state : int = 42) -> dict:
    """
            :param param_config:
                range with model dimensionality parameters
            :param coeff_params:
                range with coefficients defining causal influence 
            :param tau_params:
                range of time lags
            :param random_state:
                seed for reproducibility

            :returns configuration:
                random configuration retrieved from input parameters
    """

    np.random.seed(seed = random_state)

    n_of_var, n_of_instant, n_of_lagged = var_param_config.values() 
    effects = set(range(n_of_var))
    configuration = dict({effect : [((effect, -1), np.random.choice(coeff_params))] for effect in effects})
    arcs = list(itertools.permutations(effects, 2))

    idxs = np.arange(0, len(arcs), 1)

    instant_idxs = np.random.choice(idxs, size = n_of_instant)
    lagged_idxs = np.random.choice(idxs, size = n_of_lagged)

    for instant_idx in instant_idxs:
        lagged_cause, lagged_effect = arcs[instant_idx]
        configuration[lagged_effect].append(((lagged_cause, 0), np.random.choice(coeff_params)))

    for lagged_idx in lagged_idxs:
        lagged_cause, lagged_effect = arcs[lagged_idx]
        configuration[lagged_effect].append(((lagged_cause, -np.random.choice(tau_params)), np.random.choice(coeff_params)))

    return configuration

if __name__ == "__main__":
    var_params, coeff_params, tau_params, sample_params, func_params, oracle_params = get_params()
    
