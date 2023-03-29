import itertools
import iteration_utilities
import numpy as np
import time
from data_recs import save_config, record_time

# TODO: improve and fix code

# TODO: functionality to get all parameter settings

# TODO: functionality to generate N configurations for fixed parameter N

#TODO: develop efficient method for generating configurations (perhaps: don't generate every graph with n connections; instead get a random one with n connections and then
#       do the experiment m times)

def get_params() -> tuple:
    """
        :returns var_params:

        :returns coeff_params:

        :returns func_params:

        :returns tau_params:
    """

    var_params = np.arange(3, 13)

    coeff_params = np.concatenate((np.arange(-0.1, -1.0, -0.1),\
                                        np.arange(0.1, 1.0, 0.1)))
    func_params = {lambda x : x, \
                        lambda x : (1-4*np.e**(-x**2/2))*x, \
                            lambda x : (1-4*x**3 * np.e**(-x**2/2))*x} 
    tau_params = np.array([1,2]) 

    return var_params, coeff_params, func_params, tau_params

def get_var_configs(var_params : np.array) -> dict:
    """
            :param var_params:

            :returns valid_param_configs:
    """
    valid_param_configs = dict()
    idx = 0

    for n_of_variables in var_params:
        n_of_effects = 1 if n_of_variables == 2 else int(np.floor(1.5*n_of_variables)) 
        for n_of_instant_effects in range(n_of_effects):
            for n_of_lagged_effects in range(n_of_effects):
                valid_param_configs[idx] = dict({"n_of_variables" : n_of_variables, 
                                                "n_of_instant_effects" : n_of_instant_effects, 
                                                "n_of_lagged_effects" : n_of_lagged_effects})
                idx += 1

    return valid_param_configs

def generate_random_config(param_config : dict, coeff_params : np.array, tau_params : np.array, 
                                                                        random_state : int = 42) -> dict:
    """
            :param param_config:

            :param coeff_params:

            :param tau_params:

            :param random_state:

            :returns configuration:
    """

    np.random.seed(seed = random_state)

    n_of_variables, n_of_instant_effects, n_of_lagged_effects = param_config.values() 
    effects = set(range(n_of_variables))
    configuration = dict({effect : [((effect, -1), np.random.choice(coeff_params))] for effect in effects})
    arcs = list(itertools.permutations(effects, 2))

    instant_effects = iteration_utilities.random_combination(arcs, n_of_instant_effects)
    lagged_effects = iteration_utilities.random_combination(arcs, n_of_lagged_effects)

    for (instant_cause, instant_effect) in instant_effects:
        configuration[instant_effect].append(((instant_cause, 0), np.random.choice(coeff_params)))

    for (lag_cause, lag_effect) in lagged_effects:
         configuration[lag_effect].append(((lag_cause, -np.random.choice(tau_params)), np.random.choice(coeff_params)))

    return configuration

def generate_random_configs(var_param_configs : dict, coeff_params : np.array, tau_params : np.array, 
                                            N: int, verbose : bool = True, random_state : int = 42) -> None:
    """
            :param var_param_configs:

            :param coeff_params:

            :param tau_params:

            :param N:

            :param verbose:

            :param random_state:

    """
    
    k = len(var_param_configs)

    for idx, _ in enumerate(var_param_configs):
        if verbose:
            print(f"Generating sample of size {N} for configuration with index {idx+1}/{len(var_param_configs)}...")
        for j in range(N):
            configuration = generate_random_config(param_config = var_param_configs[idx], 
                                                   coeff_params= coeff_params, tau_params = tau_params)
            save_config(config = configuration, filename = f"config-{i}-{j}")


if __name__ == "__main__":
    var_params, coeff_params, func_params, tau_params = get_params()
    var_param_configs = get_var_configs(var_params = var_params)

    N = 100
    k = len(var_param_configs)

    start = time.time()

    for i in range(k):
        print(f"Generating sample of size {N} for configuration {i+1}/{k}...")
        for j in range(N):
            configuration = generate_random_config(param_config = var_param_configs[i], 
                                                   coeff_params = coeff_params, tau_params = tau_params)
            save_config(config = configuration,  
                        filename = f"config-{i}-{j}-{var_param_configs[i]['n_of_variables']}-{var_param_configs[i]['n_of_instant_effects']}-{var_param_configs[i]['n_of_lagged_effects']}")
    end = time.time()

    print(end-start)




