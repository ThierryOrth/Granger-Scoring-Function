import os, pickle
import numpy as np
import pandas as pd

def save_results(results, savepath : str) -> None:
    """ Saves results of experiment.

            :param results: 
                matrix of scores per iteration of experiment

            :param filename: 
                filename of destination
 
    """

    if os.path.exists(savepath):
        return

    if isinstance(results, np.ndarray):
        np.save(savepath, results)
    
    if isinstance(results, dict):
        with open(savepath, 'wb') as file: 
            pickle.dump(results, file)

def get_results(filename : str) -> np.array:
    """ Loads score arrays from file.

            :param filename: 
                name of file to be loaded

            :returns results: 
                matrix with scores per iteration of experiment
    
    """

    folder = "results"
    rel_path = f"{folder}\{filename}"

    if not os.path.exists(rel_path):
        raise NotADirectoryError(f"Warning: could not find file in directory \"{rel_path}\"!")

    results = np.load(rel_path)
        
    return results

def save_config(config : dict, filename : str, folder : str = "configs"):
    """ Pickles a specific parameter configuration.
    
            :param config: 
                parameter configuration to be saved

            :param filename: 
                filename of destination
    """

    rel_path = f"{folder}\{filename}"

    if not os.path.exists(folder):
        os.makedirs(folder)

    with open(rel_path, "wb") as config_file:
        pickle.dump(config, config_file) 

def get_config(filename : str, folder_name : str = "configs") -> dict:
    """ Loads configuration from file. 

            :param filename: 
                name of file to be loaded
    """
    rel_path = f"{folder_name}\{filename}"

    if not os.path.exists(rel_path):
        raise NotADirectoryError(f"Warning: could not find file in directory \"{rel_path}\"!")

    with open(rel_path, "rb") as config_file:
        config = pickle.load(config_file)
        return config

def gather_data(experiment: str, var_params : np.ndarray, sample_params : np.ndarray, n_of_iter : int)-> tuple:
    dim_matr = np.zeros((5, 4, len(var_params), n_of_iter), dtype=float)
    sample_matr = np.zeros((5, 4, len(var_params), n_of_iter, len(sample_params)), dtype=float)
    inst_prop_matr = np.zeros((5, 4, len(var_params), n_of_iter), dtype=float)
    lag_prop_matr = np.zeros((5, 4, len(var_params), n_of_iter), dtype=float)
    equiv_class_data = np.zeros((len(var_params), n_of_iter, 3), dtype = float)

    for d in var_params:
        var_idx = d - 2

        for iteration in range(n_of_iter):
            config_path = f"configs_{experiment}\\{d}-{iteration}.pickle"
            results_path = f"results_{experiment}\\{d}-{iteration}.pickle"

            if not os.path.exists(config_path):
                continue
            if not os.path.exists(results_path):
                continue

            with open(config_path, "rb") as handle:
                config = pickle.load(handle)
                effects = config.keys()
                n_of_lagged = sum([sum([1 for (_, lag), _ in config[effect] if lag < 0]) for effect in effects])
                n_of_instant = sum([sum([1 for (_, lag), _ in config[effect] if lag == 0]) for effect in effects])   

                prop_instant = n_of_instant / (n_of_lagged + n_of_instant)
                prop_lagged = n_of_lagged / (n_of_lagged + n_of_instant)

            with open(results_path, "rb") as handle:
                datapoint = pickle.load(handle)
                n_of_methods = 4 if experiment == "with_access" else 5
                _, n_of_scores = list(datapoint.values())[0].shape

                for method_idx in range(n_of_methods):
                    for score_idx in range(n_of_scores):
                        results = datapoint[method_idx][:, score_idx]
                    
                        dim_matr[method_idx, score_idx, var_idx, iteration] = np.mean(datapoint[method_idx][:, score_idx])
                        inst_prop_matr[method_idx, score_idx, var_idx, iteration] = prop_instant
                        lag_prop_matr[method_idx, score_idx, var_idx, iteration] = prop_lagged

                        for sample_idx in range(len(results)):
                            sample_matr[method_idx, score_idx, var_idx, iteration, sample_idx] = results[sample_idx]

                n_of_members, n_of_arcs, n_of_edges = datapoint[5], datapoint[6], datapoint[7]/2

                equiv_class_data[var_idx, iteration, 0] = n_of_members
                equiv_class_data[var_idx, iteration, 1] = n_of_edges
                equiv_class_data[var_idx, iteration, 2] = 0 if n_of_edges==0 else n_of_edges/(n_of_edges+n_of_arcs) 

    return dim_matr, sample_matr, inst_prop_matr, lag_prop_matr, equiv_class_data

def write_data_to_table(data : pd.DataFrame, filename : str):
    with open(f"{filename}.tex", "w") as file:
        file.write(
             pd.DataFrame(data = data).to_latex(index=False)
        )