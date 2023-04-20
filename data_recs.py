import os
import pickle
import numpy as np
from time import time

# array consisting of scores for each iteration: precision, recall, shd
# final array consisting of average scores 

def save_results(results, filename : str, folder : str) -> None:
    """ Saves results of experiment.

            :param results: 
                matrix of scores per iteration of experiment

            :param filename: 
                filename of destination
 
    """

    rel_path = f"{folder}\\{filename}"

    if os.path.exists(rel_path):
        return

    if isinstance(results, np.ndarray):
        np.save(rel_path, results)
    
    if isinstance(results, dict):
        with open(f"{rel_path}.pickle", 'wb') as file:
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

def save_config(config : dict, filename : str, folder_name : str = "configs"):
    """ Pickles a specific parameter configuration.
    
            :param config: 
                parameter configuration to be saved

            :param filename: 
                filename of destination
    """

    rel_path = f"{folder_name}\{filename}"

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    if not os.path.exists(rel_path):
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
