import os
import pickle
import numpy as np
from time import time

# array consisting of scores for each iteration: precision, recall, shd
# final array consisting of average scores 

def save_results(results : np.array, filename : str) -> None:
    """ Saves results of experiment.

            :param results: 
                matrix of scores per iteration of experiment

            :param filename: 
                filename of destination
 
    """
    folder = "results"
    rel_path = f"{folder}\{filename}"

    if not os.path.exists(folder):
        os.makedirs(folder)

    np.save(rel_path, results)

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

def save_config(config : dict, filename : str):
    """ Pickles a specific parameter configuration.
    
            :param config: 
                parameter configuration to be saved

            :param filename: 
                filename of destination
    """

    folder = "configs"
    rel_path = f"{folder}\{filename}"

    if not os.path.exists(folder):
        os.makedirs(folder)

    with open(rel_path, "wb") as config_file:
        pickle.dump(config, config_file)

def get_config(filename : str) -> dict:
    """ Loads configuration from file. 

            :param filename: 
                name of file to be loaded
    """
    folder = "configs"
    rel_path = f"{folder}\{filename}"

    if not os.path.exists(rel_path):
        raise NotADirectoryError(f"Warning: could not find file in directory \"{rel_path}\"!")

    with open(rel_path, "rb") as config_file:
        config = pickle.load(config_file)
        print(config)
        return config

def record_time(func):
    """ Wrapper for recording runtime of wrapped function. """
    def wrapper(*args, **kwargs):
        start = time()
        result = func(*args, **kwargs)
        end = time()

        print(f"Duration of {func.__name__!r}: {(end-start):.4f}s.")
    return wrapper()
