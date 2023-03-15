import numpy as np
from independence_tests import Oracle
from scoring_function import score_equivalence_class, get_best_candidate, structural_hamming_distance
from graph_representations import dag_to_cpdag

### IN PROGRESS ###

# TODO: write function for random configurations

def run_experiment(T : int, links : dict, map : function, oracle : Oracle, score_precision: int = 3, instant_effects:bool=False, lag_specific:bool=False, N:int=100) -> float:
    """
    
    """
    distances = np.zeros(N)

    for i in range(N):
        dscm = DSCM(links=links)
        obs_data = dscm.generate_obs_data(T = T)

        lag_matrix, instant_matrix = dscm.get_adjacency_matrices()
        _, cpdag_repr, _ = dag_to_cpdag(instant_matrix) 

        T, d = obs_data.shape
        _, n = lag_matrix.shape
        max_lag = (n/d)-1

        equivalence_class, scores = score_equivalence_class(obs_data = obs_data, cpdag_repr=cpdag_repr, \
                                                max_lag = max_lag, true_graph = instant_matrix, oracle = oracle,\
                                                    score_precision=score_precision, instant_effects = instant_effects, lag_specific=lag_specific) 

        candidate_graph = get_best_candidate(equivalence_class=equivalence_class, scores=scores)

        distances[i] = structural_hamming_distance(true_graph = instant_matrix, candidate_graph = candidate_graph)

    average_distance = np.sum(distances)/len(distances)
    
    return average_distance

def get_random_configurations(time_params : np.array, coeff_params : np.array, var_params : np.array, density_params : np.array):
    pass

# TODO: assign p-value matrices only for those cells in the PCDAG where m[i,j]==1

# TODO: construct p-value matrix for all (V_i, V_j) s.t. m[i,j]==1

# TODO: for different densities, simply build up the p-value matrix from what we have before

# TODO: for a different number of variables, we use a new p-value matrix

# TODO: save data in folders to save time

if __name__ == "__main__":

    # randomise tau
    
    N = 100
    time_params = np.arange(10, N+1, 10)
    coeff_params = np.concatenate((np.arange(-0.1, -1.0, -0.1),\
                                            np.arange(0.1,1,0.1)))
    var_params = np.arange(3, 20)
    density_params = np.array([[np.arange(1, n+1, 1)] for n in var_params])