import numpy as np
from tigramite.pcmci import PCMCI
from dscm import DSCM
from tigramite import data_processing as pp
from indep_tests import Oracle, PartialCorrelation
from graph_reps import cpdag_from_dag, equivalence_class_from_cpdag, get_graph_index, get_structural_hamming_distance

#TODO: throw out tau_max from the scoring function

def run_pmcmi(obs_data : np.array, tau_max: int, oracle : Oracle, 
                                        method : str = "pcmciplus") -> np.array:
    _, d = obs_data.shape

    pcmci = PCMCI(dataframe = pp.DataFrame(obs_data, var_names=[f"$X_{i}$" for i in range(d)]),
                                                        cond_ind_test = oracle, verbosity = 0)
    if method == "pcmci":
        results = pcmci.run_pcmci(tau_max = tau_max)
    elif method == "pcmciplus":
        results = pcmci.run_pcmciplus(tau_max = tau_max)

    raise NotImplementedError()

def get_score_matrix(obs_data: np.array, edges : list, tau_max: int, oracle: Oracle, 
                                                        method : str = "pcmciplus") -> np.array:
    """ Performs conditional independence tesst on selected bivariates X^{t-tau}_i, X^t_j for each lag tau and 
        returns a scoring matrix composed of p-values for the hypothesis that X^{t-tau} causes X^t_j for some tau.

            :param obs_data:
                observational data matrix of shape T,d
            :param edges:
                list of bivariates (X_i, X_j) to be scored
            :param tau_max:
                maximal lag at which X_i can influence X_j
            :param oracle:
                conditional independence oracle
            :param method:
                method for estimating p-values

            :returns score_matrix:
                matrix of scores where each score_matrix[i,j] holds the score for X_i -> X_j
    """

    _, d = obs_data.shape
    p_matrix = np.ones((d,d))

    pcmci = PCMCI(dataframe = pp.DataFrame(obs_data, var_names=[f"$X_{i}$" for i in range(d)]),
                                                        cond_ind_test = oracle, verbosity = 0)
    if method == "pcmci":
        results = pcmci.run_pcmci()
    elif method == "pcmciplus":
        results = pcmci.run_pcmciplus()
    else:
        raise ValueError("Input method with name \"{method}\" not available!")

    p_matrices = results["p_matrix"]

    for (i,j) in edges:
        p_matrix[i, j] = oracle.get_combined_pvalues(p_matrices[i, j, :], \
                                                        method = "harmonic")
    score_matrix = np.subtract(1, p_matrix)

    return score_matrix

def score_equivalence_class(obs_data: np.array, edges : list, equivalence_class : np.array, 
                                                        tau_max: int, oracle: Oracle, method : str) -> np.array:
    """ Scores each member of the input equivalence class using observational data and independence oracle.

            :param obs_data: 
                observational data matrix of shape T,d
            :param edges:
                list of edges to be scored
            :param equivalence class: 
                matrix representation of MEC
            :param tau_max:
                maximal time lag to be evaluated
            :param oracle: 
                conditional independence oracle
            :param method:
                method for computing p-values    
            
            :returns scores: 
                vector of scores on each DAG

    """

    scores = np.zeros(len(equivalence_class))

    score_matrix = get_score_matrix(obs_data = obs_data, edges = edges, \
                                tau_max = tau_max, oracle = oracle, method = method)

    for idx, member in enumerate(equivalence_class):
        ### compute p-value that X_i \-> X_j for all i,j ###
        scored_member = np.multiply(member, score_matrix)
        scores[idx] = oracle.get_combined_pvalues(pvalues = scored_member[np.nonzero(scored_member)].flatten(), 
                                                                                            method = "harmonic")
    return scores

def get_optimal_graph_index(scores: np.array) -> tuple:
    """ Obtain graph that optimises scoring function.

            :param equivalence class: 
                matrix representation of MEC 

            :param scores: 
                vector of scores for members of MEC

            :returns opt_score_idx: 
                index of score optimising graph

            :returns ranking:
                score-based ranking of graphs 
    """

    # opt_score = np.amax(scores)
    opt_score = np.amax(scores)
    opt_score_idxs = np.argwhere(scores == opt_score).flatten()
    opt_score_idx = np.random.choice(opt_score_idxs)
    ranking = np.argsort(scores)
    
    return opt_score_idx, ranking

if __name__ == "__main__":

    links = {
        0: [((0, -1), 0.7)],
        1: [((1, -1), 0.7), ((0, 0), 0.4)],
        2: [((2, -1), 0.5), ((1, 0), 0.2)],
        3: [((3, -1), 0.5), ((2, 0), 0.5)],
        4: [((4, -1), 0.5), ((1, 0), 0.5)],
        # 5: [((5, -1), 0.5), ((1, -1), 0.2)],
        # 6: [((4, -1), 0.5), ((6, -1), 0.2)],
    }

    f = lambda x : x
    dscm = DSCM(links, f)
    obs_data = dscm.generate_obs_data(T=100, log_transform = True)
    adj_matrices = dscm.get_adjacency_matrices()
    summary_graph = dscm.get_summary_graph(adj_matrices=adj_matrices)
    _, cpdag_repr = cpdag_from_dag(summary_graph)
    edges = list(zip(*np.where(cpdag_repr == 1)))

    _, equivalence_class = equivalence_class_from_cpdag(cpdag_repr=cpdag_repr)

    oracle = PartialCorrelation()
    tau_max = len(adj_matrices) - 1

    scores = score_equivalence_class(obs_data = obs_data, edges = edges, equivalence_class = equivalence_class, \
                                                                tau_max = tau_max, oracle = oracle, method = "pcmciplus")
    
    opt_idx, *_ = get_optimal_graph_index(scores = scores)
    opt_graph = equivalence_class[opt_idx]

    print(equivalence_class)

    print(scores)

    print(summary_graph)

    print(opt_graph)

    true_idx = get_graph_index(equivalence_class, summary_graph)

    print(true_idx)
    
    print(opt_idx)

    print(np.isclose(scores[true_idx], scores[opt_idx]))

    print(get_structural_hamming_distance(summary_graph, opt_graph))