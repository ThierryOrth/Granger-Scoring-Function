import dscm, indep_tests, graph_reps
import numpy as np
from tigramite.pcmci import PCMCI
from tigramite import data_processing as pp

def run_pcmci(obs_data : np.array, oracle : indep_tests.Oracle, method : str = "pcmciplus", pc_alpha : float = None, tau_max : int = 5) -> tuple:
    """ Runs PCMCI algorithm for retrieving p-value matrices and estimated graph.

            :param obs_data:
                observational data matrix with obs_data[t,i] corresponding 
                to the observation of the i'th variable at time t
            :param oracle:
                conditional independence oracle
            :param method:
                method for computing p-values  
            :param v_structs:
                background knowledge for PCMCI algorithm
            :param alpha_level:
                significance threshold for selected PCMCI algorithm
            
            :returns p_matrices:
                matrices with p-values for causal connections with p_matrices[tau,i,j] 
                corresponding to the p-value for H_0 : X_i /-> X_j at lag tau
            :returns pcmci_graph:
                graph predicted using selected PCMCI algorithm

    """
    
    _, d = obs_data.shape
    axes = (2,0,1)
    pcmci = PCMCI(dataframe = pp.DataFrame(obs_data, var_names=[f"$X_{i}$" for i in range(d)]),
                                                        cond_ind_test = oracle, verbosity = 0)
    if method == "pcmci":
        results = pcmci.run_pcmci(pc_alpha = pc_alpha, tau_max = tau_max)
    elif method == "pcmciplus":
        results = pcmci.run_pcmciplus(pc_alpha = pc_alpha, tau_max = tau_max)
    else:
        raise ValueError("Warning: input method with name \"{method}\" not available!")
    
    p_matrices = np.transpose(a = results["p_matrix"], axes = axes)
    pcmci_window_graph = np.transpose(a = np.where(results["graph"]=="", 0, 1).astype(int), axes=axes)

    pcmci_summary_graph = graph_reps.get_summary_graph(pcmci_window_graph)

    return p_matrices, pcmci_summary_graph

def get_score_matrix(p_matrices : np.array, arcs : list, oracle: indep_tests.Oracle, pvalue_method : str = "harmonic") -> np.ndarray:
    """ Performs conditional independence tests on selected bivariates X^{t-tau}_i, X^t_j for each lag tau and 
        returns a scoring matrix composed of p-values for the hypothesis that X^{t-tau} causes X^t_j for some tau.

            :param p_matrices:
                matrices with p-values for causal connections with p_matrices[tau,i,j] 
                corresponding to the p-value for H_0 : X_i /-> X_j at lag tau
            :param arcs:
                list of bivariates (X_i, X_j) to be scored
            :param oracle:
                conditional independence oracle
            :param method:
                method for estimating p-values

            :returns score_matrix:
                matrix of scores where each score_matrix[i,j] holds the score for X_i -> X_j
    """
    *_, d = p_matrices.shape
    p_matrix = np.ones((d,d))

    for (i,j) in arcs:
        p_matrix[i, j] = oracle.get_combined_pvalues(p_matrices[:, i, j], \
                                                        method = pvalue_method)
    score_matrix = np.subtract(1, p_matrix)

    return score_matrix

def score_equivalence_class(equiv_class : np.ndarray, arcs : list, p_matrices : np.ndarray, oracle : indep_tests.Oracle, pvalue_method : str = "harmonic") -> np.ndarray:
    """ Scores each member of the input equivalence class using observational data and independence oracle.

            :param obs_data: 
                observational data matrix of shape T,d
            :param arcs:
                list of arcs to be scored
            :param equivalence class: 
                matrix representation of MEC
            :param oracle: 
                conditional independence oracle
            :param method:
                method for computing p-values  
            :param return_pcmci_graph:
            
            :returns scores: 
                vector of scores on each DAG

    """

    
    score_matrix = get_score_matrix(p_matrices = p_matrices, arcs = arcs, 
                                        oracle = oracle, pvalue_method = pvalue_method)

    scores = np.zeros(len(equiv_class))

    for idx, member in enumerate(equiv_class):
        scored_member = np.multiply(member, score_matrix)
        pvalues = scored_member[np.nonzero(scored_member)].flatten()
        scores[idx] = oracle.get_combined_pvalues(pvalues, method = "harmonic") if pvalues != [] else 0.0
    
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

    opt_score = np.amax(scores)
    opt_score_idxs = np.argwhere(scores == opt_score).flatten()
    opt_score_idx = np.random.choice(opt_score_idxs)
    ranking = np.argsort(scores)
    
    return opt_score_idx, ranking

def estimate_graph(equiv_class: np.ndarray, arcs : list, p_matrices : np.ndarray, oracle: indep_tests.Oracle,
                   method: str = "pcmciplus", true_graph_equiv: bool = True, get_highest : bool = False) -> tuple:
    """ Estimate graph from observational data.

            :param obs_data:
                matrix of observational data for time series
            :param arcs:
                list of arcs
    """

    scores = score_equivalence_class(equiv_class  = equiv_class, arcs = arcs, 
                                     p_matrices = p_matrices, oracle = oracle)

    phi_graph_idx, *_ = get_optimal_graph_index(scores = scores)

    phi_graph = equiv_class[phi_graph_idx]

    return phi_graph

