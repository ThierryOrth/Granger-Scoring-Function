import copy, cdt
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, mean_squared_error

def get_structural_hamming_distance(true_graph: np.array, est_graph: np.array) -> int:
    """ Computes Structural Hamming Distance (SHD) between input graphs. """
    
    shd = cdt.metrics.SHD(target = true_graph, pred = est_graph)

    return shd

def get_frobenius_norm(true_graph : np.array, est_graph : np.array) -> float:
    """ Computes difference between adjacency matrices of true and estimated graph using Frobenius norm. """

    frob_norm = np.linalg.norm(x = (true_graph - est_graph), ord = "fro")

    return frob_norm

def get_metrics(true_graph : np.array, est_graph : np.array) -> tuple:
    """ Computes accuracy scores for estimated graph. """

    T, T_hat = true_graph.flatten(), est_graph.flatten()
    
    accuracy = accuracy_score(y_true = T, y_pred = T_hat)
    (precision, _), (recall, _), (f1, _), _ = precision_recall_fscore_support(y_true = T, y_pred = T_hat, beta = 1.0, zero_division = 0)
    mse = mean_squared_error(y_true = T, y_pred = T_hat)

    return accuracy, precision, recall, f1, mse

def get_prop_correct(true_graph : np.array, est_graph : np.array, edges : list) -> int:
    """ Computes number of correctly estimated edges. """

    n, *_ = true_graph.shape

    true_abs_entries = true_graph.flatten()
    est_abs_entries = est_graph.flatten()

    abs_prop = np.sum(true_abs_entries == est_abs_entries)/len(est_abs_entries)

    mask = np.zeros((n,n))
    mask[tuple(zip(*edges))] = True

    true_rel_entries = true_graph[np.where(mask==True)]
    est_rel_entries = est_graph[np.where(mask==True)]

    rel_prop = np.sum(true_rel_entries==est_rel_entries)/len(est_rel_entries)

    return abs_prop, rel_prop

def get_all_scores(true_graph : np.array, est_graph : np.array, edges : list) -> np.array:
    """ Gathers all scores. """

    shd = get_structural_hamming_distance(true_graph, est_graph)
    frob_norm = get_frobenius_norm(true_graph, est_graph)
    abs_prop, rel_prop = get_prop_correct(true_graph, est_graph, edges)

    scores = [shd, frob_norm, abs_prop, rel_prop]
                      
    return scores
