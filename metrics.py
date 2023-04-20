import copy
import cdt
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

def get_n_of_correct(true_graph : np.array, est_graph : np.array, edges : list) -> int:
    """ Computes number of correctly estimated edges. """
    est_edges = copy.deepcopy(est_graph).astype(object)

    est_edges[tuple(zip(*edges))] = [None] * len(edges)

    correct_entries = np.where(est_edges == true_graph)
    est_entries = np.where(est_edges != None)

    return len(correct_entries)/len(est_entries)

def get_all_scores(true_graph : np.array, est_graph : np.array, edges : list = None) -> np.array:
    """ Gathers all scores. """

    shd = get_structural_hamming_distance(true_graph, est_graph)
    frob_norm = get_frobenius_norm(true_graph, est_graph)
    n_of_correct = get_n_of_correct(true_graph, est_graph, edges)
    acc, prec, rec, f1, mse = get_metrics(true_graph, est_graph)

    scores = [shd, frob_norm, n_of_correct]
                      
    return scores
