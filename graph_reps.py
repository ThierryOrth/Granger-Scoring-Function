import numpy as np
import causaldag as cd
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Overview article of evaluation methods in causal discovery: https://arxiv.org/pdf/2202.02896.pdf.

def cpdag_from_dag(dag_repr: np.array) -> tuple:
    """ Constructs CPDAG representing the MEC of the input DAG.

            :param dag_repr: 
                DAG object or matrix representation of DAG

            :returns cpdag: 
                PDAG object representation of CPDAG
            :returns adj_matrix: 
                adjacency matrix representation of CPDAG
            :returns node_list: 
                list of nodes of CPDAG

        """

    dag = cd.DAG.from_amat(dag_repr) \
        if isinstance(dag_repr, np.ndarray) else dag_repr

    cpdag = dag.cpdag()
    adj_matrix, _ = cpdag.to_amat()

    return cpdag, adj_matrix

def equivalence_class_from_cpdag(cpdag_repr: np.array) -> tuple:
    """ Construct the MEC as represented by the input CPDAG. 

        :param cpdag_repr: 
            PDAG object or matrix representation of CPDAG
        
        :returns equivalence class:
            PDAG object representation of MEC
        :returns adj_matrices:
            matrix representation of MEC

    """

    cpdag = cd.PDAG.from_amat(cpdag_repr) \
        if isinstance(cpdag_repr, np.ndarray) \
        else cpdag_repr

    vertices = cpdag.nodes
    markov_equivalence_class = cpdag.all_dags()

    adj_matrices = np.array([cd.DAG(vertices, dag).to_amat()[0]
                            for dag in markov_equivalence_class])

    return markov_equivalence_class, adj_matrices


def get_graph_index(graphs: np.array, graph: np.array) -> int:
    """ Finds index of input graph in collection of graphs.

            :param graphs: 
                matrix representation of set of graphs
            :param graph: 
                matrix representation of relevant graph

            :returns graph_idx: 
                index of input graph in set of graphs

    """

    N = len(graphs)

    for graph_idx in range(N):
        if(graphs[graph_idx] == graph).all():
            return graph_idx

    return None


def is_identical_graph(first_graph: np.array, second_graph: np.array) -> bool:
    """ Structural equality comparison of input graphs.

            :param first_graph, second_graph: 
                matrix representation of relevant graphs

            :returns is_equal: 
                Boolean indicating if input graphs are structurally equal
    """

    is_equal = (first_graph == second_graph).all()

    return is_equal

def is_cyclic(graph_repr: np.array):
    """ Check cyclicity of graph by checking for non-zero entries 
        in the graph matrix diagonal.

            :param graph_repr: 
                matrix representation of graph

            :returns zero_diagonal: 
                Boolean indicating if input graph is cyclic
    """
    diagonal = np.diagonal(graph_repr)
    diagonal_sum = np.sum(diagonal)

    # TODO: cyclicity could occur in other ways as well

    # TODO: turn into DAG and then check if adj matrix is the same

    return False if diagonal_sum == 0 else True

def get_mse_scores(true_graph : np.array, optimal_graph : np.array) -> float:
    """ Computes MSE between true and estimated graph.

            :param true_graph:
                matrix representation of true graph
            :param optimal_graph:
                matrix representation of optimal graph
            
            :returns mse:
                mean squared error between graphs

    """

    T, T_hat = true_graph.flatten(), optimal_graph.flatten()
    mse = 1/len(T)*np.sum([(t-t_hat)**2 for t, t_hat in zip(T, T_hat)])

    return mse

def get_structural_hamming_distance(true_graph: np.array, optimal_graph: np.array) -> int:
    """ Computes Structural Hamming Distance (SHD) between input graphs.

            :param true_graph: 
                matrix representation of true graph
            :param optimal_graph:
                matrix representation of optimal graph

            :returns shd: 
                SHD distance between input graphs
    """

    shd = cd.DAG.from_amat(true_graph).shd(
            cd.DAG.from_amat(optimal_graph)
            )

    return shd

def get_accuracy_scores(true_graph : np.array, optimal_graph : np.array) -> float:
    """ Computes accuracy score of estimated graph.
    
            :param true_graph:
                matrix representation of true graph
            :param optimal_graph:
                matrix representation of optimal graph

            :returns accuracy:
                accuracy score of optimal graph
    """

    T, T_hat = true_graph.flatten(), optimal_graph.flatten()
    
    accuracy = accuracy_score(y_true = T, y_pred = T_hat)

    return accuracy

def get_precision_recall_scores(true_graph : np.array, optimal_graph : np.array) -> float:
    """ Computes precision and recall scores of estimated graph.
    
            :param true_graph:
                matrix representation of true graph
            :param optimal_graph:
                matrix representation of optimal graph

            :returns precision:
                precision score of optimal graph
            :returns recall:
                recall score of optimal graph
    """
    T, T_hat = true_graph.flatten(), optimal_graph.flatten()

    precision = precision_score(y_true = T, y_pred = T_hat)
    recall = recall_score(y_true = T, y_pred = T_hat)

    return precision, recall