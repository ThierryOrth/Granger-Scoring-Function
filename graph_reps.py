import numpy as np
import causaldag as cd

def get_summary_graph(amats : np.array = None, dag_repr : bool = True) -> np.array:
    """ Constructs summary graph from adjacency matrices, defined as summary_graph[i,j] = 1 iff
        amats[i,j,tau] = 1 for some tau. 

            :param amats: 
                adjacency matrices representing contemporaneous and lagged causes of DSCM
            :param dag_repr:
                Boolean to convert summary graph into DAG representation
            
            :returns summary_graph:
                summary graph induced by adjacency matrices
    """

    *_ , d = amats.shape
    summary_graph = np.zeros((d,d)).astype(int)

    for i in range(d):
        for j in range(d):
            summary_graph[i,j] = 1 if 0 < np.sum(amats[:, i, j]) else 0

    if dag_repr:
        np.fill_diagonal(summary_graph, 0)

    return summary_graph

def cpdag_from_dag(dag_repr: np.array) -> tuple:
    """ Constructs CPDAG representing the MEC of the input DAG.

            :param dag_repr: 
                DAG object or matrix representation of DAG

            :returns cpdag: 
                PDAG object representation of CPDAG
            :returns amat: 
                adjacency matrix representation of CPDAG
            :returns vstructs: 
                list of v-structures in CPDAG

        """

    dag = cd.DAG.from_amat(amat = dag_repr) \
        if isinstance(dag_repr, np.ndarray) else dag_repr

    cpdag = dag.cpdag()
    amat, _ = cpdag.to_amat()
    vstructs = cpdag.vstructs()

    return cpdag, amat, vstructs

def equivalence_class_from_cpdag(cpdag_repr: np.array) -> tuple:
    """ Construct the MEC as represented by the input CPDAG. 

        :param cpdag_repr: 
            PDAG object or matrix representation of CPDAG
        
        :returns equivalence class:
            PDAG object representation of MEC
        :returns amats:
            matrix representation of MEC
        :returns vstructs: 
            list of v-structures in MEC

    """

    cpdag = cd.PDAG.from_amat(amat = cpdag_repr) \
        if isinstance(cpdag_repr, np.ndarray) else cpdag_repr

    vertices = cpdag.nodes
    vstructs = cpdag.vstructs()
    markov_equivalence_class = cpdag.all_dags()

    amats = np.array([cd.DAG(vertices, dag).to_amat()[0]
                            for dag in markov_equivalence_class])

    return markov_equivalence_class, amats, vstructs

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

def is_cyclic(graph_repr: np.array) -> bool:
    """ Check cyclicity of graph by checking for non-zero entries 
        in the graph matrix diagonal.

            :param graph_repr: 
                matrix representation of graph

            :returns is_cyclic: 
                Boolean indicating if input graph is cyclic
    """
    is_cyclic = False

    try:
        _ = cd.DAG.from_amat(amat = graph_repr)
    except:
        is_cyclic = True

    return is_cyclic

