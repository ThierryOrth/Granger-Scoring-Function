import numpy as np
import causaldag as cd

def dag_to_cpdag(dag_repr : np.array) -> tuple:
    """ Retrieve the CPDAG representing the Markov Equivalence Class
        of the input DAG.

            :param dag_repr: DAG, represented as CausalDAG or array object

            :returns cpdag: CPDAG, represented as CausalDAG object

            :returns adj_matrix: CPDAG, represented as adjacency matrix

            :returns node_list: node list associated with CPDAG
            
        """
    
    dag = cd.DAG.from_amat(dag_repr) \
          if isinstance(dag_repr, np.ndarray) else dag_repr
    
    cpdag = dag.cpdag()
    adj_matrix, node_list = cpdag.to_amat()
    
    return cpdag, adj_matrix, node_list

def cpdag_to_markov_equivalence_class(cpdag_repr: np.array, k : int = 0) -> tuple:
    """  Retrieve the Markov Equivalence Class represented by the input CPDAG
    
        :param cpdag_repr: PCDAG, represented as CausalDAG or array object

        :returns markov_equivalence_class: MEC represented as CausalDAG objects

        :returns adj_matrices: MEC represented as array of adjacency matrices

    """

    cpdag = cd.PDAG.from_amat(cpdag_repr) \
                        if isinstance(cpdag_repr, np.ndarray) \
                                                else cpdag_repr

    vertices = cpdag.nodes
    markov_equivalence_class = cpdag.all_dags()
   
    adj_matrices = np.array([cd.DAG(vertices, dag).to_amat()[0] \
                            for dag in markov_equivalence_class])

    return markov_equivalence_class, adj_matrices

if __name__ == "__main__":
    A = np.array([[0, 1, 1],
                  [0, 0, 0],
                  [0, 0, 0]])
    
    cpdag, adj_matrix, *_ = dag_to_cpdag(A)
    _, mec = cpdag_to_markov_equivalence_class(cpdag)
    print(mec)

    

