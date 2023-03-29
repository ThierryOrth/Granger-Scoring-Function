import numpy as np
from dscm import DSCM
from time import time
from data_recs import record_time, save_results, get_results, save_config, get_config
from indep_tests import Oracle, PartialCorrelation, GaussianProcessRegression
from scoring_function import score_equivalence_class, get_optimal_graph_index
from graph_reps import cpdag_from_dag, equivalence_class_from_cpdag, \
    get_structural_hamming_distance, get_graph_index, \
    is_identical_graph, is_cyclic, \
    get_accuracy_scores, get_precision_recall_scores, get_mse_scores

### Runge's experimental description is found here: https://arxiv.org/pdf/2003.03685.pdf ###

### test 1: performance differences with PCMCI+ ###

### test 2: effect of increasing number of variables in the model ###

### test 3: effect of adding more lagged causes ###

### test 4: effect of adding more contemporaneous causes ###

### test 5: effect of varying sample size ###

## test 6: effect of dependencies ###


def run_experiment(obs_data: np.array, edges: list, equivalence_class: np.array, tau_max: int, oracle: Oracle,
                   method: str, constraint_based: bool = False) -> np.array:

    scores = score_equivalence_class(obs_data=obs_data, edges=edges, equivalence_class=equivalence_class,
                                     tau_max=tau_max, oracle=oracle, method=method)
    opt_idx, *_ = get_optimal_graph_index(scores=scores)

    # TODO: test constraint-based method for comparing scores

    if constraint_based:
        pass

    return opt_idx

# @record_time


def run_epoch(config: dict, f: callable, oracle: Oracle, T: np.array, method: str) -> dict:
    """
            :param config:
                configuration defining DSCM
            :param f:
                transformation applied to linear sum of causes
            :param oracle:
                conditional independence oracle
            :param T:
                sample size of time series
            :param N:
                number of repetitions of experiment
            :param method:
                method for computing p-values

            :returns performance_scores:
                ### ###
        """

    n_of_idxs = len(T)

    score_names = ["shd_scores", "accuracy_scores", "precision_scores",
                   "recall_scores", "mse_scores", "equality_scores"]

    performance_scores = dict({"config": config}) | \
                         dict({score_name: np.zeros(n_of_idxs) for score_name in score_names})
    
    dscm = DSCM(links=config, f=f)

    if not dscm.check_stationarity():
        print(f"Warning: cannot perform experiment on non-stationary DSCM!")
        return dict()

    adj_matrices = dscm.get_adjacency_matrices()
    summary_graph = dscm.get_summary_graph(adj_matrices=adj_matrices)

    if is_cyclic(graph_repr=summary_graph):
        print(f"Warning: cannot perform experiment on cyclic graph!")
        return dict()

    _, cpdag_repr = cpdag_from_dag(dag_repr=summary_graph)
    _, equivalence_class = equivalence_class_from_cpdag(cpdag_repr=cpdag_repr)

    if n := len(equivalence_class) == 1:
        print(f"Warning: cannot perform experiment on MEC of size {n}!")
        return dict()

    tau_max = len(adj_matrices) - 1
    edges = list(zip(*np.where(cpdag_repr == 1)))
    true_idx = get_graph_index(graphs=equivalence_class,
                               graph=summary_graph)

    for time_idx, t in enumerate(T):
        obs_data = dscm.generate_obs_data(T=t)
        opt_idx = run_experiment(obs_data=obs_data, edges=edges, equivalence_class=equivalence_class,
                                    tau_max=tau_max, oracle=oracle, method=method)
        opt_graph = equivalence_class[opt_idx]

        performance_scores["shd_scores"][time_idx] = get_structural_hamming_distance(true_graph=summary_graph, optimal_graph=opt_graph)
        performance_scores["mse_scores"][time_idx] = get_mse_scores(true_graph=summary_graph, optimal_graph=opt_graph)
        performance_scores["accuracy_scores"][time_idx] = get_accuracy_scores(true_graph=summary_graph, optimal_graph=opt_graph)
        performance_scores["precision_scores"][time_idx],  performance_scores["recall_scores"][time_idx] = get_precision_recall_scores(
                                                                                    true_graph=summary_graph, optimal_graph=opt_graph)
        performance_scores["equality_scores"][time_idx] = 1 if is_identical_graph(first_graph = summary_graph, second_graph=opt_graph) else 0

    return performance_scores


def test_configurations(configs: dict, f: callable, oracle: Oracle, T: int, N: int, method: str):

    for iteration in N:
        # TODO: generate random configuration with fixed parameter settings
        # TODO: get performance scores
        # TODO: get averages, map parameter setting to averages
        pass

    ### configs: mapping from integers to dictionaries ###

    ## TEST FOR DIFFERENT VALUES OF T ###

    config_scores = np.zeros(len(configs))

    ### run epoch for every configuration ###

    ### save scores to file using config index ###

    # for each configuraiton,


if __name__ == "__main__":
    def f(x): return x
    def g(x): return (1-4*np.e**(-x**2/2))*x
    def h(x): return (1-4*x**3 * np.e**(-x**2/2))*x

    # links = {
    #         0: [((0,-1), 0.5)],
    #         1: [((1,-1), 0.4), ((0,0), 0.4), ((2,0), 0.5)],
    #         2: [((2,-1), 0.8)]
    # }

    links = {
        0: [((0, -1), 0.7)],
        1: [((1, -1), 0.7), ((0, 0), -0.8)],
        2: [((2, -1), 0.5), ((1, 0), 0.5)],
        # 3: [((3, -1), 0.5), ((0, 0), 0.5)],
        # 4: [((4, -1), 0.5), ((1, 0), 0.5)],
        # 5: [((5, -1), 0.5), ((1, -1), 0.2)],
        # 6: [((6, -1), 0.5), ((5, 0), 0.5), ((1, -1), 0.2)],
        # 7: [((7, -1), 0.5), ((5, 0), 0.5), ((1, -1), 0.2)],
        # 8: [((8, -1), 0.5), ((1, -1), 0.2)],
        # 9: [((4, -1), 0.5), ((7, -1), 0.2)],
        # 10: [((4, -1), 0.5), ((9, 0), 0.5)],
        # 11: [((4, -1), 0.5), ((8, 0), 0.5)],
    }

    oracle = PartialCorrelation()

    performance_scores = run_epoch(config=links, f=f, oracle=oracle,
                                   T=np.arange(10, 151, 10), N=100, method="pcmciplus")
    print(performance_scores)

    # t = run_experiment(T = 100, links = links, map = f, oracle = pc, \
    #                    N = 100, score_precision = 10, instant_effects = False, lag_specific = False)
