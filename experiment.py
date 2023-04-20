import os
import sys
import numpy as np
from dscm import DSCM
from param_configs import get_params, get_random_config
from metrics import get_all_scores
from data_recs import save_results, get_config
from indep_tests import Oracle
from scoring_function import score_equivalence_class, get_optimal_graph_index
from graph_reps import cpdag_from_dag, equivalence_class_from_cpdag, is_cyclic, get_graph_index, get_summary_graph

### Runge's experimental description is found here: https://arxiv.org/pdf/2003.03685.pdf ###

def estimate_graph(obs_data: np.array, arcs: list, equivalence_class: np.array, oracle: Oracle,
                   method: str = "pcmciplus") -> tuple:
    """ Estimate graph from observational data.

            :param obs_data:
                matrix of observational data for time series
            :param arcs:
                list of arcs

    """

    scores, pcmci_graph = score_equivalence_class(obs_data, arcs, equivalence_class, oracle, method=method)
    phi_graph_idx, *_ = get_optimal_graph_index(scores=scores)
    pcmci_graph_idx = get_graph_index(graphs=equivalence_class, graph=pcmci_graph)

    phi_graph = equivalence_class[phi_graph_idx]

    return phi_graph, pcmci_graph #, phi_graph_idx, pcmci_graph_idx


def run_experiment(obs_data: np.array, arcs: list, true_graph: np.array, equivalence_class: np.array,
                   oracle: Oracle, method: str = "pcmciplus") -> np.array:
    """ Run experiment and retrieve scores.

    """

    phi_graph, pcmci_graph = estimate_graph(obs_data, arcs, equivalence_class, oracle, method=method)

    phi_scores = get_all_scores(true_graph, phi_graph, arcs)
    pcmci_scores = get_all_scores(true_graph, pcmci_graph, arcs)

    return phi_scores, pcmci_scores, phi_graph, pcmci_graph

def run_epoch(var_param_config: dict, coeff_params: np.array, tau_params: np.array, func: callable,
              oracle: Oracle, t: int,  method: str = "pcmciplus", data_transform: callable = np.cbrt, num_of_exp: int = 100, verbose: bool = True) -> tuple:
    """ Runs experiment on a single variable parameter configuration with random
        coefficients for num_of_exp times.

    """

    phi_scores_epoch = np.zeros((num_of_exp, 3))
    pcmci_scores_epoch = np.zeros((num_of_exp, 3))

    iteration = 0

    while iteration < num_of_exp:
        random_config = get_random_config(var_param_config=var_param_config,
                                          coeff_params=coeff_params, tau_params=tau_params)
        
        dscm = DSCM(links = random_config, f = func)

        if not dscm.check_stationarity(): continue

        obs_data = dscm.generate_obs_data(t = t, data_transform = data_transform)

        true_graph = get_summary_graph(amats=dscm.get_adjacency_matrices())

        if is_cyclic(true_graph): continue

        _, cpdag_repr, _ = cpdag_from_dag(true_graph)
        arcs = list(zip(*np.where(cpdag_repr == 1)))
        _, equivalence_class, _ = equivalence_class_from_cpdag(cpdag_repr=cpdag_repr)

        if len(equivalence_class) == 1: continue

        phi_scores_iter, pcmci_scores_iter, phi_graph, pcmci_graph = run_experiment(obs_data, arcs, true_graph, 
                                                                                    equivalence_class, oracle,  method=method)
        phi_scores_epoch[iteration, :] = phi_scores_iter
        pcmci_scores_epoch[iteration, :] = pcmci_scores_iter

        iteration += 1

    return phi_scores_epoch, pcmci_scores_epoch, phi_graph, pcmci_graph, true_graph

def test_configs(var_param_configs: list, coeff_params : np.array, tau_params : np.array, sample_params : np.array, func_params : list, 
                 oracle_params : list, method: str = "pcmciplus", data_transform: callable = np.cbrt, verbose : bool = True, folder : str = "results") -> dict:
    """
            Evaluates all input configurations on varying sample sizes and dependencies.
    """
    num_configs = len(var_param_configs)

    for iteration, var_param_config in enumerate(var_param_configs):
        scores = dict()
        n_of_var, n_of_instant, n_of_lagged = tuple(var_param_config.values())
        filename = f"{n_of_var}-{n_of_instant}-{n_of_lagged}"

        if verbose: print(f"Testing for configuration {filename}...")

        if os.path.exists(f"{folder}\\{filename}.pickle"):
            continue

        for t in sample_params:
            scores[t] = dict()

            if verbose: print(f"\t Testing for sample size {t}...")

            nonlinear_idx = np.random.choice(np.arange(0, len(func_params), 1))

            linear_func, linear_oracle = func_params[0], oracle_params[0]
            nonlinear_func, nonlinear_oracle = func_params[nonlinear_idx], oracle_params[nonlinear_idx]

            if verbose: print(f"\t\t Testing linear dependencies...")

            linear_phi_scores, linear_pcmci_scores, linear_phi_graph, linear_pcmci_graph, true_graph = run_epoch(var_param_config = var_param_config, 
                                                                                                                            coeff_params=coeff_params, 
                                                                                                                            tau_params = tau_params, 
                                                                                                                            func = linear_func, 
                                                                                                                            oracle = linear_oracle, 
                                                                                                                            t=t, method = method) 
            if verbose: print(f"\t\t Testing non-linear dependencies...")

            nonlinear_phi_scores, nonlinear_pcmci_scores, nonlinear_phi_graph, nonlinear_pcmci_graph, _ = run_epoch(var_param_config = var_param_config, 
                                                                                                                                coeff_params=coeff_params, 
                                                                                                                                tau_params = tau_params, 
                                                                                                                                func = nonlinear_func, 
                                                                                                                                oracle = nonlinear_oracle, 
                                                                                                                                t=t, method = method) 
            scores[t]["true_graph"] = true_graph
            scores[t]["linear"] = dict({"phi": linear_phi_scores, 
                                        "pcmci" : linear_pcmci_scores,
                                        "linear_phi_graph" : linear_phi_graph,
                                        "linear_pcmci_graph" : linear_pcmci_graph})
            scores[t]["nonlinear"] = dict({"phi": nonlinear_phi_scores, 
                                        "pcmci" : nonlinear_pcmci_scores,
                                        "nonlinear_phi_graph" : nonlinear_phi_graph,
                                        "nonlinear_pcmci_graph" : nonlinear_pcmci_graph})
        
        if verbose: print(f"Passed experiment {iteration+1}/{num_configs}!")

        save_results(results = scores, 
                     filename = filename, 
                     folder = folder)

if __name__ == "__main__":
    var_params, coeff_params, tau_params, sample_params, func_params, oracle_params = get_params()

    var_param_configs = []
    config_names = os.listdir(f"{os.getcwd()}\\configs")

    for config_name in config_names:
        if config_name.endswith("0-0"): continue
        if config_name[0] not in ["6", "7", "8", "9"]: continue 

        var_param_config = get_config(config_name)
        var_param_configs.append(var_param_config)
            
    test_configs(var_param_configs, coeff_params, tau_params, 
                 sample_params, func_params, oracle_params, folder = "results")

