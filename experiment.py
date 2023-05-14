import os, dscm, configs, metrics, random,data_recs, indep_tests, scoring_function, graph_reps
import numpy as np

def get_scores(true_graph : np.ndarray, phi_graph : np.ndarray, pcmci_graph : np.ndarray, cpdag_repr : np.ndarray, edges : list, true_graph_equiv : bool = True) -> tuple:
    optimal_scores = [None]*4
    
    gsf_scores = metrics.get_all_scores(true_graph, phi_graph, edges)
    pcmci_scores = metrics.get_all_scores(true_graph, pcmci_graph, edges)

    if not true_graph_equiv:
        _, equiv_class = graph_reps.equivalence_class_from_cpdag(cpdag_repr = pcmci_graph)
        scores = [metrics.get_all_scores(true_graph, graph, edges)[2] for graph in equiv_class]
        optimal_scores = metrics.get_all_scores(true_graph, equiv_class[np.argmax(scores)], edges)

    return gsf_scores, pcmci_scores, optimal_scores

def run_experiment(obs_data: np.ndarray, true_graph: np.ndarray, true_cpdag: np.ndarray,
                   oracle: indep_tests.Oracle, method: str = "pcmciplus", true_graph_equiv: bool = True) -> np.ndarray:
    """ Run experiment and retrieve scores.

    """

    p_matrices, pcmci_graph = scoring_function.run_pcmci(obs_data = obs_data, oracle = oracle, method = method)

    cpdag_repr = true_cpdag if true_graph_equiv else pcmci_graph

    _, equiv_class = graph_reps.equivalence_class_from_cpdag(cpdag_repr = cpdag_repr)
    arcs = list(zip(*np.where(cpdag_repr == 1))) 

    phi_graph = scoring_function.estimate_graph(equiv_class = equiv_class, arcs = arcs,
                                                p_matrices = p_matrices, oracle = oracle, 
                                                true_graph_equiv = true_graph_equiv)
    
    adjacencies = list(zip(*np.where(cpdag_repr == 1)))
    edges = [(i,j) for (i,j) in adjacencies if (j,i) in adjacencies]
    arcs = [(i,j) for (i,j) in adjacencies if (j,i) not in adjacencies]
    equiv_class_size = len(equiv_class)
    n_of_edges = len(edges)
    n_of_arcs = len(arcs)
    
    gsf_scores, pcmci_scores, optimal_scores = get_scores(true_graph = true_graph, phi_graph = phi_graph, pcmci_graph = pcmci_graph,
                                                                        cpdag_repr = cpdag_repr, edges = edges, true_graph_equiv = true_graph_equiv)
            

    return gsf_scores, pcmci_scores, optimal_scores, equiv_class_size, n_of_edges, n_of_arcs

def generate_models(d : int, coeff_params : np.ndarray, tau_params : np.ndarray, f_params : list, oracle_params : list) -> tuple:
    config = configs.get_random_config(d = d, coeff_params = coeff_params, 
                                                tau_params = tau_params)
    linear_idx = 0
    nonlinear_idx = np.random.choice(np.arange(0, len(f_params), 1))

    models = [dscm.DSCM(links = config, f = f_params[linear_idx]), 
              dscm.DSCM(links = config, f = f_params[nonlinear_idx])]
    oracles = [oracle_params[linear_idx], oracle_params[nonlinear_idx]]

    true_graph = graph_reps.get_summary_graph(amats = models[0].get_adjacency_matrices())

    return config, models, oracles, true_graph

def is_valid_model(models : list, true_graph_equiv : bool = True) -> bool:
    for model in models:
        if not model.is_stationary_process():
            return False

        true_graph = graph_reps.get_summary_graph(amats = model.get_adjacency_matrices())

        if graph_reps.is_cyclic(true_graph):
                return False
        
        _, cpdag_repr, _ = graph_reps.cpdag_from_dag(true_graph)
        _, equiv_class = graph_reps.equivalence_class_from_cpdag(cpdag_repr)

        if len(equiv_class) == 1 and true_graph_equiv: 
            return False
    
    return True

def run_epoch(d : int, coeff_params: np.ndarray, tau_params: np.ndarray, sample_params : np.ndarray, f_params : list, 
              oracle_params: list, experiment : str, method: str = "pcmciplus", data_transform: callable = np.cbrt, n_of_iter : int = 100, 
              verbose: bool = True, true_graph_equiv: bool = True, seed : int = 42) -> tuple:
    """ Runs experiment on a random configuration [...]

    """
    
    iteration = 0

    results_folder = f"results_{experiment}"

    if not os.path.exists(results_folder):
         os.makedirs(results_folder)

    random.seed(seed)

    while iteration < n_of_iter:
        savepath = f"{results_folder}\\{d}-{iteration}.pickle"

        if os.path.exists(savepath):
             iteration+=1
             continue
        
        results = dict({method: np.zeros((len(sample_params), 4)) for method in range(0,5)})

        if verbose: print(f"Started experiment {iteration+1}/{n_of_iter} for d={d}...")
                    
        config, models, oracles, true_graph = generate_models(d = d, coeff_params = coeff_params, 
                                                  tau_params = tau_params, f_params = f_params, oracle_params = oracle_params)
        
        if not is_valid_model(models): continue
      
        _, true_cpdag, _ = graph_reps.cpdag_from_dag(dag_repr = true_graph)
        data_recs.save_config(config = config, filename = f"{d}-{iteration}.pickle", 
                                                    folder = f"configs_{experiment}")

        for t_idx, t in enumerate(sample_params):
            if verbose: print(f"\t Evaluating sample size t = {t}...")

            for result_idx, (model, oracle) in enumerate(zip(models, oracles)):
                obs_data = model.generate_obs_data(t = t , data_transform = data_transform)
                dependencies = "linear" if result_idx==1 else "nonlinear"
                if verbose: print(f"\t\t Evaluating {dependencies} dependencies...")
                
                gsf_scores, pcmci_scores, optimal_scores, n_of_members, n_of_edges, n_of_arcs = run_experiment(obs_data = obs_data, true_graph = true_graph, 
                                                                                                true_cpdag = true_cpdag, oracle = oracle, method = method, 
                                                                                                                    true_graph_equiv = true_graph_equiv)
                
                results[(result_idx*2)][t_idx, :] = gsf_scores
                results[(result_idx*2) + 1][t_idx, :] = pcmci_scores
                results[4][t_idx, :] = optimal_scores

            results.update({5: n_of_members, 6 : n_of_edges, 7 : n_of_arcs})

 
        data_recs.save_results(results = results, 
                               savepath = savepath)

        iteration+=1
        
        if verbose: print(f"Successfully passed {iteration+1}'th iteration for d={d}!")

    if verbose: print(f"Successfully passed epoch of {n_of_iter} iterations for d={d}!")

def test_configs(var_params: list, coeff_params: np.array, tau_params: np.array, sample_params: np.array,
                 f_params: list, oracle_params: list, experiment: str, method: str = "pcmciplus", data_transform: callable = np.cbrt,
                 verbose: bool = True, n_of_iter : int = 100, true_graph_equiv: bool = True, seed : int = 42) -> dict:
    """ Evaluates all input configurations on varying sample sizes and dependencies.

    """

    for d in var_params:
        run_epoch(d = d, coeff_params = coeff_params, tau_params = tau_params, sample_params = sample_params, f_params = f_params, 
                  oracle_params = oracle_params, method = method, data_transform = data_transform, n_of_iter = n_of_iter, verbose = verbose, 
                  true_graph_equiv = true_graph_equiv, seed = seed, experiment = experiment)


if __name__ == "__main__":
    var_params, coeff_params, tau_params, sample_params, f_params, oracle_params = configs.get_params()

    test_configs(var_params, coeff_params, tau_params, sample_params, 
                 f_params, oracle_params, experiment = "with_access", true_graph_equiv=True, n_of_iter = 100)
    test_configs(var_params, coeff_params, tau_params, sample_params, 
                 f_params, oracle_params, experiment = "without_access", true_graph_equiv=False, n_of_iter = 100)






