import configs, data_recs, os
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from variables import COLORS, SCORES, METHODS, EXPERIMENTS

def dim_plots(dim_matr : np.ndarray, experiment : str) -> tuple:
    _, n_of_scores, n_of_vars, _ = dim_matr.shape
    fig, axes = plt.subplots(nrows = 1, ncols = n_of_scores, figsize=(22,9))
    n_of_methods = 4 

    for score_idx in range(n_of_scores):
        
        if experiment == "without_access":
            optimal_scores = np.mean(dim_matr[4, score_idx, :, :], axis = 1)
            axes[score_idx].plot(np.arange(2, n_of_vars + 2), optimal_scores, color = COLORS[4], label = f"{METHODS[4]}")

        for method_idx in range(n_of_methods):
            results = np.mean(dim_matr[method_idx, score_idx, :, :], axis=1)

            axes[score_idx].plot(np.arange(2, n_of_vars + 2), results, color = COLORS[method_idx], label = f"{METHODS[method_idx]}")
            
            axes[score_idx].set_xlabel(f"$d$")
            axes[score_idx].set_box_aspect(1)
            axes[score_idx].set_ylabel(f"{SCORES[score_idx]}")

            # for pcmci, we do not care about relative proportion when there is no access to the MEC 
            # for in that case, the accuracy is not meaningful

    handles, labels = axes[score_idx].get_legend_handles_labels()
    fig.legend(handles = handles, labels = labels,  loc = "lower right")

    return fig, axes

    # for second experiment, add best performing method in same plots

def sample_plots(sample_matr : np.ndarray, sample_params : np.ndarray) -> tuple:
    _, n_of_scores, n_of_vars, _, n_of_sample = sample_matr.shape
    fig, axes = plt.subplots(nrows = n_of_sample, ncols = n_of_scores, figsize=(20,28))

    n_of_methods = 4 

    for sample_idx in range(n_of_sample):
        for method_idx in range(n_of_methods):
            for score_idx in range(n_of_scores):
                results = np.mean(sample_matr[method_idx, score_idx, :, :, sample_idx], axis = 1)

                axes[sample_idx, score_idx].plot(np.arange(2, n_of_vars + 2), results, 
                                                 color = COLORS[method_idx], label = f"{METHODS[method_idx]}")

                axes[sample_idx, score_idx].set_xlabel(f"$d$")
                axes[sample_idx, score_idx].set_ylabel(f"{SCORES[score_idx]}")
                axes[sample_idx, score_idx].set_box_aspect(1)
                axes[sample_idx, score_idx].set_title(f"$t={sample_params[sample_idx]}$")

    handles, labels = axes[sample_idx, score_idx].get_legend_handles_labels()
    fig.legend(handles = handles, labels = labels,  loc = "lower right")

    return fig, axes

def aggr_prop_plots(dim_matr : np.ndarray, prop_matr : np.ndarray) -> tuple:
    _, n_of_scores, _, _ = prop_matr.shape
    n_of_methods = 4

    fig, axes = plt.subplots(nrows = n_of_methods, ncols = n_of_scores, figsize=(20,24))

    for score_idx in range(n_of_scores):
        for method_idx in range(n_of_methods):
            scores = defaultdict(lambda : [])

            props = prop_matr[method_idx, score_idx, :, :].flatten()
            results = dim_matr[method_idx, score_idx, :, :].flatten()

            for prop, result in zip(props, results):
                scores[prop].append(result)

            props = {prop: np.mean(result) for prop, result in scores.items()}
            
            axes[method_idx, score_idx].set_xlabel(f"$\%$")
            axes[method_idx, score_idx].set_ylabel(f"{SCORES[score_idx]}")
            axes[method_idx, score_idx].set_box_aspect(1)

            if score_idx == 0:
                axes[method_idx, score_idx].scatter(props.keys(), props.values(), 
                                                    color = COLORS[method_idx], label = f"{METHODS[method_idx]}")
                continue

            axes[method_idx, score_idx].scatter(props.keys(), props.values(), 
                                                    color = COLORS[method_idx])

    # handles, labels = axes[:, 0].get_legend_handles_labels()
    # handles = handles, labels = labels, 

    fig.legend(loc = "lower right")
    
    return fig, axes

def plot_equiv_class_data(equiv_class_data : np.ndarray) -> tuple:
    n_of_var, _, n_of_data = equiv_class_data.shape
    fig, axes = plt.subplots(nrows = 1, ncols = n_of_data, figsize=(12,7)) 
    names = ["Number of members", "Number of edges", "Percentage of edges"]

    for idx, name in enumerate(names):
        results = np.mean(equiv_class_data[:, :, idx], axis = 1)
        axes[idx].plot(np.arange(2, n_of_var+2), results, 
                       color = COLORS[idx], label = f"{name}")
        axes[idx].set_box_aspect(1)
        axes[idx].set_xlabel(f"$d$")
        axes[idx].set_ylabel(f"{name}")

    #handles, labels = axes[idx].get_legend_handles_labels()
    fig.legend(loc = "lower right")
    #fig.legend(handles = handles, labels = labels, loc = "lower right")
    return fig, axes

def generate_plots(var_params : np.ndarray,sample_params : np.ndarray, n_of_iter : int,
                   experiment : str, plot_folder: str = None, style: str = "fivethirtyeight"):
    if style in plt.style.available:
        plt.style.use(style)

    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)

    dim_matr, sample_matr, inst_prop_matr, lag_prop_matr, equiv_class_data = data_recs.gather_data(experiment = experiment, 
                                                                                                var_params = var_params, 
                                                                                                sample_params = sample_params, 
                                                                                                n_of_iter = n_of_iter)
    match indep_var:
        case "dim":
            fig, _ = dim_plots(dim_matr = dim_matr, experiment=experiment)

        case "sample":
            fig, _ = sample_plots(sample_matr = sample_matr, 
                                  sample_params = sample_params)

        case "aggr_instant":
            fig, _ = aggr_prop_plots(dim_matr = dim_matr,
                                           prop_matr = inst_prop_matr)

        case "aggr_lagged":
            fig, _ = aggr_prop_plots(dim_matr = dim_matr,
                                           prop_matr = lag_prop_matr)

        case "equiv_class_data":
            fig, _ = plot_equiv_class_data(equiv_class_data = equiv_class_data)

    if plot_folder:
        plt.savefig(f"{plot_folder}\\{experiment}_{indep_var}.png", bbox_inches='tight')
    else:
        plt.show()

if __name__ == "__main__":
    var_params, coeff_params, tau_params, sample_params, func_params, oracle_params = configs.get_params()
    n_of_iter = 50
    
    indep_vars = ["dim", "sample", "aggr_instant", 
                  "aggr_lagged", "equiv_class_data"]

    for experiment in EXPERIMENTS:
        for indep_var in indep_vars:
            generate_plots(experiment = experiment, sample_params = sample_params, 
                           var_params = var_params, n_of_iter = n_of_iter, plot_folder = "plots")