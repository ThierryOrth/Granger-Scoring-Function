import configs, data_recs, os, scipy, pingouin
import numpy as np
from collections import defaultdict
from variables import METHODS, SCORES, EXPERIMENTS

def get_ks_distance(x : np.ndarray, y : np.ndarray, precision : int = 3) -> tuple:
    results = scipy.stats.ks_2samp(x, y)

    ks_dist = round(results.statistic, precision)
    pval = round(results.pvalue, precision)
    
    return ks_dist, pval

def get_cor(x : np.ndarray, y : np.ndarray, precision : int = 3) -> tuple:    
    cor, pval = scipy.stats.spearmanr(x,y)
    cor = round(cor, precision)
    pval = round(pval, precision)

    return cor, pval

# make single function here to set data

def get_sample_results(sample_matr : np.ndarray, sample_params: np.ndarray) -> defaultdict:
    data = defaultdict(lambda : [])
    n_of_methods = 4
    n_of_methods, (_, n_of_scores, n_of_var, n_of_iter, n_of_sample) = 4, sample_matr.shape

    sample_param_matr = np.zeros((n_of_var, n_of_iter, n_of_sample))
    
    for sample_idx in range(n_of_sample):
        sample_param_matr[:, :, sample_idx] = sample_params[sample_idx]
    sample_param_matr = sample_param_matr.flatten()

    # every column in the final matrix corresponds to a sample size,

    for method_idx in range(n_of_methods):
        data[f"methods"].append(METHODS[method_idx])

        for score_idx in range(n_of_scores):
            res = sample_matr[method_idx, score_idx, :, :, :].flatten()
            dcor, pval = get_cor(x = res, y = sample_param_matr)

            data[f"{SCORES[score_idx]}"].append(f"$\\delta={dcor}, p={pval}$")


    return data

def get_dim_results(dim_matr : np.ndarray, equiv_class_data : np.ndarray, var_params : np.ndarray) -> defaultdict:
    data = defaultdict(lambda : [])
    n_of_methods = 4
    member_score_idx, edge_score_idx, prop_score_idx = 0, 1, 2
    _, n_of_scores, n_of_var, n_of_iter = dim_matr.shape
    var_matr = np.zeros((n_of_var, n_of_iter))

    for var_idx in range(n_of_var):
        var_matr[var_idx, :] = var_params[var_idx]

    for method_idx in range(n_of_methods):
        for score_idx in range(n_of_scores):
            data["indices"].append(method_idx)

            results = dim_matr[method_idx, score_idx, :, :].flatten()
            var_corr = get_cor(results, var_matr.flatten())
            member_corr = get_cor(results, equiv_class_data[:, :, member_score_idx].flatten())
            edge_corr = get_cor(results, equiv_class_data[:, :, edge_score_idx].flatten())
            prop_corr = get_cor(results, equiv_class_data[:, :, prop_score_idx].flatten())

            #print(METHODS[method_idx], SCORES[score_idx], var_corr, member_corr, edge_corr, prop_corr)
            for corr in [var_corr, member_corr, edge_corr, prop_corr]:
                data[f"{SCORES[score_idx]}"].append(f"$\\delta={corr[0]}, p={corr[1]}$")

    return data

def get_line_dist(dim_matr : np.ndarray, experiment : str, precision : int = 3) -> dict:
    data = defaultdict(lambda : [])
    n_of_methods, (_, n_of_scores, n_of_var, n_of_iter) = 4, dim_matr.shape

    for method_idx in range(0, n_of_methods, 2):
        data[f"methods"].append(f"{METHODS[method_idx]}/{METHODS[method_idx+1]}")

        if experiment == "without_access":
            data[f"methods"].append(f"{METHODS[method_idx]}/optimal")
            data[f"methods"].append(f"{METHODS[method_idx+1]}/optimal")

        for score_idx in range(n_of_scores):
            gsf_results = dim_matr[method_idx, score_idx, :, :].flatten()
            pcmci_results = dim_matr[method_idx + 1, score_idx, :, :].flatten()

            method_dist = np.abs(gsf_results - pcmci_results)
            data[f"{SCORES[score_idx]}"].append(f"$\\mu = {round(np.mean(method_dist), precision)}, \\sigma = {round(np.std(method_dist), precision)}$")

            if experiment == "without_access":
                opt_results = dim_matr[4, score_idx, :, :].flatten()

                for results in [gsf_results, pcmci_results]:
                    opt_dist = np.abs(results - opt_results)
                    data[f"{SCORES[score_idx]}"].append(f"$\\mu={round(np.mean(opt_dist), precision)}, \\sigma={round(np.std(opt_dist), precision)}$")

    return data

def get_prop_results(dim_matr : np.ndarray, prop_matr : np.ndarray) -> defaultdict:
    data = defaultdict(lambda : [])
    _, n_of_scores, *_ = dim_matr.shape
    n_of_methods = 4

    for method_idx in range(n_of_methods):
        data[f"methods"].append(METHODS[method_idx])

        for score_idx in range(n_of_scores):
            props = prop_matr[method_idx, score_idx, :, :].flatten()
            results = dim_matr[method_idx, score_idx, :, :].flatten()
            dcor, pval = get_cor(props, results)

            data[f"{SCORES[score_idx]}"].append(f"$\\delta={dcor}, p={pval}$")

    return data

# def get_method_results(dim_matr : np.ndarray) -> defaultdict:
#     data = defaultdict()
#     _, n_of_scores, *_ = dim_matr.shape
#     n_of_methods = 4

#     for method_idx in range(0, n_of_methods, 2):
#         for score_idx in range(n_of_scores):
#             gsf_res = dim_matr[method_idx, score_idx, :, :].flatten()
#             pcmci_res = dim_matr[method_idx+1, score_idx, :, :].flatten()
#             ks_res = get_ks_distance(gsf_res, pcmci_res)

#             data[f"{METHODS[method_idx]}/{METHODS[method_idx+1]}/{SCORES[score_idx]}"] = [f"$ks={ks_res[0]}, p={ks_res[1]}$"] 
        
#     return data

# def get_dep_results(dim_matr : np.ndarray) -> defaultdict:
#     data = defaultdict()
#     _, n_of_scores, *_ = dim_matr.shape
#     n_of_methods = 4

#     for method_idx in range(0, n_of_methods - 2):
#         for score_idx in range(n_of_scores):
#             lin_res = dim_matr[method_idx, score_idx, :, :].flatten()
#             nonlin_res = dim_matr[method_idx+2, score_idx, :, :].flatten()
#             ks_res = get_ks_distance(lin_res, nonlin_res)

#             data[f"{METHODS[method_idx]}/{METHODS[method_idx+1]}/{SCORES[score_idx]}"] = [f"$ks={ks_res[0]}, p={ks_res[1]}$"] 

#     return data

def write_descriptive_stats(experiment : str, n_of_iter : int, folder : str = f"descriptive_statistics"):
    if not os.path.exists(folder):
        os.makedirs(folder)

    var_params, coeff_params, tau_params, sample_params, func_params, oracle_params = configs.get_params()
    dim_matr, sample_matr, inst_prop_matr, lag_prop_matr, equiv_class_data = data_recs.gather_data(experiment = experiment, var_params = var_params, 
                                                                                                sample_params = sample_params, n_of_iter = n_of_iter)

    #print(dim_matr.shape, sample_matr.shape)
    data_recs.write_data_to_table(data = get_sample_results(sample_matr = sample_matr, sample_params = sample_params), 
                                                            filename = f"{folder}\\sample_results_{experiment}")
    
    data_recs.write_data_to_table(data = get_prop_results(dim_matr = dim_matr, prop_matr = lag_prop_matr), 
                                                            filename = f"{folder}\\lagged_results_{experiment}")
    
    data_recs.write_data_to_table(data = get_prop_results(dim_matr = dim_matr, prop_matr = inst_prop_matr), 
                                                            filename = f"{folder}\\instant_results_{experiment}")

    data_recs.write_data_to_table(data = get_dim_results(dim_matr = dim_matr, equiv_class_data = equiv_class_data, var_params = var_params),
                                                                filename = f"{folder}\\dim_results_{experiment}")
    data_recs.write_data_to_table(data = get_line_dist(dim_matr = dim_matr, experiment = experiment),
                                                                filename = f"{folder}\\distance_results_{experiment}")
    
    # data_recs.write_data_to_table(data = get_method_results(dim_matr = dim_matr), 
    #                               filename = f"{folder}\\method_results_{experiment}")

    # data_recs.write_data_to_table(data = get_dep_results(dim_matr = dim_matr), 
    #                               filename = f"{folder}\\dep_results_{experiment}")

if __name__ == "__main__":    
    for experiment in EXPERIMENTS:
        write_descriptive_stats(experiment = experiment, n_of_iter = 50)

