from dscm import DSCM
from independence_tests import Oracle, PartialCorrelation, GaussianProcessRegression
from graph_representations import dag_to_cpdag, cpdag_to_markov_equivalence_class
import numpy as np

def get_indices(edge_idxs: np.array, num_of_vars: int) -> tuple:
    """ Converts indices in time lagged matrix to normal format.
        :param edge_idxs: indices of present edges
        :num_of_vars: number of time series elements
        :returns cause_idx: 
        :returns effect_idx:
        :returns lag:
    """

    cause_idx, effect_idx = np.remainder(edge_idxs, num_of_vars)
    time_idxs = np.floor(np.divide(edge_idxs, num_of_vars)).astype(int)
    lag = time_idxs[1] - time_idxs[0]

    return cause_idx, effect_idx, lag


def get_condition_sets(obs_data: np.array, obs_idxs: np.array, cause_idx: int, effect_idx: int, lag: int, tau_max: int, \
                                                                                            lag_specific: bool = False) -> np.array:
    """ Return condition sets for conditional independence tests.
<<<<<<< HEAD
        :param obs_data: array of observations for time series elements
        :param cause_idx: index of cause at each time
        :param effect_idx: index of effect at each time
        :param lag: lag between cause and effect
        :tau_max: absolute value of largest lag
        :lag_specific: check whether to partition data for lag-specific causes
        :returns cause: array of observations for cause
        :returns effect: array of observations for effect
        :returns condition: array of observations for conditional set 
=======

        :param obs_data: array of observations for time series elements

        :param cause_idx: index of cause at each time

        :param effect_idx: index of effect at each time

        :param lag: lag between cause and effect

        :tau_max: absolute value of largest lag

        :lag_specific: check whether to partition data for lag-specific causes

        :returns cause: array of observations for cause

        :returns effect: array of observations for effect

        :returns condition: array of observations for conditional set 

>>>>>>> ab43b717a8e6464e7f708c8625c3976bffd325f0
    """

    past_idxs = np.array(list(zip(obs_idxs - tau_max, obs_idxs - 1))).astype(int)
    past = obs_data[past_idxs]
    present = obs_data[obs_idxs]

    ### partition for classic Granger causality ###
    if not lag_specific:
        cause = past[:, 0, cause_idx]
        effect = present[:, effect_idx]

        condition = np.delete(obs_data[:-tau_max], cause_idx, axis=1)

        return cause, effect, condition

    ### partition for instantaneous effects ### 
    elif lag == 0:
        cause = obs_data[:, cause_idx]
        effect = obs_data[:, effect_idx]
        condition = np.delete(obs_data, cause_idx, axis=1)

        return cause, effect, condition
    
    ### partition for non-instantaneous, lag-specific effects ###

    cause = past[:, -lag, cause_idx] if 0 < lag \
                        else present[:, cause_idx]
    effect = present[:, effect_idx]

    condition = np.copy(past[:, :, :])
    condition[:, -lag, cause_idx] = np.nan
    condition = np.array([np.ndarray.flatten(row[~ np.isnan(row)]) for row in condition])

    return cause, effect, condition


def get_score_matrix(obs_data: np.array, cpdag_repr: np.array, max_lag: int, oracle: Oracle = lambda x, y, z: np.nan, \
                            score_precision: int = 3, instant_effects: bool = True, lag_specific:bool=False) -> np.array:
    """ Constructs scoring matrix from observational data given an independence oracle.
        :param obs_data: observational data where each column is a time series realisation
        :param max_lag: maximal lag (get from graph instead of from DSCM)
        :param oracle: conditional independence tester
        :returns score_matrix: matrix consisting of scores for each non-zero entry in the CPDAG
    """

    T, d = obs_data.shape
    tau_max = abs(max_lag)
    gamma = tau_max + 1
    idx_to_values, values_to_score = dict(), dict()
    score_matrix = np.zeros((d, d))
    edge_idxs = list(zip(*np.where(cpdag_repr == 1)))
    obs_idxs = np.arange(0, T - tau_max, 1) + tau_max

    for map_idx, edge_idx in enumerate(edge_idxs):
        (cause_idx, effect_idx), lag = edge_idx, 0 if instant_effects \
                    else get_indices(edge_idxs=edge_idx, num_of_vars=d)

        idx_to_values[map_idx] = (cause_idx, effect_idx, lag)

        ### assign score if already known ###
        if (score := values_to_score.get((cause_idx, effect_idx, lag))):
            score_matrix[edge_idx] = score
            continue

        cause, effect, condition = get_condition_sets(obs_data=obs_data, obs_idxs=obs_idxs, \
                                                      cause_idx=cause_idx, effect_idx=effect_idx, \
                                                        lag=lag, tau_max=tau_max, lag_specific=lag_specific)

        score_matrix[edge_idx] = round(1 - oracle.get_significance(X=cause, Y=effect, \
                                            Z=condition, T=T, d=d), score_precision)

        values_to_score[(cause_idx, effect_idx, lag)] = score_matrix[edge_idx]

    return score_matrix

def score_equivalence_class(obs_data: np.array, cpdag_repr: np.array, max_lag: int, true_graph: np.array, oracle: Oracle, \
                                       score_precision: int = 3, instant_effects: bool = True, lag_specific:bool=False) -> np.array:
    """ Scores input equivalence class represented as PCDAG.
            :param obs_data: observational data where each column is a time series realisation
            :param cpdag_repr: PCDAG, represented in array format

            :param true_graph: ground truth DAG, represented in array format

            :param oracle: conditional independence tester
            
            :returns scores: array of scores on each DAG

            #TODO
    """

    T, d = obs_data.shape

    max_lag = int(n/d) - 1
    _, equivalence_class = cpdag_to_markov_equivalence_class(
        cpdag_repr=cpdag_repr)
    scores = np.zeros(len(equivalence_class))

    score_matrix = get_score_matrix(obs_data = obs_data, cpdag_repr = cpdag_repr, \
                                    max_lag = max_lag, oracle = oracle, score_precision = score_precision, \
                                        instant_effects = instant_effects, lag_specific=lag_specific)
                                    
    for idx, member in enumerate(equivalence_class):
        score = np.sum(np.multiply(member, score_matrix))
        scores[idx] = score

    return equivalence_class, scores


def get_best_candidate(equivalence_class: np.array, scores: np.array):
    opt_score_idx = np.argmax(scores)

    return equivalence_class[opt_score_idx]


def structural_hamming_distance(true_graph : np.array, candidate_graph : np.array) -> int:
    raise NotImplementedError()

    return None


if __name__ == "__main__":
    def f(x): return x
    def g(x): return (1-4*np.e**(-x**2/2))*x
    def h(x): return (1-4*x**3 * np.e**(-x**2/2))*x

    links_coeffs = {
        0: [((0, -1), 0.7), ((0, -2), -0.8)],
        1: [((1, -1), 0.7), ((0, 0), -0.8)],
        2: [((2, -1), 0.5), ((1, 0), 0.5), ((1, -1), 0.2)],
        3: [((3, -1), 0.5), ((0, 0), 0.5), ((1, -1), 0.2)],
        4: [((4, -1), 0.5), ((1, 0), 0.5), ((1, -1), 0.2)],
        5: [((4, -1), 0.5), ((1, -1), 0.2)],
        6: [((4, -1), 0.5), ((6, 0), 0.5), ((1, -1), 0.2)],
        # 7: [((4, -1), 0.5), ((5, 0), 0.5), ((1, -1), 0.2)],
        # 8: [((4, -1), 0.5), ((1, -1), 0.2)],
        # 9: [((4, -1), 0.5), ((7, -1), 0.2)],
        # 10: [((4, -1), 0.5), ((9, 0), 0.5), ((1, -1), 0.2)],
        # 11: [((4, -1), 0.5), ((8, 0), 0.5), ((1, -1), 0.2)],
    }

    pc = PartialCorrelation()
    dscm = DSCM(links=links_coeffs, mapping=f)

    obs_data = dscm.generate_obs_data(T=20)
    lag_matrix, instant_matrix = dscm.get_adjacency_matrices()

    print(f"True matrix:\n \
            {instant_matrix}")

    _, cpdag_repr, _ = dag_to_cpdag(instant_matrix)

    T, d = obs_data.shape
    _, n = lag_matrix.shape
    max_lag = (n/d)-1

    equivalence_class, scores = score_equivalence_class(obs_data = obs_data, cpdag_repr = cpdag_repr, \
                                                max_lag = max_lag, true_graph=instant_matrix, \
                                                oracle = pc, score_precision = 3, instant_effects = False, lag_specific=False)

    idx = np.argmax(scores)
    best_candidate = equivalence_class[idx]

    true_idx = None

    for i in range(len(equivalence_class)):
        if np.array_equal(equivalence_class[i], instant_matrix):

            true_idx = idx

    print("index of true graph in MEC", true_idx)

    print(f"Picked index {idx} from scores {scores}")
    print(f"Size of MEC: {len(equivalence_class)}")
    # print(f"MEC:\n \
    #       {equivalence_class}")

    print(f"Best candidate:\n \
          {best_candidate}")

    print(f"Number of errors: {np.sum(best_candidate != instant_matrix)}")

    print(f"Identical matrix: {(best_candidate == instant_matrix).all()}")