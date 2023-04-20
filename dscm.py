import numpy as np
import matplotlib.pyplot as plt

class DSCM:
    """ Defines a Dynamic Structural Causal Model.

            :param links: 
                set of parameters describing causal influence
            :param f: 
                transformation applied to non-reflexive causal influences
    """

    def __init__(self, links: dict, f: callable):
        self.links = links
        self.f = f

        self.num_of_vars = len(self.links)
        self.coeff_idx, self.lag_idx, self.cause_idx = 0, 1, 2

        self.matr_repr = self.get_matr_repr()
        self.max_tau = abs(self.get_max_tau())

    def get_max_tau(self):
        max_lag = 0

        for effect in range(self.num_of_vars):
            lag = min(map(lambda t: t[0][1], self.links[effect]))
            max_lag = lag if lag < max_lag else max_lag

        return max_lag

    def get_matr_repr(self) -> dict:
        """ Transforms dictionary to matrix representation for faster data generation.

                :returns matr_repr:
        """

        matr_repr = dict()

        for effect, causes in self.links.items():
            idx = 1
            matr_repr[effect] = np.zeros((len(causes), 3), dtype=object)

            for (cause, lag), coeff in causes:
                if cause == effect:
                    matr_repr[effect][0, :] = [coeff, lag, cause]
                else:
                    matr_repr[effect][idx, :] = [coeff, lag, cause]
                    idx += 1

        return matr_repr

    def generate_obs_data(self, t: int, gauss_init: bool = True, gauss_noise: bool = True, random_state: int = 42, data_transform: callable = np.cbrt) -> np.array:
        """ Generates observational data sample over the time interval {1, ..., T} 
            for each time series in the DSCM.

            :param gauss_init: 
                Boolean for initialising previous timesteps with standard Gaussian noise
            :param gauss_noise: 
                Boolean for additive standard Gaussian noise
            :param random_state: 
                seed for reproducibility
            :param transform:

            :returns obs_data:
                array of observations of shape T x d, with d the number of time series
        """

        np.random.seed(random_state)
        obs_data = np.zeros((self.max_tau + t, self.num_of_vars))

        if gauss_init:
            obs_data[:self.max_tau, :] = np.random.normal(0, 1, size=(self.max_tau, self.num_of_vars))

        for i in range(self.max_tau, self.max_tau+t):
            for j in range(self.num_of_vars):
                matr_repr = self.matr_repr[j]

                obs_data[i, j] = np.dot(matr_repr[0, self.coeff_idx], obs_data[i + matr_repr[0, self.lag_idx], matr_repr[0, self.cause_idx]]) + \
                                 (self.f(np.dot(matr_repr[1:, self.coeff_idx].astype(float), obs_data[i + matr_repr[1:, self.lag_idx].astype(int), matr_repr[1:, self.cause_idx].astype(int)].astype(float))) or 0.0) + \
                                                                                                                                                                        np.random.normal(0.0, 1.0) if gauss_noise else 0.0
                
        return data_transform(obs_data[self.max_tau:, :]) 

    def get_adjacency_matrices(self) -> np.array:
        """ Constructs adjacency matrices for contemporaneous and lagged causal relations from the DSCM
            parameters. On the assumption of causal stationarity, these matrices span the maximal time window
            defined as the absolute value of the maximal lag plus one.

                :returns adj_matrices: 
                    adjacency matrices with adj_matrices[tau, i, j] indicating whether X^{t-tau}_i -> X^t_j
        """

        adj_matrices = np.zeros((self.max_tau + 1, self.num_of_vars, self.num_of_vars)).astype(int)

        for effect in range(self.num_of_vars):
            matr_repr = self.matr_repr[effect]

            for _, row in enumerate(matr_repr):
                cause = row[self.cause_idx]
                tau = abs(row[self.lag_idx])

                adj_matrices[tau, cause, effect] = 1

        return adj_matrices

    def check_stationarity(self) -> bool:
        """ Code and comments in this function are a slight adaption from code by Jakob Runge, source: 
                https://github.com/jakobrunge/tigramite/blob/c877f0b578c357b1c26f666764be07676193d4dc/tigramite/toymodels/structural_causal_processes.py#L865

        Returns stationarity according to a unit root test. Assumes an at least 
        asymptotically linear vector autoregressive process without contemporaneous links.

            :returns stationary: 
                indicates if VAR process (DSCM) is stationary 
        """

        graph = np.zeros((self.num_of_vars, self.num_of_vars, self.max_tau))
        stability_matrix = np.zeros(
            (self.num_of_vars * self.max_tau, self.num_of_vars * self.max_tau))
        funcs = []

        for j in range(self.num_of_vars):

            for link in self.links[j]:
                (var, lag), coeff = link

                if 0 < abs(lag):
                    graph[j, var, abs(lag) - 1] = coeff

                funcs.append(self.f)

        idx = 0

        for i in range(0, self.num_of_vars*self.max_tau, self.num_of_vars):
            stability_matrix[:self.num_of_vars, i:i +
                             self.num_of_vars] = graph[:, :, idx]
            if idx < self.max_tau - 1:
                stability_matrix[i+self.num_of_vars:i+2*self.num_of_vars,
                                 i:i+self.num_of_vars] = np.identity(self.num_of_vars)
            idx += 1

        eigenvector, _ = np.linalg.eig(stability_matrix)

        return True if np.all(np.abs(eigenvector) < 1) else False

    def plot_data(self, obs_data: np.array, dim: tuple, dir: str = "") -> None:
        """ Plots observational data. 

                :param obs_data: 
                    array consisting of observations per timestep
                :param dim: 
                    dimensionality for subplots
                :param dir: 
                    directory for saving plot
        """

        T, d = obs_data.shape
        n, m = dim
        idxs = np.arange(1, T + 1, 1)
        fig, axes = plt.subplots(n, m)
        fig.tight_layout()

        k = 0

        for i in range(n):
            for j in range(m):
                if k < d:
                    axes[i, j].set_title(f"$X_{k+1}$ for $t=1,...,{T}$")
                    axes[i, j].set_xlabel(f"$T$")
                    axes[i, j].set_ylabel(f"$X_{k+1}$")
                    axes[i, j].plot(idxs, obs_data[:, k], alpha=0.85,
                                    color="slategrey")
                k += 1

        if dir:
            plt.savefig(dir)
        else:
            plt.show()