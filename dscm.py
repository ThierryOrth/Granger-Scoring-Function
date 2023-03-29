import numpy as np
import matplotlib.pyplot as plt

#TODO: improve comments

class DSCM:
    """ Defines a Dynamic Structural Causal Model.

            :param links: 
                set of parameters describing causal influence
            :param map: 
                transformation applied to linear sum of causal influences

    """
    def __init__(self, links : dict, f : callable):
        self.links = links
        self.num_of_vars = len(self.links)
        self.f = f
        self.causes, self.lags, self.coeffs = self.dictionarise()
        self.max_tau = abs(min(lag for lags in self.lags.values() for lag in lags))
    
    def dictionarise(self) -> tuple:
        """ Dictionarise causes, lags, coefficients and functions for easy retrieval.
        
                :returns effect_to_cause: 
                    mapping from effects to causes
                :returns effect_to_lag: 
                    mapping from effects to time lags of causes
                :returns effect_to_coeffs: 
                    mapping from effects to causal coefficients 

        """
        effect_to_cause, effect_to_lag, effect_to_coeffs = dict(), dict(), dict()

        for effect, causes in self.links.items():        
            effect_to_cause.update({effect : [cause for (cause, _), _ in causes]})
            effect_to_lag.update({effect : [lag for (_, lag),_ in causes]})
            effect_to_coeffs.update({effect : [coeff for (_, _), coeff in causes]})

        return effect_to_cause, effect_to_lag, effect_to_coeffs
    
    def generate_obs_data(self, T : int, gauss_init : bool = True, gauss_noise : bool = True, random_state: int = 42, log_transform : bool = True) -> np.array:
        """ Generates observational data sample over the time interval {1, ..., T} 
            for each time series in the DSCM.

            :param gauss_init: 
                Boolean for initialising previous timesteps with standard Gaussian noise
            :param gauss_noise: 
                Boolean for additive standard Gaussian noise
            :param random_state: 
                insert for result reproducibility

            :returns obs_data:
                array of observations of shape T x d, with d the number of time series
        """

        np.random.seed(random_state)
        obs_data = np.zeros((self.max_tau + T, self.num_of_vars))

        if gauss_init:
            obs_data[:self.max_tau, :] = np.random.normal(0, 1, size=(self.max_tau, self.num_of_vars))
        
        for t in range(self.max_tau, self.max_tau+T):
            for j in range(self.num_of_vars):
                noise = np.random.normal(0.0, 1.0) if gauss_noise else 0.0
                obs_data[t, j] = self.f(np.dot(self.coeffs[j], 
                                            obs_data[t + np.array(self.lags[j]), self.causes[j]])
                                                        ) + noise
                
        return np.cbrt(obs_data[self.max_tau:, :]) if log_transform else obs_data[self.max_tau:, :]
    
    def get_adjacency_matrices(self) -> np.array:
        """ Constructs adjacency matrices for contemporaneous and lagged causal relations from the DSCM
            parameters. On the assumption of causal stationarity, these matrices span the maximal time window
            defined as the absolute value of the maximal lag plus one.

                :returns adj_matrices: 
                    adjacency matrices with adj_matrices[tau, i, j] indicating whether X^{t-tau}_i -> X^t_j
        """

        adj_matrices = np.zeros((self.max_tau + 1, self.num_of_vars, self.num_of_vars)).astype(int)
        
        for effect in range(self.num_of_vars):
            for idx, cause in enumerate(self.causes[effect]):
                cause = self.causes[effect][idx]
                tau = self.lags[effect][idx]
                
                adj_matrices[abs(tau), cause, effect] = 1

        return adj_matrices

    def get_summary_graph(self, adj_matrices : np.array = None, dag_repr : bool = True) -> np.array:
        """ Constructs summary graph from adjacency matrices, defined as summary_graph[i,j] = 1 iff
            adj_matrices[i,j,tau] = 1 for some tau. 

                :param adj_matrices: 
                    adjacency matrices representing contemporaneous and lagged causes of DSCM
                :param dag_repr:
                    Boolean to convert summary graph into DAG representation
                
                :returns summary_graph:
                    summary graph induced by adjacency matrices
        """

        *_ , d = adj_matrices.shape
        summary_graph = np.zeros((d,d)).astype(int)

        for i in range(d):
            for j in range(d):
                summary_graph[i,j] = 1 if 0 < np.sum(adj_matrices[:, i, j]) else 0

        if dag_repr:
            np.fill_diagonal(summary_graph, 0)

        return summary_graph
    
    def check_stationarity(self) -> bool:
        """ Code and comments in this function are a slight adaption from code by Jakob Runge, source: 
                https://github.com/jakobrunge/tigramite/blob/c877f0b578c357b1c26f666764be07676193d4dc/tigramite/toymodels/structural_causal_processes.py#L865
        
        Returns stationarity according to a unit root test. Assumes an at least 
        asymptotically linear vector autoregressive process without contemporaneous links.
            
            :returns stationary: 
                indicates if VAR process (DSCM) is stationary 
        """
        
        graph = np.zeros((self.num_of_vars, self.num_of_vars, self.max_tau))
        stability_matrix = np.zeros((self.num_of_vars * self.max_tau, self.num_of_vars * self.max_tau))
        funcs = []

        for j in range(self.num_of_vars):

            for link in self.links[j]:
                (var, lag), coeff = link

                if 0 < abs(lag):
                    graph[j, var, abs(lag) - 1] = coeff

                funcs.append(self.f)

        idx = 0

        for i in range(0, self.num_of_vars*self.max_tau, self.num_of_vars):
            stability_matrix[:self.num_of_vars,i:i+self.num_of_vars] = graph[:, :, idx]
            if idx < self.max_tau - 1:
                stability_matrix[i+self.num_of_vars:i+2*self.num_of_vars, i:i+self.num_of_vars] = np.identity(self.num_of_vars)
            idx += 1

        eigenvector, _ = np.linalg.eig(stability_matrix)

        return True if np.all(np.abs(eigenvector) < 1) else False
    
    def plot_data(self, obs_data : np.array, dim : tuple, dir : str = "") -> None:
        """ Plots observational data. 
        
                :param obs_data: 
                    array consisting of observations per timestep
                :param dim: 
                    dimensionality for subplots
                :param dir: 
                    directory for saving plot
        """

        T, d = obs_data.shape
        n,m = dim
        idxs = np.arange(1, T+1, 1)
        fig, axes = plt.subplots(n, m)
        fig.tight_layout() 
        k = 0

        for i in range(n):
            for j in range(m):
                if k<d:
                    axes[i,j].set_title(f"$X_{k+1}$ for $t=1,...,{T}$")
                    axes[i,j].set_xlabel(f"$T$")
                    axes[i,j].set_ylabel(f"$X_{k+1}$")
                    axes[i,j].plot(idxs, obs_data[:, k], alpha = 0.85, \
                                                    color = "slategrey") 
                k+=1

        if dir:
            plt.savefig(dir)
        else:
            plt.show()

if __name__ == "__main__":
    f = lambda x : x
    g = lambda x : (1-4*np.e**(-x**2/2))*x
    h = lambda x : (1-4*x**3 * np.e**(-x**2/2))*x

    links = {
                0 : [((0, -1), 0.7), ((1, 0), -0.8)],
                1 : [((1, -1), 0.8), ((3, 0), 0.8)],
                2 : [((2, -1), 0.5), ((1, -2), 0.5), ((3, 0), 0.6)],
                3 : [((3, -1), 0.4)],
            }

    dscm = DSCM(links=links, map = g)
    obs_data = dscm.generate_obs_data(T=1000)
    # dscm.plot_data(obs_data, dim=(2,2))
    adjacency_matrices = dscm.get_adjacency_matrices()
    summary_graph = dscm.get_summary_graph()
    print(summary_graph)