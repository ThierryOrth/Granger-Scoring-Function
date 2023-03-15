
import numpy as np
import matplotlib.pyplot as plt

class DSCM:
    """ Defines a Dynamic Structural Causal Model.

            :param links: parameter dictionary that defines causal influence

        Note: this format for DSCMs is similar to Jakob Runge's format, but slightly differs due to the simplifying 
              assumption that mappings are defined over the linear sum of causes instead over causes separately. 

    """
    def __init__(self, links : dict, mapping):
        self.links = links
        self.num_of_vars = len(self.links)
        self.mapping = mapping
        self.causes, self.lags, self.coeffs = self.dictionarise()
        self.max_lag = min(lag for lags in self.lags.values() for lag in lags)

    def __str__(self):
        return "\n".join(f'X^t_{i}={"+".join([f"{coeff}*X^{lag}_{cause}" for cause, lag, coeff in zip(self.causes[i], self.lags[i], self.coeffs[i])])}' \
                                                            for i, _ in enumerate(self.links))
    
    def dictionarise(self) -> tuple:
        """ Dictionarise causes, lags, coefficients and functions for easy retrieval.
        
                :returns effect_to_cause: map from effects to causes

                :returns effect_to_lag: map from effects to time lags of causes

                :returns effect_to_coeffs: map from effects to coefficients of causes

        """
        effect_to_cause, effect_to_lag, effect_to_coeffs = dict(), dict(), dict()

        for effect, causes in self.links.items():        
            effect_to_cause.update({effect : [cause for (cause, _), _ in causes]})
            effect_to_lag.update({effect : [lag for (_, lag),_ in causes]})
            effect_to_coeffs.update({effect : [coeff for (_,_), coeff in causes]})

        return effect_to_cause, effect_to_lag, effect_to_coeffs
    
    def check_stationarity(self) -> bool:
        """ Code and comments in this function are a slight adaption from code by Jakob Runge, source: 
                https://github.com/jakobrunge/tigramite/blob/c877f0b578c357b1c26f666764be07676193d4dc/tigramite/toymodels/structural_causal_processes.py#L865
        
        Returns stationarity according to a unit root test. Assumes an at least asymptotically linear vector autoregressive process without contemporaneous links.

            :returns stationary: indicates if VAR process is stationary 

        """
        idx = 0
        max_lag = abs(self.max_lag)
        graph = np.zeros((self.num_of_vars, self.num_of_vars, max_lag))
        stability_matrix = np.zeros((self.num_of_vars * max_lag, self.num_of_vars * max_lag))
        funcs = []

        for j in range(self.num_of_vars):
            func = self.mapping

            for link in self.links[j]:
                (var, lag), coeff = link

                if 0 < abs(lag):
                    graph[j, var, abs(lag) - 1] = coeff

                funcs.append(func)
    
        for i in range(0, self.num_of_vars*max_lag, self.num_of_vars):
            stability_matrix[:self.num_of_vars,i:i+self.num_of_vars] = graph[:, :, idx]
            if idx < max_lag - 1:
                stability_matrix[i+self.num_of_vars:i+2*self.num_of_vars, i:i+self.num_of_vars] = np.identity(self.num_of_vars)
            idx += 1

        eigenvector, _ = np.linalg.eig(stability_matrix)

        return True if np.all(np.abs(eigenvector) < 1) else False
    
    def generate_obs_data(self, T : int, set_off_gaussian : bool = True, gaussian_noise : bool = True, random_state: int = 42) -> np.array:
        """ Generates observational data over the interval {1, ..., T}.

            :param set_off_gaussian: switch for initialising previous timesteps with Gaussian noise N(0,1) 

            :param gaussian_noise: switch for adding normally distributed noise N(0,1) 

            :param random_state: facilitate reproducibility of results

            :returns obs_data: array consisting of observational data, with rows corresponding to time indices and columns to time series elements 

        """

        np.random.seed(random_state)

        start = abs(self.max_lag)
        obs_data = np.zeros((start + T, self.num_of_vars))

        if set_off_gaussian:
            obs_data[:start, :] = np.random.normal(0, 1, size=(start, self.num_of_vars))
        
        for t in range(start, start+T):
            for j in range(self.num_of_vars):
                noise = np.random.normal(0.0, 1.0) if gaussian_noise else 0.0
                obs_data[t, j] = self.mapping(np.dot(self.coeffs[j], 
                                            obs_data[t + np.array(self.lags[j]), self.causes[j]])
                                                        ) + noise
        return obs_data[start:, :]
    
    def get_adjacency_matrices(self, k : int = 0) -> tuple:
        """ Retrieve adjacency matrices for lagged and contemporaneous causal relations. In both matrices,
            the i'th row corresponds to cause i whilst the j'th column corresponds to effect j. 

                :param k: determines where to zero the lagged adjacency matrix

                :returns lag_matrix: adjacency matrix of size d*gamma x d*gamma for lagged causal relations   

                :returns instant_matrix: adjacency matrix of size d x d for contemporaneous causal relations 

        """
        
        gamma = abs(self.max_lag)+1
        dim = self.num_of_vars * gamma
        lag_matrix = np.zeros((dim,dim))
        instant_matrix = np.zeros((self.num_of_vars, self.num_of_vars)) 

        for j in range(0, self.num_of_vars):
            lags = np.array(self.lags[j])
            causes = np.array(self.causes[j])
            instant_cause_idxs = causes[lags == 0]
            rel_cause_idxs = np.add(np.multiply(self.num_of_vars, lags[lags<0]), 
                                        np.subtract(causes[lags<0], j))
        
            for t in range(1, gamma):
                ### compute cause and effect indices ###
                lagged_effect_idx = self.num_of_vars*t+j
                lagged_cause_idxs = np.add(rel_cause_idxs, lagged_effect_idx)

                ### fill adjacency matrices ### 
                lag_matrix[lagged_cause_idxs, lagged_effect_idx] = 1
                instant_matrix[instant_cause_idxs, j] = 1

        ### zero lagged matrix to ensure that causes precede effects ###
        lag_matrix = np.triu(lag_matrix, k = k)

        return lag_matrix, instant_matrix

    def plot_data(self, obs_data : np.array, dim : tuple, dir : str = "") -> None:
        """ Plots observational data. 
        
                :obs_data: array consisting of observations per timestep

                :dim: dimensionality for subplots

                :dir: stringified directory for saving plot

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

    links_coeffs = {
                        0 : [((0, -1), 0.7), ((1, 0), -0.8)],
                        1 : [((1, -1), 0.8), ((3, 0), 0.8)],
                        2 : [((2, -1), 0.5), ((1, -2), 0.5), ((3, 0), 0.6)],
                        3 : [((3, -1), 0.4)],
                    }

    dscm = DSCM(links=links_coeffs, mapping = g)
    obs_data = dscm.generate_obs_data(T=1000)
    dscm.plot_data(obs_data, dim=(2,2))
    lag_matrix, cont_matrix = dscm.get_adjacency_matrices()

   