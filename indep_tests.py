import os, scipy, tigramite
import numpy as np
from tigramite.independence_tests.parcorr import ParCorr as pc
from tigramite.independence_tests.gpdc import GPDC as gpr

class Oracle:
    """ Parent class for independence oracles. In all child classes, we test X _|_ Y | Z using
        Zhang et al.'s correspondence X _|_ Y | Z <=> r_X _|_ r_Y for residuals r_X, r_Y defined as
        r_X = X - f_X(Z) and r_Y = Y - f_Y(Z) for regressors f_X and f_Y. Zhang et al.'s article  
        is found here: https://ojs.aaai.org/index.php/AAAI/article/view/10698."""
    
    def __init__(self):
        pass

    def get_combined_pvalues(self, pvalues: np.array, method: str) -> float:
        """ Given p-values p_1, ..., p_n for corresponding hypotheses H_1, ..., H_n, computes 
            p-value for the conjunctive hypothesis H_1 & ... & H_n.

                :param pvalues: 
                    vector of p-values p_1, ..., p_n for hypotheses H^1, ..., H^n
                :param method: 
                    method used to combine p-values

                :returns pvalue: 
                    p-value for the conjunctive hypothesis

        """

        if method == "harmonic":
            pvalue = scipy.stats.hmean(a=pvalues)
        elif method == "fisher" or method == "stouffer":
            pvalue = scipy.stats.combine_pvalues(
                pvalues=pvalues, method=method)[1]
        else:
            raise ValueError("Input method with name \"{method}\" not available!")

        return pvalue

class PartialCorrelation(pc, Oracle):
    """ Conditional independence oracle based on partial correlation, which 
        assumes a linear parametrisation on the regressors f_X and f_Y."""

    def __init__(self):
        super().__init__()

    def get_significance_value(self, obs_data : np.array, T : int, d : int, xyz: np.array = None) -> float:
        """ Computes significance value for null hypothesis r_X _|_ r_Y.

                :param obs_data: 
                    observations matrix
                :param T: 
                    number of time indices
                :param d: 
                    number of time series elements
                :param xyz: 
                    mandatory argument to Runge's function, 
                    no effect on computation of p-value 

                :returns p_value: 
                    p-value for null hypothesis r_X _|_ r_Y
        """

        value = self.get_dependence_measure(array = obs_data, xyz = xyz)
        p_value = self.get_analytic_significance(value = value, T = T, dim = d)

        return p_value

class GaussianProcessRegression(gpr, Oracle):
    """ Conditional independence oracle based on Gaussian Process Regression, which
        is a non-parametric method for estimating f_X and f_Y. """

    def __init__(self):
        super().__init__()

    def get_significance_value(self, obs_data : np.array, T : int, d : int, xyz : np.array = None) -> float:
        """ Computes significance value for null hypothesis r_X _|_ r_Y.

                :param obs_data: 
                    observations matrix
                :param T: 
                    number of time indices
                :param d: 
                    number of time series elements
                :param xyz: 
                    mandatory argument to Runge's function, 
                    no effect on computation of p-value 

                :returns p_value: 
                    p-value for null hypothesis r_X _|_ r_Y
        """

        value = self.get_dependence_measure(array = obs_data, xyz = xyz)
        p_value = self.get_analytic_significance(value = value, T = T, dim = d)

        return p_value

