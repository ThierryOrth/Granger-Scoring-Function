# https://arxiv.org/abs/1702.07007
import copy
import scipy
import dcor
import numpy as np
from sklearn import gaussian_process 

class Oracle:
    def __init__(self):
        pass

    def set_correct_dimensions(self, arr : np.array) -> np.array:
        """ 
                :param arr: array consisting observations for condition
        """

        arr_copy = copy.deepcopy(arr)
        
        if len(arr_copy.shape) == 1:
            arr_copy = arr_copy.reshape(-1, 1)
        elif len(arr_copy.shape) == 2:
            arr_copy = np.transpose(arr_copy)

        return arr_copy

class PartialCorrelation(Oracle):
    """ Conditional independence oracle based on
        partial correlation. """

    def __init__(self):
        super().__init__()

    # do residuals with lag, i.e., compute residuals for specific values only

    def get_residuals(self, feature_obs : np.array, target_obs : np.array) -> np.array:
        """ Get residuals from predicting Y = f(X) using the Least Squares Solution. 
        
                :param feature_obs: array consisting of features for prediction
                :return residuals: distances between observations and regression line
        """
    
        w_hat, *_ = np.linalg.lstsq(feature_obs, target_obs, \
                                                    rcond=None)
        target_pred = np.dot(feature_obs, w_hat)

        residuals = target_obs - target_pred

        return residuals

    def get_significance(self, X : np.array, Y : np.array, Z : np.array, T: int, d : int) -> float:
        """ Get significance values for H_0: r_X _|_ r_Y. """
        deg_f = T - d
        
        r_X = self.get_residuals(feature_obs = Z, \
                                    target_obs = X)
        r_Y = self.get_residuals(feature_obs = Z, \
                                    target_obs = Y)
        r_squared, *_ = scipy.stats.pearsonr(x = r_X, \
                                            y = r_Y)
        
        if np.isclose(abs(r_squared), 1.0):
            return 0.0
        
        t_statistic = r_squared * np.sqrt(deg_f / (1.0 - r_squared**2)) 
    
        p_value = scipy.stats.t.sf(np.abs(t_statistic), deg_f) #* 2

        return p_value
    
#TODO: understand and write code for GPR

# https://www.cs.cmu.edu/~neill/papers/TIST2015.pdf

class GaussianProcessRegression(Oracle):
    """ Conditional independence oracle based on
        Gaussian Process Regression. """
    def __init__(self):
        super().__init__()

    def get_residuals(self, X : np.array, Y : np.array) -> np.array:
        """ Get residuals from predicting Y = f(X) using a Gaussian process. """
        gp_regressor = gaussian_process.GaussianProcessRegressor().fit(X,Y)
        Y_hat = gp_regressor.predict(X)
        return Y - Y_hat

    def get_significance(self, X: np.array, Y: np.array, Z: np.array) -> float:
        """ Get significance values for H_0: r_X _|_ r_Y. """
        Z = self.set_correct_dimensions(Z)
        
        r_X = self.get_residuals(X=Z,\
                                Y=X)
        r_Y = self.get_residuals(X=Z,\
                                 Y=Y)
        
        p_value, *_ = dcor.independence.distance_correlation_t_test(x = r_X, \
                                                                    y = r_Y) 
        
        return p_value
