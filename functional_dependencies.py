import numpy as np

f = lambda x : x
g = lambda x : (1-4*np.e**(-x**2/2))*x
h = lambda x : (1-4*x**3 * np.e**(-x**2/2))*x