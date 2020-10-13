from scipy.special import jv as bess #first kind Bessel function
import numpy as np
import matplotlib.pyplot as plt

def prob(x, t):
    """
    Returns probability of being at position x for continuous quantum walk on the number line at time t.
    Assumes that the walk starts from the origin.
    """
    bess_coeff = bess(x, t)
    return np.abs((-1j)**x * bess_coeff)**2