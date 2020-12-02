import quantum_custom.walks.discrete as disc
from quantum_custom.core import *
import numpy as np
import time

def cyclic_graph(d):
    """
    The matrix representation of an (undirected) cyclic graph with specified dimension.

    Parameters
    ----------
    d : int
        The dimention of the cyclic graph.
    
    Returns
    -------
    matrix : numpy.ndarray
        The matrix representation of the cyclic graph.
    """
    matrix = shift_plus(d) + shift_minus(d)
    return matrix

def shift_plus(d):
    """
    A matrix that representing a map from each x_n basis vector to the x_(n+1) (mod d) basis vector.

    Parameters
    ----------
    d : int
        The dimension of the vector space.

    Returns
    -------
    matrix : numpy.ndarray
        The matrix representation of the map.
    """
    matrix = np.roll(np.identity(d), 1, 0)
    return matrix

def shift_minus(d):
    """
    A matrix that representing a map from each x_n basis vector to the x_(n-1) (mod d) basis vector.

    Parameters
    ----------
    d : int
        The dimension of the vector space.

    Returns
    -------
    matrix : numpy.ndarray
        The matrix representation of the map.
    """
    matrix = np.roll(np.identity(d), -1, 0)
    return matrix

def coin(eta):
    amp = np.sqrt(eta)
    neg_amp = np.sqrt(1 - eta)
    matrix = np.ones((2,2))
    matrix[0,0] = amp
    matrix[1,1] = -amp
    matrix[0,1] = neg_amp
    matrix[1,0] = neg_amp
    return matrix

d = 6
t = 6

coin_op = coin(1/3)
print(coin_op)

walk_operator = disc.eye_kron(d, coin_op)
walk_operator = np.dot(np.kron(shift_plus(d), H00) + np.kron(shift_minus(d), H11), walk_operator)

state0 = np.kron([1,0,0,0,0,0], SPIN_UP)
print(state0)
statet = np.dot(np.linalg.matrix_power(walk_operator, t), state0)
print(statet)