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
    """A matrix that representing a map from each x_n basis vector to the x_(n+1) (mod d) basis vector.

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
    """A matrix that representing a map from each x_n basis vector to the x_(n-1) (mod d) basis vector.

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

def coin(eta, delta = 0):
    amp = np.sqrt(eta)
    neg_amp = np.sqrt(1 - eta)
    matrix = np.ones((2,2))
    matrix[0,0] = amp
    matrix[1,1] = -amp
    matrix[0,1] = neg_amp
    matrix[1,0] = neg_amp
    return matrix


d = 9
t = 30

pos0 = np.zeros((d, 1))
pos0[0] = 1 # start at vertex 0

h_state0_up = np.kron(pos0, SPIN_UP)
h_state0_down = np.kron(pos0, SPIN_DOWN)

total_state0_summand1 = np.kron(h_state0_up, h_state0_up)
total_state0_summand2 = np.kron(h_state0_down, h_state0_down)

total_state0 = (1/2**0.5) * (total_state0_summand1 + total_state0_summand2) # Initial total state

# Create operators.
# Walk operator for the first walk.
h1_coin_op = coin(1/3)
h1_walk_operator = disc.eye_kron(d, h1_coin_op)
h1_walk_operator = np.dot(np.kron(shift_plus(d), H00) + np.kron(shift_minus(d), H11), h1_walk_operator)

# Walk operator for the second walk.
h2_coin_op = coin(1/7)
h2_walk_operator = disc.eye_kron(d, h2_coin_op)
h2_walk_operator = np.dot(np.kron(shift_plus(d), H00) + np.kron(shift_minus(d), H11), h2_walk_operator)

# Total walk operator.
total_walk_operator = np.kron(h1_walk_operator, h2_walk_operator)

for i in range(t + 1):
    total_statet = np.dot(np.linalg.matrix_power(total_walk_operator, i), total_state0)
    total_rhot = np.outer(total_statet, total_statet)
    reduced_rhot = partial_trace(total_rhot, (d,2,d,2), 1)
    reduced_rhot = partial_trace(reduced_rhot, (d,d,2), 2)
    reduced_rhot = partial_trace(reduced_rhot, (d,d), 0)
    print(entanglement(reduced_rhot, base = d))