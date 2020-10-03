import numpy as np
from quantum_custom.constants import H00, H11, H

#define walk operators

def walk_operator(max_N):
    position_count = 2 * max_N + 1
    shift_plus = np.roll(np.eye(position_count), 1, axis = 0)
    shift_minus = np.roll(np.eye(position_count), -1, axis = 0)
    step_operator = np.kron(H00, shift_plus) + np.kron(H11, shift_minus)
    return step_operator.dot(np.kron(H, np.eye(position_count)))


def flip_once(state, max_N):
    """
    Flips the Hadamard coin once and acts on the given state appropriately.
    Returns the state after the Hadamard coin flip.
    """
    walk_op = walk_operator(max_N)
    next_state = walk_op.dot(state)
    return next_state

def get_prob(state, max_N):
    """
    For the given state, calculates the probability of being in any possible position.
    Returns an array of probabilities.
    """
    position_count = 2 * max_N + 1
    prob = np.empty(position_count)
    for k in range(position_count):
        posn = np.zeros(position_count)
        posn[k] = 1
        posn_outer = np.outer(posn, posn)
        alt_measurement_k = eye_kron(2, posn_outer)
        proj = alt_measurement_k.dot(state)
        prob[k] = proj.dot(proj.conjugate()).real       
    return prob

def eye_kron(eye_dim, mat):
    """
    Speeds up the calculation of the tensor product of an identity matrix of dimension eye_dim with a given matrix.
    This exploits the fact that majority of values in the resulting matrix will be zeroes apart from on the leading diagonal where we simply have copies of the given matrix.
    Returns a matrix.
    """
    mat_dim = len(mat)
    result_dim = eye_dim * mat_dim #dimension of the resulting matrix
    result = np.zeros((result_dim, result_dim))
    result[0:mat_dim, 0:mat_dim] = mat
    result[mat_dim:result_dim, mat_dim:result_dim] = mat
    return result