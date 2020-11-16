import numpy as np
from quantum_custom.core import H00, H11, H, QuantumState, PlotData


def walk_operator(max_N):
    """
    An matrix that represents the action of the walk operator upon a quantum state.

    Parameters
    ----------
    max_N : int
        The maximum number of steps that is to be taken on the quantum walk.

    Returns
    -------
    step_operator : numpy.ndarray
        The matrix representation of the walk operator.
    """
    position_count = 2 * max_N + 1
    shift_plus = np.roll(np.eye(position_count), 1, axis = 0)
    shift_minus = np.roll(np.eye(position_count), -1, axis = 0)
    step_operator = np.kron(H00, shift_plus) + np.kron(H11, shift_minus)
    return step_operator.dot(np.kron(H, np.eye(position_count)))


def flip_once(state, N):
    """
    "Flips" the Hadamard coin once and acts on the given state appropriately.
    Returns the state after the Hadamard coin flip.
    """
    operator = walk_operator(N)
    next_state = operator.dot(state)
    return next_state

def prob(state, N):
    """
    For the given state, calculates the probability of being in any possible position.
    Returns an array of probabilities.
    """
    position_count = 2 * N + 1
    prob = np.empty(position_count)
    for k in range(position_count):
        posn = np.zeros(position_count)
        posn[k] = 1
        posn_outer = np.outer(posn, posn)
        alt_measurement_k = eye_kron(2, posn_outer)
        proj = alt_measurement_k.dot(state)
        prob[k] = proj.dot(proj.conjugate()).real       
    return prob

def eye_kron(eye_dim, mat, reverse = False):
    """
    Calculates the tensor product of the identity matrix with a matrix (in that order).
    This exploits the fact that majority of values in the resulting matrix will be zeroes apart from on the leading diagonal where we simply have copies of the given matrix.

    Parameters
    ----------
    eye_dim : int
        The dimension of the identity matrix for this tensor product.
    mat : numpy.ndarray
        The matrix which is to be tensored with the identity.
    reverse : bool
        Currently doesn't do anything. Future implementation will hopefully allow for the tensor product of M x I to be calculated efficiently by changing this parameter to True.

    Returns
    -------
    result : numpy.ndarray
        The resultant matrix from this calculation.
    """
    mat_dim = len(mat)
    result_dim = eye_dim * mat_dim # Dimension of the resulting matrix
    result = np.zeros((result_dim, result_dim))
    if reverse:
        pass # Awaiting implementation of reverse ordering.
    else:
        result[0:mat_dim, 0:mat_dim] = mat
        result[mat_dim:result_dim, mat_dim:result_dim] = mat
    return result

def state0(spin0, N):
    """
    Initial coin and walker state (tensored in that order), where walker starts from origin.

    Returns QuantumState() instance
    """
    positions = 2 * N + 1
    position0 = np.zeros(positions)
    position0[N] = 1

    return QuantumState(np.kron(spin0, position0))

def stateN(state0, operator, N):
    """
    Acts the operator N times on the initial state.
    """
    return np.linalg.matrix_power(operator, N).dot(state0.state)


def pdf(x, N, spin0):
    """
    The probability distribution function of a Hadamard coined discrete quantum walk after N quantum coin flips.

    Returns PlotData() instance.

    Removes zero probability positions.
    """
    positions = 2 * N + 1

    #initial conditions
    position0 = np.zeros(positions)
    position0[N] = 1
    state0 = np.kron(spin0, position0) #initial state in complete Hilbert space is initial spin tensor product with the initial position.
    quantum_state = QuantumState(state0)
    operator = walk_operator(N)

    #conduct walk
    quantum_state.state = np.linalg.matrix_power(operator, N).dot(quantum_state.state)
    probs = prob(quantum_state.state, N)

    x = x[0::2]
    probs = probs[0::2]

    return PlotData(x, probs, N)