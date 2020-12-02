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
    walk_op : numpy.ndarray
        The matrix representation of the walk operator.
    """
    position_count = 2 * max_N + 1
    shift_plus = np.roll(np.eye(position_count), 1, axis = 0)
    shift_minus = np.roll(np.eye(position_count), -1, axis = 0)
    step_operator = np.kron(H00, shift_plus) + np.kron(H11, shift_minus)
    walk_op = step_operator.dot(np.kron(H, np.eye(position_count)))
    return walk_op


def time_step(state0, N):
    """
    Increments a discrete quantum walk by a singular time step.

    Parameters
    ----------
    state0 : QuantumState
        The input state that is to be incremented upon once.
    N : int
        The maximum number of steps that will be taken on the quantum walk.
    
    Returns
    -------
    statef : QuantumState
        The result state having incremented the quantum state.
    """
    operator = walk_operator(N)
    statef = QuantumState(operator.dot(state0.state), H_space_dims = state0.state)
    return statef

def prob(state, N):
    """
    Calculate the probability of being in each position.
    
    Parameters
    ----------
    state : QuantumState
        The state which is a superposition of many probabilities.
    N : int
        The number of time steps the state has been evolved by.

    Returns
    -------
    probs : numpy.ndarray
        An array of probabilities, with index position corresponding to the probability of being in that position. Index position 0 representing position -N.
    """
    position_count = 2 * N + 1
    probs = np.empty(position_count)
    for k in range(position_count):
        posn = np.zeros(position_count)
        posn[k] = 1
        posn_outer = np.outer(posn, posn)
        alt_measurement_k = eye_kron(2, posn_outer)
        proj = alt_measurement_k.dot(state)
        probs[k] = proj.dot(proj.conjugate()).real       
    return probs

def eye_kron(eye_dim, mat):
    """
    Calculates the tensor product of the identity matrix with a matrix (in that order).
    This exploits the fact that majority of values in the resulting matrix will be zeroes apart from on the leading diagonal where we simply have copies of the given matrix.

    Parameters
    ----------
    eye_dim : int
        The dimension of the identity matrix for this tensor product.
    mat : numpy.ndarray
        The matrix which is to be tensored with the identity.

    Returns
    -------
    result : numpy.ndarray
        The resultant matrix from this calculation.
    """
    mat_dim = len(mat)
    result_dim = eye_dim * mat_dim # Dimension of the resulting matrix
    result = np.zeros((result_dim, result_dim))
    index_partitions = [i for i in np.arange(0, result_dim + 1, mat_dim)]
    for i in range(len(index_partitions) - 1):
        result[index_partitions[i]:index_partitions[i + 1], index_partitions[i]:index_partitions[i + 1]] = mat
    return result

def state0(spin0, N):
    """
    Initial coin and walker state (tensored in that order), where the walker starts from origin.

    Parameters
    ----------
    spin0 : numpy.ndarray
        The initial spin of the state.
    N : int
        The maximum number of time steps that the state will be evolved by.

    Returns
    -------
    state0 : QuantumState
        The initial QuantumState corresponding to the input parameters. Starts from the origin.
    """
    positions = 2 * N + 1
    position0 = np.zeros(positions)
    position0[N] = 1
    state0 = QuantumState(np.kron(spin0, position0))
    return state0

def stateN(state0, operator, N):
    """
    Acts a given operator on a quantum state N times.

    Parameters
    ----------
    state0 : QuantumState
        The initial state that is to be acted upon.
    Operator : numpy.ndarray
        The operator to act upon the given initial state.
    
    Returns
    -------
    statef : QuantumState
        The result of acting the operator on the initial state N times.
    """
    statef = QuantumState(np.linalg.matrix_power(operator, N).dot(state0.state), H_space_dims = state0.H_space_dims)
    return statef


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