import quantum_custom.core as core
import quantum_custom.walks.discrete as disc
import scipy.linalg as la
import numpy as np
from math import log, log2, e

N_steps = 100

def entanglement(matrix, base = e):
    """
    Computes the entanglement of a density matrix. A value of 0 (or extremely close to) indicates a seperable state.

    Parameters
    ----------
    matrix : numpy.ndarray
        The (reduced) density matrix that represents a quantum state, of which the entanglement is to be quantified.
    base : int or float or numpy.float64
        The base of the log. Typically used to change the base to the dimension of Hilbert space, in which case an entanglement of 1 indicates the state is maximally entangled.
    
    Returns
    -------
    entang : numpy.float64
        The entanglement of the given matrix.
    """
    eigenvals = la.eig(matrix)[0].real
    entang = 0
    for eigenval in eigenvals:
        if eigenval < 10**-15: # Sometimes eigenvalues that should be zero come out as negative (eigenvalues of density matrices are non-negative). Fixes this. Also skips over eigenvalues that are 0.
            continue
        entang += -eigenval * log(eigenval) / log(base)
    return entang

def partial_trace(matrix, dims, trace_over):
    """
    Computes the partial trace of a matrix. See https://scicomp.stackexchange.com/questions/27496/calculating-partial-trace-of-array-in-numpy for in depth explanation.

    Parameters
    ----------
    matrix : numpy.ndarray
        The matrix over which the partial trace is to be computed.
    dims : tuple
        The dimensions of the Hilbert subspaces of which the overall Hilbert space is a product of.
    trace_over : int
        The index of the Hilbert space which is to be traced over. If Hilbert Space = A x B x C, then "1" would mean that the partial trace over subspace B is taken.
    
    Returns
    -------
    result : numpy.ndarray
        The reduced matrix that has had the desired Hilbert space traced out.
    """
    tensor = matrix.reshape(dims + dims)
    result = np.trace(tensor, axis1 = trace_over, axis2 = trace_over + len(dims))
    resultant_space_dim = 1
    for i in result.shape[:len(result.shape) // 2]:
        resultant_space_dim *= i
    result = result.reshape(resultant_space_dim, resultant_space_dim)
    return result

spin0 = core.SPIN_UP
spin0 = np.array([1,0])
x = np.arange(1, 100)
y = []
for N in x:
    state0 = disc.state0(spin0, N)
    walk_operator = disc.walk_operator(N)

    statef = np.linalg.matrix_power(walk_operator, N).dot(state0.state)

    state = core.QuantumState(statef, (2, 2 * N + 1))
    rho = partial_trace(state.rho, state.H_space_dims, 0)
    y.append(entanglement(rho, base = 2))

core.plot([x], [y], "Entanglement evolution of Discrete Quantum Walk", [], axis_labels=["Time Steps", "Entanglement"])