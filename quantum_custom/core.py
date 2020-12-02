import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from math import e, log
import numpy.linalg as la

class QuantumState:
    """
    The QuantumState object is used to store information regarding a specific quantum state.

    Parameters
    ----------
    state : numpy.ndarray
        This is the mean parameter in defining the QuantumState object and is the only compulsory argument. It's a numpy array in an arbitrary basis that should be kept consistent by the user.
    H_space_dims: None or tuple
        If this state exists in a Hilbert space that is the product of subspaces, stores the dimensions of those subspaces. Defaults to the length of the state when not specified.
    """
    def __init__(self, state, H_space_dims = None):
        self.state = state
        if H_space_dims == None:
            self.H_space_dims = (len(state),)
        else:
            self.H_space_dims = H_space_dims
        self.conj = np.conjugate(state)
        self.rho = np.outer(state, self.conj) # Density matrix of this state.
    
    def tensor(self, state2):
        """
        Computes the tensor (Kronecker) product of this QuantumState with another one (in that order).

        Parameters
        ----------
        state2 : QuantumState
            The other state which is to be tensor producted with this state.

        Returns
        -------
        result : QuantumState
            A QuantumState which is the result of the tensor product.
        """
        result = QuantumState(np.kron(self.state, state2.state), H_space_dims = self.H_space_dims + state2.H_space_dims)
        return result

    def state_entanglement(self):
        """
        Quantifies how entangled this state is.

        Returns
        -------
        result : numpy.float64
            The quantified entanglement of the state in the desired subspace.
        """
        rho = self.rho
        reduced_rho = partial_trace(rho, self.H_space_dims, 0)
        entang = entanglement(reduced_rho)
        return entang

# Define PlotData class.
class PlotData():
    def __init__(self, x, y, N):
        self.x = x
        self.y = y
        self.N = N

def entanglement(matrix, base = e):
    """
    Computes the entanglement of a state represented by a density matrix. A value of 0 (or extremely close to) indicates a seperable state.

    Parameters
    ----------
    matrix : numpy.ndarray
        The (reduced) density matrix that represents a quantum state, of which the entanglement is to be quantified.
    base : int or float or numpy.float64
        The base of the log. Typically used to change the base to the dimension of Hilbert space, in which case an entanglement of 1 indicates the state is maximally entangled.
    
    Returns
    -------
    entang : numpy.float64
        The entanglement of the state represented by the given matrix.
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
    return result.reshape(resultant_space_dim, resultant_space_dim)

# Define spin up and spin down vectors in standard basis.
SPIN_UP = np.array([1,0])
SPIN_DOWN = np.array([0,1])

# Define the Hadamard operator, H, in the standard basis, in terms of ith, jth entries, Hijj
H00 = np.outer(SPIN_UP, SPIN_UP)
H01 = np.outer(SPIN_UP, SPIN_DOWN)
H10 = np.outer(SPIN_DOWN, SPIN_UP)
H11 = np.outer(SPIN_DOWN, SPIN_DOWN)
H = (H00 + H01 + H10 - H11)/np.sqrt(2.0) # Matrix representation of Hadamard gate in standard basis.

def plot(xs, ys, title, labels, axis_labels = ["x", "Probability"]):
    """
    Plot x against y with given title and label.
    """
    _, ax = plt.subplots()

    for i, x in enumerate(xs):
        ax.scatter(x, ys[i], marker = "x")
        if len(labels) > 0:
            ax.plot(x, ys[i], label = labels[i])
        else:
            ax.plot(x, ys[i])

    ax.legend()

    plt.title(title)

    ax.xaxis.set_major_locator(MaxNLocator(integer=True)) # Forces the x tick labels to be integer.

    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])

    plt.show()