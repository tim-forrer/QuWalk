import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Create the QuantumState class
class QuantumState:
    """
    The QuantumState object is used to store information regarding a specific quantum state.

    Parameters
    ----------
    state : list or numpy.ndarray
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
        self.projector = np.outer(state, state) # Matrix that projects other states onto this state.
    
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

# Define PlotData class.
class PlotData():
    def __init__(self, x, y, N):
        self.x = x
        self.y = y
        self.N = N

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