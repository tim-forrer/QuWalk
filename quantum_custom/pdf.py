import numpy as np
from scipy.special import comb

from quantum_custom.core import spin_down, spin_up, H00, H11, H, QuantumState
import quantum_custom.walks.discrete as disc
import quantum_custom.walks.classical as classc
import quantum_custom.walks.continuous as cont

import matplotlib.pyplot as plt
import matplotlib.animation as animation

class PlotData():
    def __init__(self, x, y, N):
        self.x = x
        self.y = y
        self.N = N

def classc_pdf(x, N):
    """
    The probability distribution function of a classical walk after N coin flips.

    Returns an PlotData() instance.
    """


def disc_pdf(x, N, spin0):
    """
    The probability distribution function of a Hadamard coined discrete quantum walk after N quantum coin flips.

    Returns an PlotData() instance.

    Removes zero probability positions by default.
    """
    positions = 2 * N + 1

    #initial conditions
    position0 = np.zeros(positions)
    position0[N] = 1
    state0 = np.kron(np.matmul(H, spin0), position0) #initial state in complete Hilbert space is Hadamard acting on initial state, tensor product with the initial position
    quantum_state = QuantumState(state0)
    walk_operator = disc.walk_operator(N)

    #conduct walk
    quantum_state.state = np.linalg.matrix_power(walk_operator, N + 1).dot(quantum_state.state) #fudge
    probs = disc.prob(quantum_state.state, N)

    x = x[1::2]
    probs = probs[1::2]

    return PlotData(x, probs, N)

def cont_pdf(x, N):
    """
    The probability distribution function of a continuous quantum walk after N units of time.

    Returns an PlotData() instance.
    """
    return PlotData(x, cont.prob(x, N), N)