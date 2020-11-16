import numpy as np
import quantum_custom.walks.continuous as cont
import quantum_custom.core as core
from collections import defaultdict
from scipy.special import comb
from scipy.optimize import fmin
import time

def prob_success(t, n):
    """
    For n qubits, gives the peak probability of search success at time t.
    """
    N = 2**n

    # Setup initial conditions.
    # Setup the target vertex.
    target_vertex = 0
    target_state = np.zeros(N)
    target_state[target_vertex] = 1
    state0 = (1 / np.sqrt(N)) * np.ones(N) # Equal superposition of all vertices.
    gamma = 0
    for r in range(1, n + 1):
        gamma += (1 / N) * comb(n, r) * (1 / r) # Optimal choice of gamma, see equation 10 https://arxiv.org/pdf/1709.00371.pdf.

    qws_H = cont.search_H(n, target_state, gamma, "H") # QWS Hamiltonian on hypercube of dimension N with target_state = |target_vertex>.

    # Perform search.
    
    statet = cont.state_t(state0, qws_H, t)
    probs = cont.measure(statet)
    return max(probs)

max_n = 10

x_arr = np.arange(1, max_n + 1)
probs = []
for n in x_arr:
    N = 2**n

    t_guess = (np.pi / 2) * np.sqrt(N)

    _, max_prob, _, _, _ = fmin(lambda t: -prob_success(t, n), t_guess, full_output=True)
    probs.append(-max_prob)

core.plot([x_arr], [probs], title = "Peak Probability of Success against qubit number", labels = [], axis_labels = ["Number of Qubits", r"$P_{peak}$"])