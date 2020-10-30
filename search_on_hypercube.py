import numpy as np
import quantum_custom.walks.continuous as cont
import quantum_custom.core as core
from collections import defaultdict
from scipy.special import comb
import time

def prob_success(n):
    """
    For n qubits, gives the peak probability of search success.
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
    t = np.sqrt(N) * 0.5 * np.pi # Time for first maximum overlap
    statet = cont.state_t(state0, qws_H, t)
    probs = cont.measure(statet)
    if 0.9999 <= sum(probs) <= 1.0001: 
        return max(probs)
    else:
        print("error")
        return

x = np.arange(1, 13)
probs = []
for n in x:
    probs.append(prob_success(n))

core.plot([x], [probs], "QWS on Hypercube", labels = ["QWS"])