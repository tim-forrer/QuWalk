import numpy as np
import quantum_custom.walks.continuous as cont
import quantum_custom.core as core
from collections import defaultdict
import time

n = 4
N = 2**n
gamma = 1 / n
origin = 0

adj_mat = cont.adj_mat(n, graph = "H")

H = cont.H(adj_mat, gamma)

state_0 = np.zeros(N)
state_0[origin] = 1
state_t = cont.state_t(state_0, H, 2 * n * np.pi / 4)
prob_amps = cont.measure(state_t)

xs = np.arange(0, N)
groups = [i for i in range(n + 1)]
grouped_probs = [0 for _ in range(n + 1)]

for i, x in enumerate(xs):
    ham_dist = cont.hamming_dist(x, origin)
    print(ham_dist)
    grouped_probs[ham_dist] += prob_amps[i]

core.plot([groups], [grouped_probs], title = "Cont. Time QW on Hypercube.", labels = [r"Cont. Time Walk, $\gamma = 0.25, t = 2\pi$."], axis_labels = [r"Hamming distance from origin of walk, $|0000\rangle$", "Probability"])