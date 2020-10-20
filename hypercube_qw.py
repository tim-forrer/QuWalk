import numpy as np
import quantum_custom.walks.continuous as cont
import quantum_custom.core as core
from collections import defaultdict
import time

gamma = 1
N = 4
origin = 7

adj_mat = cont.adj_mat(N, graph = "H")

H = cont.H(adj_mat, gamma)

state_0 = np.zeros(2**N)
state_0[origin] = 1
state_t = cont.state_t(state_0, H, np.pi / 4)
prob_amps = cont.measure(state_t)

xs = np.arange(0, 2**N)
groups = [i for i in range(N + 1)]
grouped_probs = [0 for _ in range(N + 1)]

for i, x in enumerate(xs):
    ham_dist = cont.hamming_dist(x, origin)
    grouped_probs[ham_dist] += prob_amps[i]

core.plot([groups], [grouped_probs], title = "Cont. Time QW on Hypercube.", labels = ["Cont. Time Walk"])