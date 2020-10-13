import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from quantum_custom.core import spin_down, spin_up, H00, H11, H, QuantumState
import quantum_custom.walks.discrete as disc
import quantum_custom.walks.classical as classc
import quantum_custom.walks.continuous as cont

from scipy.special import comb
from math import factorial

#"coin flips"
max_N = 55
positions = 2 * max_N + 1

#initial conditions
initial_spin = 1/(np.sqrt(2)) * (spin_down - 1j * spin_up)
initial_position = np.zeros(positions)
initial_position[max_N] = 1
initial_state = np.kron(np.matmul(H, initial_spin), initial_position) #initial state is Hadamard acting on intial state, tensor product with the initial position
quantum_state = QuantumState(initial_state)
walk_operator = disc.walk_operator(max_N)

#conduct walk
quantum_state.state = np.linalg.matrix_power(walk_operator, max_N + 1).dot(quantum_state.state) #fudge
probs = disc.prob(quantum_state.state, max_N)

#create arrays to be plotted and remove 0 points
##discrete walk
x = np.arange(positions)
cleaned_x = x[1::2] - 1 #always start from index 1 since arange adds additional point?
cleaned_probs = probs[1::2]

##classical walk
classic_probs = []
cleaned_x -= max_N

for m in cleaned_x:
    classic_probs.append(classc.prob(m, max_N))

cleaned_x += max_N

##continous walk
x_cont = np.arange(positions) - 55
cont_prob_dist = cont.prob(x_cont, 40)
x_cont += 55

#plot the graph
fig, ax = plt.subplots()
x = np.arange(positions)

ax.plot(cleaned_x, cleaned_probs)
ax.plot(x_cont, cont_prob_dist)
ax.scatter(cleaned_x, cleaned_probs, s = 20, marker = "x", label = "Discrete Quantum Walk, 55 steps")
ax.scatter(x_cont, cont_prob_dist, s = 40, marker = "1", label = "Cont. Quantum Walk, 40 steps")
ax.scatter(cleaned_x, classic_probs, s = 10, marker = "o", label = "Classical Random Walk, 55 steps")

ax.legend()

loc = range(0, (positions // 10) * 10, 10)
plt.xticks(loc)
plt.xlim(0, positions)
plt.ylim((0, 0.12))
ax.set_xticklabels(range(-max_N, max_N, 10))
ax.set_xlabel("x")
ax.set_ylabel("Probability")

plt.show()