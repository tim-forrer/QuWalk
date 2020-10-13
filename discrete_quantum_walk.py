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
max_N = 100
positions = 2 * max_N + 1

#initial conditions
initial_spin = spin_down
initial_position = np.zeros(positions)
initial_position[max_N] = 1
initial_state = np.kron(np.matmul(H, initial_spin), initial_position) #initial state is Hadamard acting on intial state, tensor product with the initial position
quantum_state = QuantumState(initial_state)
walk_operator = disc.walk_operator(max_N)

#conduct walk
quantum_state.state = np.linalg.matrix_power(walk_operator, max_N + 1).dot(quantum_state.state) #fudge
probs = disc.prob(quantum_state.state, max_N)

#create arrays to be plotted and remove 0 points
x = np.arange(positions)
start_index = (max_N + 1) % 2
cleaned_x = x[1::2] - 1
cleaned_probs = probs[1::2]

classic_probs = []
cleaned_x -= 100

for m in cleaned_x:
    classic_probs.append(classc.prob(m, max_N))

cleaned_x += 100

#plot the graph
fig, ax = plt.subplots()
plt.title("N = 100")
x = np.arange(positions)

ax.plot(cleaned_x, cleaned_probs)
ax.scatter(cleaned_x, cleaned_probs, s = 20, marker = "x", label = "Quantum Random Walk")
ax.scatter(cleaned_x, classic_probs, s = 20, marker = "o", label = "Classical Random Walk")

ax.legend()

loc = range(0, positions, positions // 10)
plt.xticks(loc)
plt.xlim(0, positions)
plt.ylim((0, cleaned_probs.max()))

ax.set_xticklabels(range(-max_N, max_N + 1, positions // 10))
ax.set_xlabel("x")
ax.set_ylabel("Probability")

plt.show()