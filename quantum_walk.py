import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from quantum_custom.constants import spin_down, spin_up, H00, H11, H
import quantum_custom.walk as walk

class QuantumState:
    def __init__(self, state):
        self.state = state

#"coin flips"
max_N = 100 #this will be the final number of coin flips
positions = 2 * max_N + 1

#initial conditions
initial_spin = spin_down
initial_position = np.zeros(positions)
initial_position[max_N] = 1
initial_state = np.kron(np.matmul(H, initial_spin), initial_position) #initial state is Hadamard acting on intial state, tensor product with the initial position
quantum_state = QuantumState(initial_state)

#conduct walk
for i in range(max_N + 1):
    next_state = walk.flip_once(quantum_state.state, max_N)
    quantum_state.state = next_state

probs = walk.get_prob(quantum_state.state, max_N)

#create arrays to be plotted
start_index = max_N % 2 + 1
x = np.arange(positions)
cleaned_x = x[start_index::2]
cleaned_probs = probs[start_index::2]

#plot the graph
fig, ax = plt.subplots()
plt.title("N = 100")
x = np.arange(positions)
ax.plot(cleaned_x, cleaned_probs)

loc = range(0, positions, positions // 10)
plt.xticks(loc)
plt.xlim(0, positions)
plt.ylim((0, cleaned_probs.max()))

ax.set_xticklabels(range(-max_N, max_N + 1, positions // 10))
ax.set_xlabel("x")
ax.set_ylabel("Probability")

plt.show()