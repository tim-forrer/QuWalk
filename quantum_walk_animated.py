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
positions = 2*max_N + 1

#initial conditions
initial_spin = spin_down
initial_position = np.zeros(positions)
initial_position[max_N] = 1
initial_state = np.kron(np.matmul(H, initial_spin), initial_position) #initial state is Hadamard acting on intial state, tensor product with the initial position
quantum_state = QuantumState(initial_state)

#plot the graph
fig, ax = plt.subplots()
plt.title("N = 0")
x = np.arange(positions)
line, = ax.plot([],[])

loc = range(0, positions, positions // 10)
plt.xticks(loc)
plt.xlim(0, positions)
plt.ylim((0, 1))

ax.set_xticklabels(range(-max_N, max_N + 1, positions // 10))
ax.set_xlabel("x")
ax.set_ylabel("Probability")

def init():
    line.set_data([],[])
    return line,

def update(N):
    next_state = walk.flip_once(quantum_state.state, max_N)
    probs = walk.get_prob(next_state, max_N)
    quantum_state.state = next_state
    start_index = N % 2 + 1
    cleaned_probs = probs[start_index::2]
    cleaned_x = x[start_index::2]
    line.set_data(cleaned_x, cleaned_probs)
    if cleaned_probs.max() != 0:
        plt.ylim((0, cleaned_probs.max()))
    plt.title(f"N = {N}")
    return line,


anim = animation.FuncAnimation(
    fig, 
    update,
    frames = max_N + 1,
    init_func = init,
    interval = 20,
    repeat = False,
    blit = True,
    )

anim.save("animated.gif", writer = "ffmpeg", fps = 15)