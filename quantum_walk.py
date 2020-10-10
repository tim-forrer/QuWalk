import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from quantum_custom.constants import spin_down, spin_up, H00, H11, H
import quantum_custom.walk as walk
from scipy.special import comb
from math import factorial

class QuantumState:
    def __init__(self, state):
        self.state = state

#"coin flips"
max_N = 100
positions = 2 * max_N + 1

#initial conditions
initial_spin = spin_down
initial_position = np.zeros(positions)
initial_position[max_N] = 1
initial_state = np.kron(np.matmul(H, initial_spin), initial_position) #initial state is Hadamard acting on intial state, tensor product with the initial position
quantum_state = QuantumState(initial_state)
walk_operator = walk.walk_operator(max_N)

#conduct walk
quantum_state.state = np.linalg.matrix_power(walk_operator, max_N).dot(quantum_state.state) #fudge

probs = walk.get_probs(quantum_state.state, max_N)

#create arrays to be plotted and remove 0 points
x = np.arange(positions)
cleaned_x = x[::2] #- 1 #fudge
cleaned_probs = probs[::2]
# for i, prob in enumerate(probs):
#     if prob != 0:
#         cleaned_x = np.append(cleaned_x, x[i])
#         cleaned_probs = np.append(cleaned_probs, prob)
print(cleaned_probs)
print(cleaned_x)

def classic_prob_m(m, N):
    """
    Given target position m and N flips of the coin, returns the probability of finishing on m.

    Using convention that heads is +1 and tails is -1.
    Assuming m > 0 we know that we need at least m heads, and following that equal numbers of heads and tails.
    Then count the number of combinations that are possible of having that number of heads and divide by total number of combinations.

    Since this distribution is symmetric, if m < 0 we can reverse its sign as the probability will be the same.
    """
    if m < 0:
        m = -m
    Nheads = int(m + (N - m) / 2)

    prob = comb(N, Nheads, exact = False) * (0.5)**N
    return prob

classic_probs = []
cleaned_x -= 100

for m in cleaned_x:
    classic_probs.append(classic_prob_m(m, max_N))
cleaned_x += 100

print(classic_probs[49:52])

#plot the graph
fig, ax = plt.subplots()
plt.title("N = 100")
x = np.arange(positions)
ax.plot(cleaned_x, cleaned_probs, marker = "x", markersize = 5) #quantum walk data points
ax.plot(cleaned_x, classic_probs, marker = "o", markersize = 2, lw = 0) 

loc = range(0, positions, positions // 10)
plt.xticks(loc)
plt.xlim(0, positions)
plt.ylim((0, cleaned_probs.max()))

ax.set_xticklabels(range(-max_N, max_N + 1, positions // 10))
ax.set_xlabel("x")
ax.set_ylabel("Probability")

plt.grid(True)

plt.show()