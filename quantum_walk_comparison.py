import numpy as np
from scipy.special import comb

from quantum_custom.core import SPIN_DOWN, SPIN_UP, H00, H11, H, QuantumState
import quantum_custom.walks.discrete as disc
import quantum_custom.walks.classical as classc
import quantum_custom.walks.continuous as cont

import matplotlib.pyplot as plt
import matplotlib.animation as animation

disc_N = 55 # "Coin flips".
spin0 = 1/(np.sqrt(2)) * (SPIN_DOWN - 1j * SPIN_UP) # Initial spin state for discrete quantum walk.

cont_N = 40 # Units of time for continous walk.

plot_N = max(disc_N, cont_N)
print(plot_N)

# Create PlotData() instances to be plotted and remove 0 points.
## Discrete walk.
x_disc = np.arange(-plot_N, plot_N + 1)
disc_walk = disc.pdf(x_disc, disc_N, spin0)
print(disc_walk.x[0:3])

## Continous walk.
x_cont = np.arange(-plot_N, plot_N + 1)
cont_walk = cont.pdf(x_cont, cont_N)

# Plot graph.
fig, ax = plt.subplots()

ax.plot(disc_walk.x, disc_walk.y)
ax.scatter(disc_walk.x, disc_walk.y, s = 20, marker = "x", label = "Discrete Quantum Walk, 55 'coin flips'.")

ax.plot(cont_walk.x, cont_walk.y)
ax.scatter(cont_walk.x, cont_walk.y, s = 40, marker = "1", label = "Cont. Quantum Walk, 40 time steps.")

ax.legend()

plt.title("Comparison of discrete and continuous quantum walks for initial state $|\phi\\rangle = |1\\rangle - i|0\\rangle$")

loc = np.arange(-60, 60 + 1, 20)
plt.xticks(loc, loc)
plt.xlim(-60, 60)
plt.ylim((0, 0.12))
ax.set_xlabel("x")
ax.set_ylabel("Probability")

plt.show()