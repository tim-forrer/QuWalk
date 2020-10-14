import numpy as np
from scipy.special import comb

from quantum_custom.core import spin_down, spin_up, H00, H11, H, QuantumState
import quantum_custom.walks.discrete as disc
import quantum_custom.walks.classical as classc
import quantum_custom.walks.continuous as cont
import quantum_custom.pdf as pdf

import matplotlib.pyplot as plt
import matplotlib.animation as animation

disc_N = 55 # "Coin flips".
spin0 = 1/(np.sqrt(2)) * (spin_down - 1j * spin_up) # Initial spin state for discrete quantum walk.

cont_N = 40 # Units of time for continous walk.

# Create PlotData() instances to be plotted and remove 0 points.
## Discrete walk.
x_disc = np.arange(-disc_N, disc_N)
disc_walk = pdf.disc_pdf(x_disc, disc_N, spin0)

## Continous walk.
x_cont = np.arange(-disc_N, disc_N)
cont_walk = pdf.cont_pdf(x_cont, cont_N)

# Plot graph.
fig, ax = plt.subplots()

ax.plot(disc_walk.x, disc_walk.y)
ax.plot(cont_walk.x, cont_walk.y)
ax.scatter(disc_walk.x, disc_walk.y, s = 20, marker = "x", label = "Discrete Quantum Walk, 55 \"coin flips\".")
ax.scatter(cont_walk.x, cont_walk.y, s = 40, marker = "1", label = "Cont. Quantum Walk, 40 time steps.")

ax.legend()

plt.title("Comparison of discrete and continuous quantum walks for initial state $|\phi\\rangle = |1\\rangle - i|0\\rangle$")

loc = np.arange(-60, 60, 20)
plt.xticks(loc, loc)
plt.xlim(-60, 60)
plt.ylim((0, 0.12))
ax.set_xlabel("x")
ax.set_ylabel("Probability")

plt.show()