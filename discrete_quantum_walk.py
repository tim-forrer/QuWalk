import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from quantum_custom.core import SPIN_DOWN, SPIN_UP, H00, H11, H, QuantumState
import quantum_custom.walks.discrete as disc
import quantum_custom.walks.classical as classc
import quantum_custom.walks.continuous as cont

disc_N = 100 # "Coin flips".
spin0 = SPIN_DOWN # Initial spin state for discrete quantum walk.

# Create PlotData() instances to be plotted and remove 0 points.
## Discrete walk.
x_disc = np.arange(-disc_N, disc_N  + 1)
disc_walk = disc.pdf(x_disc, disc_N, spin0)

## Classical walk.
classc_walk = classc.pdf(x_disc, disc_N)

# Plot the graph
fig, ax = plt.subplots()
plt.title("N = 100")

ax.plot(disc_walk.x, disc_walk.y)
ax.scatter(disc_walk.x, disc_walk.y, s = 20, marker = "x", label = "Quantum Random Walk")
ax.scatter(classc_walk.x, classc_walk.y, s = 20, marker = "o", label = "Classical Random Walk")

ax.legend()

loc = np.arange(-disc_N, disc_N, 20)
plt.xticks(loc, loc)
plt.xlim(-disc_N, disc_N)
plt.ylim((0, 0.15))
ax.set_xlabel("x")
ax.set_ylabel("Probability")

plt.show()