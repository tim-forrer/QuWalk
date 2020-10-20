import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from quantum_custom.core import SPIN_DOWN, plot
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
xs = [disc_walk.x, classc_walk.x]
ys = [disc_walk.y, classc_walk.y]
labels = [r"Disc. QW, $|\psi_0\rangle = |1\rangle$", "Classical QW"]
title = "Comparison of discrete quantum walk and classical quantum walk at N = 100."
plot(xs, ys, labels = labels, title = title)