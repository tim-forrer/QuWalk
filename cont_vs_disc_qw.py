import numpy as np
from quantum_custom.core import plot, SPIN_DOWN, SPIN_UP
import quantum_custom.walks.continuous as cont
import quantum_custom.walks.discrete as disc

x_max = 55
x = np.arange(-x_max, x_max + 1, 1)

# Continuous time quantum walk
gamma = 0.5
t = 40
A = cont.adj_mat(x_max)
H = cont.H(A, gamma)
state0 = np.zeros(2 * x_max + 1)
state0[x_max] = 1
state_t = cont.state_t(state0, H, t)
prob_amps = cont.measure(state_t)

# Discrete time quantum walk
spin0 = (1 / (np.sqrt(2))) * (SPIN_UP - 1j * SPIN_DOWN)
disc_walk = disc.pdf(x, 55, spin0)

# Plot data
xs = [x, disc_walk.x]
ys = [prob_amps, disc_walk.y]
labels = ["Cont. Time QW (t = 40)", r"Disc. Time QW (N = 55), $|spin_0\rangle = \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle$)"]

plot(xs, ys, title = "Comparison of discrete and continuous time QW.", labels = labels)

print(f"The sum of the probability amplitudes for the discrete time walk is {sum(disc_walk.y)}.")
print(f"The sum of the probability amplitudes for the continuous time walk is {sum(prob_amps)}.")