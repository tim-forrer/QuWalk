import numpy as np
import matplotlib.pyplot as plt

#"coin flips"
N = 100
positions = 2*N + 1

#define spin up and spin down vectors
spin_up = np.array([1,0])
spin_down = np.array([0,1])

#define our Hadamard operator, H, in terms of ith, jth entries, Hij
H00 = np.outer(spin_up, spin_up)
H01 = np.outer(spin_up, spin_down)
H10 = np.outer(spin_down, spin_up)
H11 = np.outer(spin_down, spin_down)
H = (H00 + H01 + H10 - H11)/np.sqrt(2.0) #matrix representation of Hadamard gate in standard basis

#initial conditions
initial_spin = spin_up
initial_position = np.zeros(positions)
initial_position[N] = 1
initial_state = np.kron(np.matmul(H, initial_spin), initial_position) #intial state is Hadamard acting on intial state, tensor product with the initial position

#define operators
shift_plus = np.roll(np.eye(positions), 1, axis = 0)
shift_minus = np.roll(np.eye(positions), -1, axis = 0)
step_operator = np.kron(H00, shift_plus) + np.kron(H11, shift_minus)
walk_operator = step_operator.dot(np.kron(H, np.eye(positions)))

final_state = np.linalg.matrix_power(walk_operator, N).dot(initial_state)

#obtain the probabilities for each positions after N coin flips
prob = np.empty(positions)
for k in range(positions):
    posn = np.zeros(positions)
    posn[k] = 1     
    measurement_k = np.kron(np.eye(2), np.outer(posn,posn)) #our measurement operator that measures at position k
    proj = measurement_k.dot(final_state)
    prob[k] = proj.dot(proj.conjugate()).real

#plot the graph
fig = plt.figure()
ax = fig.add_subplot(111)

plt.plot(np.arange(positions), prob)
loc = range(0, positions, positions // 10)
plt.xticks(loc)
plt.xlim(0, positions)
ax.set_xticklabels(range(-N, N+1, positions // 10))

plt.show()