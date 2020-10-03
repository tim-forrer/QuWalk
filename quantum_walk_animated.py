import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from quantum_custom.constants import spin_down, spin_up, H00, H11, H

#"coin flips"
max_N = 100 #this will be the final number of coin flips
positions = 2*max_N + 1

#initial conditions
initial_spin = spin_down
initial_position = np.zeros(positions)
initial_position[max_N] = 1
initial_state = np.kron(np.matmul(H, initial_spin), initial_position) #intial state is Hadamard acting on intial state, tensor product with the initial position

current_state = initial_state #this will save what state we currently have prepared

#define walk operators
shift_plus = np.roll(np.eye(positions), 1, axis = 0)
shift_minus = np.roll(np.eye(positions), -1, axis = 0)
step_operator = np.kron(H00, shift_plus) + np.kron(H11, shift_minus)
walk_operator = step_operator.dot(np.kron(H, np.eye(positions)))

def flip_once(state):
    """
    Flips the Hadamard coin once and acts on the given state appropriately.
    Returns the state after the Hadamard coin flip.
    """
    next_state = walk_operator.dot(state)
    return next_state

def get_prob(state):
    """
    For the given state, calculates the probability of being in any possible position.
    Returns an array of probabilities.
    """
    prob = np.empty(positions)
    for k in range(positions):
        posn = np.zeros(positions)
        posn[k] = 1
        posn_outer = np.outer(posn, posn)
        alt_measurement_k = eye_kron(2, posn_outer)
        proj = alt_measurement_k.dot(state)
        prob[k] = proj.dot(proj.conjugate()).real       
    return prob

def eye_kron(eye_dim, mat):
    """
    Speeds up the calculation of the tensor product of an identity matrix of dimension eye_dim with a given matrix.
    This exploits the fact that majority of values in the resulting matrix will be zeroes apart from on the leading diagonal where we simply have copies of the given matrix.
    Returns a matrix.
    """
    mat_dim = len(mat)
    result_dim = eye_dim * mat_dim #dimension of the resulting matrix
    result = np.zeros((result_dim, result_dim))
    result[0:mat_dim, 0:mat_dim] = mat
    result[mat_dim:result_dim, mat_dim:result_dim] = mat
    return result


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
    global current_state
    next_state = flip_once(current_state)
    probs = get_prob(next_state)
    current_state = next_state

    start_index = N % 2 + 1
    cleaned_probs = probs[start_index::2]
    cleaned_x = x[start_index::2]
    line.set_data(cleaned_x, cleaned_probs)
    plt.ylim((0, cleaned_probs.max()))
    plt.title(f"N = {N}")
    return line,

anim = animation.FuncAnimation(
    fig, 
    update,
    init_func = init,
    frames = max_N + 1,
    interval = 20,
    repeat = False,
    blit = False
    )

plt.show()