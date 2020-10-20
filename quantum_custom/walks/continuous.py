import numpy as np
from scipy.linalg import expm
from quantum_custom.core import PlotData

def adj_mat(N, graph = "Z"):
    """
    The adjacency matrix of the specified graph.

    More graphs will likely be added in future.
    """
    allowed_graphs = ["Z", "H"]
    if graph == "Z":
        return z_adj(N)
    elif graph == "H":
        return hypercube_adj(N)
    else:
        raise ValueError(f"Invalid graph specified. Allowed values are {allowed_graphs}")
    return

def z_adj(N):
    """
    Adjacency matrix of the discrete number line, from -N to +N (inclusive).
    """
    mat = np.zeros((2 * N + 1, 2 * N + 1))
    for i, row in enumerate(mat):
        if i > 0:
            row[i - 1] = 1
        if i < 2 * N:
            row[i + 1] = 1
    return mat

def hypercube_adj(N):
    """
    Adjacency matrix of undirected hypercube of dimension N.
    """
    mat = np.zeros((2**N, 2**N), dtype = int)
    powers = [2**n for n in range(N)]
    for i in range(2**N):
        connected_verts = [i ^ y for y in powers] # Bitwise XOR i with all the "columns" (1, 2, 4, ...) to find connected vertices.
        for vert in connected_verts:
            mat[i][vert] = 1
    return mat

def hamming_dist(x, origin):
    """
    Hamming distance of x from the origin.

    Essentially counts the number of bit flips needed to go from the origin -> x (it's a distance though so direction is irrelevant).
    """
    N = max(len(bin(x)), len(bin(origin))) - 2 # Discount the 0b at the start of the bit string.
    distance = 0
    for i in range(0, N):
        x_bit = x >> i & 1 # // 2 and then AND the result with 1 - basically tells you what the bit at position i is.
        origin_bit = origin >> i & 1 # Same as above.
        distance += x_bit != origin_bit # If the bits in the same column are not equal, we'll need a bit flip to get from origin -> x, so increases Hamming distance by 1.
    return distance

def measure(state):
    """
    Measure the probability amplitude of each positional basis state.

    Assumes the Hilbert space is spanned by the positional basis states.
    """
    N = len(state)
    conj_mat = np.zeros((N, N), dtype = complex)
    for i, entry in enumerate(state):
        conj_mat[i][i] = np.conjugate(entry) # Fill the conjugate matrix with the complex conjugates of the state along its diagonal.
    prob_amps = np.dot(conj_mat, state)
    return prob_amps.real

def state_t(state_0, H, t):
    """
    State at time t as evolved under time independent Hamiltonian H.
    """
    U = expm(-1j * H * t)
    return np.dot(U, state_0)

def H(A, gamma):
    """
    Hamiltonian of a graph with adjacency matrix A and hopping rate gamma.
    """
    H = gamma * A
    return H