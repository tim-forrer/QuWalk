import numpy as np
from scipy.linalg import expm
from quantum_custom.core import PlotData

def adj_mat(n, graph = "Z"):
    """
    The adjacency matrix of the specified graph.

    More graphs will likely be added in future.
    """
    allowed_graphs = ["Z", "H"]
    if graph == "Z":
        return z_adj(n)
    elif graph == "H":
        return hypercube_adj(n)
    else:
        raise ValueError(f"Invalid graph specified. Allowed values are {allowed_graphs}")
    return

def z_adj(n):
    """
    Adjacency matrix of the discrete number line, from -n to +n (inclusive).
    """
    mat = np.zeros((2 * n + 1, 2 * n + 1))
    for i, row in enumerate(mat):
        if i > 0:
            row[i - 1] = 1
        if i < 2 * n:
            row[i + 1] = 1
    return mat

def hypercube_adj(n):
    """
    Adjacency matrix of undirected hypercube of dimension N.
    """
    N = 2**n
    mat = np.zeros((N, N), dtype = int)
    powers = [2**i for i in range(n)]
    for i in range(N):
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

    To match common convention, gamma is divided by two.
    """
    H = (gamma / 2) * A
    return H

def search_H(n, target, gamma, graph):
    """
    The continuous time search Hamiltonian for the given graph, where a target vertex is marked as a lower energy state of the Hamiltonian.
    """
    N = 2**n
    a_mat = adj_mat(n, graph = graph)
    base_hamil = H(a_mat, gamma)
    walk_hamil = (gamma / 2) * n * np.identity(N) - base_hamil
    prob_hamil = np.identity(N) - np.outer(target, target)
    search_hamil = walk_hamil + prob_hamil
    return search_hamil