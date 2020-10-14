import numpy as np

# Create the QuantumState class
class QuantumState:
    def __init__(self, state):
        self.state = state

# Define PlotData class
class PlotData():
    def __init__(self, x, y, N):
        self.x = x
        self.y = y
        self.N = N

#define spin up and spin down vectors as standard basis
spin_up = np.array([1,0])
spin_down = np.array([0,1])

#define our Hadamard operator, H, in terms of ith, jth entries, Hij
H00 = np.outer(spin_up, spin_up)
H01 = np.outer(spin_up, spin_down)
H10 = np.outer(spin_down, spin_up)
H11 = np.outer(spin_down, spin_down)
H = (H00 + H01 + H10 - H11)/np.sqrt(2.0) #matrix representation of Hadamard gate in standard basis