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
SPIN_UP = np.array([1,0])
SPIN_DOWN = np.array([0,1])

#define our Hadamard operator, H, in terms of ith, jth entries, Hij
H00 = np.outer(SPIN_UP, SPIN_UP)
H01 = np.outer(SPIN_UP, SPIN_DOWN)
H10 = np.outer(SPIN_DOWN, SPIN_UP)
H11 = np.outer(SPIN_DOWN, SPIN_DOWN)
H = (H00 + H01 + H10 - H11)/np.sqrt(2.0) #matrix representation of Hadamard gate in standard basis