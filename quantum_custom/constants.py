import numpy as np

#define spin up and spin down vectors
spin_up = np.array([1,0])
spin_down = np.array([0,1])

#define our Hadamard operator, H, in terms of ith, jth entries, Hij
H00 = np.outer(spin_up, spin_up)
H01 = np.outer(spin_up, spin_down)
H10 = np.outer(spin_down, spin_up)
H11 = np.outer(spin_down, spin_down)
H = (H00 + H01 + H10 - H11)/np.sqrt(2.0) #matrix representation of Hadamard gate in standard basis