import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from quantum_custom.core import SPIN_UP, SPIN_DOWN, plot
import quantum_custom.walks.discrete as disc
import quantum_custom.walks.classical as classc
import quantum_custom.walks.continuous as cont
import time

N = 50 # Time steps.
spin0 = 2**(-0.5) * (SPIN_UP + 1j * SPIN_DOWN) # Initial spin state for discrete quantum walk.
state0 = disc.state0(spin0, N)

walk_op = disc.walk_operator(N)

def dagger(mat):
    transpose = mat.transpose()
    dag = np.conjugate(transpose)
    return dag

def projector(position, N):
    positions = 2 * N + 1
    pos_state_vector = np.zeros(positions)
    pos_state_vector[position + N] = 1
    reduced_projector = np.outer(pos_state_vector, np.conjugate(pos_state_vector))
    projector = disc.eye_kron(2, reduced_projector)
    return projector

def evolve_dens_matrix(rho, operator, N, p):
    PSCs = []
    PSCs_dag = []
    operator_dag = dagger(operator)
    for j in range(-N, N + 1):
        proj = projector(j, N)
        PSC = np.dot(proj, operator)
        PSCs.append(PSC)
        PSCs_dag.append(dagger(PSC))

    for _ in range(N):
        rhot = (1-p) * (operator.dot(rho.dot(operator_dag)))
        for j in np.arange(-N, N + 1):
            summand = np.dot(rho, PSCs_dag[j])
            summand = np.dot(PSCs[j], summand)
            summand *= p
            rhot += summand
        rho = rhot
    return rhot

def measure(position, rho, N):
    proj = projector(position, N)
    proj_dag = dagger(proj)
    result = proj.dot(rho.dot(proj_dag))
    return result

rho0 = state0.rho
xs = []
ys = []
ps = [0,0.05,0.1,0.2,0.5,0.75,1]

for p in ps:
    rhot = evolve_dens_matrix(rho0, walk_op, N, p) #takes long time
    x = np.arange(-N, N + 1, 2)
    y = []
    for i in x:
        y.append(np.trace(measure(i, rhot, N)).real)
    xs.append(x)
    ys.append(y)

plot(xs, ys, "Decoherence applied to discrete quantum walk.", ps)
