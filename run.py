import kwant
import matplotlib.pyplot as plt
import numpy as np

from mpi4py import MPI

import constants as const
import systems

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


overlapStart = 0.0
overlapEnd = 600.0
overlapPoints = 10001

n1 = 6
m1 = 6
n2 = 6
m2 = 6

energy = 0.001

radius = 1.5/np.pi*const.A_CC * 6
angle = -0.5*const.A_CC / radius

distance = 2.0*radius + 3.2

energies = [energy]

overlaps = np.linspace(overlapStart, overlapEnd, overlapPoints)
conductance = np.zeros(overlapPoints)

for i in range(rank, overlapPoints, size):
  overlap = overlaps[i]
  system = systems.SlidingContact(6, 6, 6, 6, overlap, distance, rot1=angle, rot2=angle)
  conductance[i] = system.transmission(0, 1, energies)[0]

conductanceRoot = np.zeros(overlapPoints)

comm.Allreduce([conductance, MPI.DOUBLE], [conductanceRoot, MPI.DOUBLE], op=MPI.SUM)

if rank == 0:
  overlaps = np.reshape(overlaps, (-1, 1))
  conductance = np.reshape(conductanceRoot, (-1, 1))

  data = np.append(overlaps, conductance, axis=1)

  np.savetxt("conductance.dat", data)
