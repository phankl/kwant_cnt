import kwant
import matplotlib.pyplot as plt
import numpy as np

from mpi4py import MPI

import constants as const
import systems

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

unitCellLength = 2.4595121467478056

#overlapStart = 8.52
#overlapPoints = 201
#overlapEnd = overlapStart + (overlapPoints-1)*3.0*unitCellLength

overlapStart = 0.0
overlapEnd = 3000.0
overlapPoints = 301

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
conductance = np.zeros((overlapPoints, 8))

for i in range(rank, overlapPoints, size):
  overlap = overlaps[i]
  system = systems.FiniteContact(6, 6, 6, 6, overlap, distance, rot1=angle, rot2=angle, offset=0.5*unitCellLength)
  conductance[i, 0] = system.transmission(0, 0, energies)[0]
  conductance[i, 1] = system.transmission(0, 1, energies)[0]
  conductance[i, 2] = system.transmission(0, 2, energies)[0]
  conductance[i, 3] = system.transmission(0, 3, energies)[0]
  conductance[i, 4] = system.transmission(2, 0, energies)[0]
  conductance[i, 5] = system.transmission(2, 1, energies)[0]
  conductance[i, 6] = system.transmission(2, 2, energies)[0]
  conductance[i, 7] = system.transmission(2, 3, energies)[0]

conductanceRoot = np.zeros((overlapPoints, 8))

comm.Allreduce([conductance, MPI.DOUBLE], [conductanceRoot, MPI.DOUBLE], op=MPI.SUM)

if rank == 0:
  overlaps = np.reshape(overlaps, (-1, 1))

  data = np.append(overlaps, conductanceRoot, axis=1)

  np.savetxt("conductance.dat", data)
