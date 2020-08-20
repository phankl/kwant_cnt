import kwant
import matplotlib.pyplot as plt
import numpy as np

from mpi4py import MPI

import constants as const
import systems

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

siteEnergy = 0.0
hoppingEnergy = 1.0

overlapStart = 1
overlapEnd = 10000
overlapPoints = overlapEnd - overlapStart

energy = 0.001

energies = [energy]

overlaps = range(overlapStart, overlapEnd)
conductance = np.zeros((overlapPoints, 8))

for i in range(rank, overlapPoints, size):
  overlap = overlaps[i]
  system = systems.Chain(overlap, site=siteEnergy, hopping=hoppingEnergy)
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

  np.savetxt("chain.dat", data)
