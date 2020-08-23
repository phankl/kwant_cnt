import kwant
import numpy as np

from mpi4py import MPI

import constants as const
import cnt
import systems

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

n1 = 0
m1 = 0
n2 = 0
m2 = 0

if rank == 0:
  print("Pairing: (", n1, ",", m1, ")", " (", n2, ",", m2, ")", flush=True)

prefix = str(n1) + "_" + str(m1) + "_" + str(n2) + "_" + str(m2) + "_"
meanName = prefix + "mean.dat"
stdName = prefix + "std.dat"
minName = prefix + "min.dat"
maxName = prefix + "max.dat"

energy = 0.5

angleSamples = 200
offsetSamples = 2
rotSamples = 2

angleOffset = 0.05 * np.pi

radius1 = 0.5 * const.A_CC * np.sqrt(3.0*(n1**2 + n1*m1 + m1**2)) / np.pi
radius2 = 0.5 * const.A_CC * np.sqrt(3.0*(n2**2 + n2*m2 + m2**2)) / np.pi

distance = radius1 + radius2 + 3.13

unitCell1 = cnt.CNT(n1, m1, 1)
unitCell2 = cnt.CNT(n2, m2, 1)
offsetMax1 = unitCell1.length
offsetMax2 = unitCell2.length
offsetStep1 = offsetMax1 / rotSamples
offsetStep2 = offsetMax2 / rotSamples
offsetMax1 -= offsetStep1
offsetMax2 -= offsetStep2

rotMax1 = 2.0 * np.pi / np.gcd(n1, m1)
rotMax2 = 2.0 * np.pi / np.gcd(n2, m2)
rotStep1 = rotMax1 / rotSamples
rotStep2 = rotMax2 / rotSamples
rotMax1 -= rotStep1
rotMax2 -= rotStep2

angles = np.linspace(angleOffset, np.pi-angleOffset, angleSamples)
offsets1 = np.linspace(0.0, offsetMax1, offsetSamples)
offsets2 = np.linspace(0.0, offsetMax2, offsetSamples)
rots1 = np.linspace(0.0, rotMax1, rotSamples)
rots2 = np.linspace(0.0, rotMax2, rotSamples)

conductanceData= np.zeros((4, angleSamples, 8))

k = 0
indices = range(rank, angleSamples, size)
for i in indices:
  angle = angles[i]

  conductance = np.zeros((offsetSamples**2 * rotSamples**2, 8))
  j = 0

  for offset1 in offsets1:
    for offset2 in offsets2:
      for rot1 in rots1:
        for rot2 in rots2:
          system = systems.angledContact(n1, m1, n2, m2, angle, distance, rot1=rot1, offset1=offset1, rot2=rot2, offset2=offset2)
          
          smatrix = kwant.smatrix(system.systemFinalized, energy)
          conductance[j, 0] = smatrix.transmission(0, 0)
          conductance[j, 1] = smatrix.transmission(0, 1)
          conductance[j, 2] = smatrix.transmission(0, 2)
          conductance[j, 3] = smatrix.transmission(0, 3)
          conductance[j, 4] = smatrix.transmission(2, 0)
          conductance[j, 5] = smatrix.transmission(2, 1)
          conductance[j, 6] = smatrix.transmission(2, 2)
          conductance[j, 7] = smatrix.transmission(2, 3)
          
          j += 1

  conductanceData[0, i] = np.mean(conductance, axis=0)
  conductanceData[1, i] = np.std(conductance, axis=0)
  conductanceData[2, i] = np.amin(conductance, axis=0)
  conductanceData[3, i] = np.amax(conductance, axis=0)

  k += 1
  print("Process", rank+1, "out of", size, "progress:", k, "/", len(indices), flush=True)

conductanceRoot = np.zeros((4, angleSamples, 8))
comm.Allreduce([conductanceData, MPI.DOUBLE], [conductanceRoot, MPI.DOUBLE], op=MPI.SUM)

if rank == 0:
  angles = np.reshape(angles, (-1, 1))

  conductanceMean = np.append(angles, conductanceRoot[0], axis=1)
  conductanceStd = np.append(angles, conductanceRoot[1], axis=1)
  conductanceMin = np.append(angles, conductanceRoot[2], axis=1)
  conductanceMax = np.append(angles, conductanceRoot[3], axis=1)

  np.savetxt(meanName, conductanceMean)
  np.savetxt(stdName, conductanceStd)
  np.savetxt(minName, conductanceMin)
  np.savetxt(maxName, conductanceMax)
