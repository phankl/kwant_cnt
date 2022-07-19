import kwant
import numpy as np

import constants as const
import cnt
import systems

n1 = 10
m1 = 10
n2 = 10
m2 = 10

index = 0

print("Pairing: (", n1, ",", m1, ")", " (", n2, ",", m2, ")", flush=True)
print("Index:", index, flush=True)

prefix = str(n1) + "_" + str(m1) + "_" + str(n2) + "_" + str(m2) + "_" + str(index) + "_"
meanName = prefix + "mean.dat"
stdName = prefix + "std.dat"
minName = prefix + "min.dat"
maxName = prefix + "max.dat"

n_energy = 101
energies = np.linspace(0.0, 1.0, n_energy)

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

conductanceData = np.zeros((4, n_energy, 8))
data = np.zeros((4, n_energy, 10))

angle = angles[index]

samples = offsetSamples**2 * rotSamples**2
conductance = np.zeros((samples, n_energy, 8))
j = 0

for offset1 in offsets1:
  for offset2 in offsets2:
    for rot1 in rots1:
      for rot2 in rots2:
        system = systems.angledContact(n1, m1, n2, m2, angle, distance, rot1=rot1, offset1=offset1, rot2=rot2, offset2=offset2)

        print(f"{j+1} / {samples} system constructed")
        smatrices = [kwant.smatrix(system.systemFinalized, energy) for energy in energies]
        conductance[j, :, 0] = np.array([smatrix.transmission(0, 0) for smatrix in smatrices])
        conductance[j, :, 1] = np.array([smatrix.transmission(0, 1) for smatrix in smatrices])
        conductance[j, :, 2] = np.array([smatrix.transmission(0, 2) for smatrix in smatrices])
        conductance[j, :, 3] = np.array([smatrix.transmission(0, 3) for smatrix in smatrices])
        conductance[j, :, 4] = np.array([smatrix.transmission(2, 0) for smatrix in smatrices])
        conductance[j, :, 5] = np.array([smatrix.transmission(2, 1) for smatrix in smatrices])
        conductance[j, :, 6] = np.array([smatrix.transmission(2, 2) for smatrix in smatrices])
        conductance[j, :, 7] = np.array([smatrix.transmission(2, 3) for smatrix in smatrices])

        print(f"{j+1} / {samples} conductance calculated")

        j += 1

conductanceData[0] = np.mean(conductance, axis=0)
conductanceData[1] = np.std(conductance, axis=0)
conductanceData[2] = np.amin(conductance, axis=0)
conductanceData[3] = np.amax(conductance, axis=0)

data[:, :, 0] = np.full((n_energy, ), angle)
data[:, :, 1] = energies
data[:, :, 2:] = conductanceData

np.savetxt(meanName, data[0])
np.savetxt(stdName, data[1])
np.savetxt(minName, data[2])
np.savetxt(maxName, data[3])

print("Done")
