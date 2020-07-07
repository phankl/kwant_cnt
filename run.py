import kwant
import matplotlib.pyplot as plt
import numpy as np

import constants as const
import systems

system = systems.InfiniteContact(6, 6, 11.336, offset=const.A_CC/2)

#energies, data = system.transmission(0, 1)

# Plot data

system.plotSystem()
system.plotBandStructure()

'''
plt.figure()
plt.plot(energies, data)
plt.xlabel("energy [eV]")
plt.ylabel("conducance [2e^2/h]")
plt.show()
'''
