import kwant
import matplotlib.pyplot as plt
import numpy as np

import constants as const
import systems

radius = 1.5/np.pi*const.A_CC * 6
angle = -0.5*const.A_CC / radius

#system = systems.InfiniteContact(6, 6, 2.0*radius+3.2, rot1=angle, rot2=angle, offset=np.sqrt(3.0)*0.5*const.A_CC)
#system = systems.InfiniteContact(6, 6, 2.0*radius+3.2, rot1=0, rot2=0)

#system = systems.FiniteContact(6, 6, 6, 6, 10, 2.0*radius+3.2, rot1=angle, rot2=angle, leads=[True, False, False, True], offset=np.sqrt(3.0)*0.5*const.A_CC)

energy = [0.001]
overlaps = np.linspace(0.0, 600.0, 1000)
conductance = np.array([systems.SlidingContact(6, 6, 6, 6, overlap, 2.0*radius+3.2, rot1=angle, rot2=angle).transmission(0, 1, energy)[0] for overlap in overlaps])

plt.figure()
plt.plot(overlaps, conductance)
plt.xlabel("overlap [A]")
plt.ylabel("conducance [2e^2/h]")
plt.show()

'''
print("System finalised!")

energies = np.linspace(0.0, -3.0*const.INTRA_HOPPING, 1000)
data = system.transmission(0, 1, energies)

# Plot data

system.plotSystem()

plt.figure()
plt.plot(energies, data)
plt.xlabel("energy [eV]")
plt.ylabel("conducance [2e^2/h]")
plt.show()
'''
