import kwant
import matplotlib.pyplot as plt
import numpy as np

import constants as const
import systems

radius = 1.5/np.pi*const.A_CC * 6
angle = -0.5*const.A_CC / radius

energy = [0.001]
overlaps = np.linspace(40.0, 70.0, 1000)
conductance = np.array([systems.SlidingContact(6, 6, 6, 6, overlap, 2.0*radius+3.2, rot1=angle, rot2=angle).transmission(0, 1, energy)[0] for overlap in overlaps])

overlaps = np.reshape(overlaps, (-1, 1))
conductance = np.reshape(conductance, (-1, 1))

data = np.append(overlaps, conductance, axis=1)

np.savetxt("conductance.dat", data)
