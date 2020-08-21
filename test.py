import kwant
import numpy as np

import constants as const
import systems


n1 = 6
m1 = 6
n2 = 6
m2 = 6

radius1 = 0.5 * const.A_CC * np.sqrt(3.0*(n1**2 + n1*m1 + m1**2)) / np.pi
radius2 = 0.5 * const.A_CC * np.sqrt(3.0*(n2**2 + n2*m2 + m2**2)) / np.pi

distance = radius1 + radius2 + 3.13
angle = 0.15

for angle in np.linspace(0.5*np.pi - 0.1, 0.5*np.pi+0.1, 10):
  system = systems.angledContact(n1, m1, n2, m2, angle, distance)
  system.plotSystem()
