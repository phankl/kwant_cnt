import kwant
import numpy as np

import constants as const
import systems


n1 = 6
m1 = 6
n2 = 4
m2 = 5

radius1 = 0.5 * const.A_CC * np.sqrt(3.0*(n1**2 + n1*m1 + m1**2)) / np.pi
radius2 = 0.5 * const.A_CC * np.sqrt(3.0*(n2**2 + n2*m2 + m2**2)) / np.pi

distance = radius1 + radius2 + 3.13
overlap = 10.0

system = systems.FiniteContact(n1, m1, n2, m2, overlap, distance, offset1=0.5, offset2=2.0, leads=(True, True, True, True))
system.plotSystem()
