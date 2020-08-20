import kwant
import numpy as np

import constants as const
import systems

radius = 1.5/np.pi*const.A_CC * 6
distance = 2.0 * radius + 3.2
system = systems.angledContact(6, 5, 4, 3, 0.2*np.pi, distance, offset1=0.4, offset2=0.8)

system.plotSystem()
