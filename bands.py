import kwant
import matplotlib.pyplot as plt
import numpy as np

import constants as const
import systems
import cnt

n1 = 6
m1 = 6
n2 = 6
m2 = 6

unitCellLength = cnt.CNT(n1, m1, 1).length

radius = 1.5/np.pi*const.A_CC * 6
circumference = 2.0*np.pi*radius 
angle1 = -0.5*const.A_CC / radius
angle2 = (circumference/12.0 - const.A_CC) / radius

distance = 2.0*radius + 3.2

system = systems.InfiniteContact(n1, m1, distance, rot1=angle1, rot2=angle2, offset=0.0*unitCellLength)

system.plotSystem()
system.plotBandStructure()
