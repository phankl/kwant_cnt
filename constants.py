# Constants for CNT Tight-Binding Simulations

import numpy as np

# Numerical Constants

EPS = 1.0e-8
COUPLING_CUTOFF = 1.0e-6

# Graphene 

A_CC = 1.42
A = np.sqrt(3.0) * A_CC

D = np.array([-A_CC, 0.0])
#B = 4.0 * np.pi / 3.0 / A_CC

A1 = 0.5 * A * np.array([np.sqrt(3), 1.0])
A2 = 0.5 * A * np.array([np.sqrt(3), -1.0])
#B1 = 0.5 * B * np.array([1.0, np.sqrt(3)])
#B2 = 0.5 * B * np.array([1.0, -np.sqrt(3)])

INTRA_HOPPING = -2.75
INTER_HOPPING = 0.125 * INTRA_HOPPING
ALPHA = 3.34
DELTA = 0.45
