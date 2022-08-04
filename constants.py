# Constants for CNT Tight-Binding Simulations

import numpy as np

# Numerical Constants

EPS = 1.0e-8
COUPLING_CUTOFF = 1.0e-3
SKIN = 0.1

# Graphene

A_CC = 1.418
A = np.sqrt(3.0) * A_CC

D = np.array([-A_CC, 0.0])

A1 = 0.5 * A * np.array([np.sqrt(3), 1.0])
A2 = 0.5 * A * np.array([np.sqrt(3), -1.0])

ALPHA = 3.34
BETA = 1.42
DELTA = 0.45

INTRA_HOPPING = -2.75
SIGMA_HOPPING = 2.75
PI_HOPPING = 0.164 * SIGMA_HOPPING
