# CNT Unit Cell Class

import constants as const
import numpy as np

class UnitCell:
  
  axis = np.array([0.0, 0.0, 1.0])

  def __init__(self, n, m):
    
    self.n = n
    self.m = m
 
    # Find atoms in unit cell on graphene sheet

    nR = np.gcd(2*n+m, 2*m+n)
    nC = 4 * (n*n + n*m + m*m) // nR
    t1 = (2*m+n) // nR
    t2 = -(2*n+m) // nR

    chiralVector = n*const.A1 + m*const.A2
    axisVector = t1*const.A1 + t2*const.A2

    chiralVectorLength = np.linalg.norm(chiralVector)
    axisVectorLength = np.linalg.norm(axisVector)

    self.latticeVector = axisVectorLength * self.axis

    chiralVectorNormalised = chiralVector / chiralVectorLength
    axisVectorNormalised = axisVector / axisVectorLength

    radius = 0.5 * chiralVectorLength / np.pi

    self.radius = radius
    self.length = axisVectorLength
   
    nMin = min(0, n, t1, n+t1)
    nMax = max(0, n, t1, n+t1)
    mMin = min(0, m, t2, m+t2)
    mMax = max(0, m, t2, m+t2)

    nRange = range(nMin, nMax+1)
    mRange = range(mMin, mMax+1)

    siteCandidatesA = np.array([n_*const.A1 + m_*const.A2 for n_ in nRange for m_ in mRange])
    siteCandidatesB = np.array([n_*const.A1 + m_*const.A2 + const.D for n_ in nRange for m_ in mRange])

    siteCandidates = np.concatenate((siteCandidatesA, siteCandidatesB))

    # Check if candidates are in unit cell and compute 3D coordinates in CNT

    sites = []

    for site in siteCandidates:
      circ = np.dot(site, chiralVectorNormalised)
      z = np.dot(site, axisVectorNormalised)

      if circ > -const.EPS and chiralVectorLength-circ > const.EPS and z > -const.EPS and axisVectorLength-z > const.EPS:
        phi = circ / radius
        x = radius * np.sin(phi)
        y = radius * np.cos(phi)
        sites.append([x, y, z])

    self.sites = np.array(sites)
    
    # Minimum nearest-neighbour distance

    aCCMin = self.radius * np.sqrt(2.0*(1-np.cos(const.A_CC/self.radius)))

    # Hoppings between nearest neighbours

    hoppings = []

    siteNumber = len(self.sites)

    for i in range(siteNumber):
      site1 = self.sites[i]
      for j in range(i+1, siteNumber):
        site2 = self.sites[j]
        dist = np.linalg.norm(site1 - site2)
        if dist - aCCMin > -const.EPS and dist - const.A_CC < const.EPS:
          #hoppings.append(0.5 * (site1 + site2))
          hoppings.append([i, j, const.INTRA_HOPPING])

    self.hoppings = hoppings
