# CNT Class

import constants as const
import numpy as np
from scipy import spatial

import unit_cell

class CNT:
  
  def __init__(self, n, m, length, cellMode=True, rot=0.0, origin=(0.0, 0.0, 0.0), axis=(0.0, 0.0, 1.0)):

    self.n = n
    self.m = m
    
    unitCell = unit_cell.UnitCell(n, m)
    self.unitCell = unitCell

    if cellMode:
      self.length = length * unitCell.length
    else:
      self.length = length
    
    self.origin = np.array(origin)
    self.axis = np.array(axis / np.linalg.norm(axis))

    self.radius = unitCell.radius

    cellLength = unitCell.length
    cellSites = unitCell.sites
    cellAxis = unitCell.axis
   
    # Rotate unit cell along origin axis

    rotationAxis = cellAxis
    rotationAngleCos = np.cos(rot)
    rotationAngleSin = np.sin(rot)

    cellSites = [rotationAngleCos*site + rotationAngleSin*np.cross(rotationAxis, site) + (1.0-rotationAngleCos)*np.dot(rotationAxis, site)*rotationAxis for site in cellSites]

    # Rotate unit cell into axis

    rotationAxis = np.cross(cellAxis, self.axis)
    rotationAngleCos = np.dot(cellAxis, self.axis)
    if rotationAngleCos + 1.0 < const.EPS:
      cellSites = -cellSites
    else:
      cellSites = [rotationAngleCos*site + np.cross(rotationAxis, site) + np.dot(rotationAxis, site)/(1.0+rotationAngleCos)*rotationAxis for site in cellSites]

    latticeVector = cellLength * np.array(self.axis)
    
    cellNumber = length
    if not cellMode:
      cellNumber = np.ceil(length/cellLength).astype('int')

    cells = [cellSites + i*latticeVector + origin for i in range(cellNumber-1)]
    lastCell = cellSites + (cellNumber-1)*latticeVector + origin
    if not cellMode:
      lastCell = lastCell[np.dot(lastCell - origin, self.axis) < length + const.EPS]
    
    if cellNumber > 1:
      cells.append(lastCell)
      self.sites = np.concatenate(cells)
    else:
      self.sites = lastCell

    # Minimum nearest-neighbour distance

    aCCMin = self.radius * np.sqrt(2.0*(1-np.cos(const.A_CC/self.radius)))

    # Hoppings between nearest neighbours

    dist = spatial.distance.cdist(self.sites, self.sites)
    hoppingIndices = np.argwhere(np.logical_and(dist-aCCMin+const.EPS > 0, dist-const.A_CC-const.EPS < 0))
    hoppingIndicesMask = np.argwhere(hoppingIndices[:,1] > hoppingIndices[:,0])[:,0]
    hoppingIndices = hoppingIndices[hoppingIndicesMask]

    hoppings = np.full((len(hoppingIndices), 3), const.INTRA_HOPPING)
    hoppings[:,:-1] = hoppingIndices
  
    self.hoppings = hoppings

  def sliceStart(self, length):
    newSites = []
    for site in self.sites:
      siteRelative = site - self.origin
      startDistance = np.dot(siteRelative, self.axis)
      if startDistance - length > -const.EPS:
        newSites.append(site)

    self.sites = np.array(newSites)
    self.origin = self.origin + length*self.axis
    self.length = self.length - length
    
    # Minimum nearest-neighbour distance

    aCCMin = self.radius * np.sqrt(2.0*(1-np.cos(const.A_CC/self.radius)))

    # Hoppings between nearest neighbours

    dist = spatial.distance.cdist(self.sites, self.sites)
    hoppingIndices = np.argwhere(np.logical_and(dist-aCCMin+const.EPS > 0, dist-const.A_CC-const.EPS < 0))
    hoppingIndicesMask = np.argwhere(hoppingIndices[:,1] > hoppingIndices[:,0])[:,0]
    hoppingIndices = hoppingIndices[hoppingIndicesMask]

    hoppings = np.full((len(hoppingIndices), 3), const.INTRA_HOPPING)
    hoppings[:,:-1] = hoppingIndices
    
    self.hoppings = hoppings


  def intraTubeHopping(cnt1, cnt2):
    aCCMin = cnt1.radius * np.sqrt(2.0*(1-np.cos(const.A_CC/cnt1.radius)))

    dist = spatial.distance.cdist(cnt1.sites, cnt2.sites)
    hoppingIndices = np.argwhere(np.logical_and(dist-aCCMin+const.EPS > 0, dist-const.A_CC-const.EPS < 0))

    hoppings = np.full((len(hoppingIndices), 3), const.INTRA_HOPPING)
    hoppings[:,:-1] = hoppingIndices
    
    return hoppings

  def interTubeHopping(cnt1, cnt2):

    cutoffDistance = const.ALPHA - const.DELTA*np.log(const.COUPLING_CUTOFF)
    
    dist = spatial.distance.cdist(cnt1.sites, cnt2.sites)
    exponentialDecay = np.where(dist < cutoffDistance, np.exp((const.ALPHA-dist)/const.DELTA), 0.0)

    siteRelative1 = cnt1.sites - cnt1.origin
    siteRelative2 = cnt2.sites - cnt2.origin
    siteAxialComponent1 = np.dot(siteRelative1, cnt1.axis)
    siteAxialComponent2 = np.dot(siteRelative2, cnt2.axis)
    siteAxial1 = np.tensordot(siteAxialComponent1, cnt1.axis, axes=0)
    siteAxial2 = np.tensordot(siteAxialComponent2, cnt2.axis, axes=0)

    siteOrbital1 = siteRelative1 - siteAxial1
    siteOrbital2 = siteRelative2 - siteAxial2

    siteOrbitalNorms1 = np.linalg.norm(siteOrbital1, axis=1, keepdims=True)
    siteOrbitalNorms2 = np.linalg.norm(siteOrbital2, axis=1, keepdims=True)

    siteOrbitalNormalised1 = siteOrbital1 / siteOrbitalNorms1
    siteOrbitalNormalised2 = siteOrbital2 / siteOrbitalNorms2

    orbitalAngleCos = np.abs(np.dot(siteOrbitalNormalised1, siteOrbitalNormalised2.T))
     
    couplingTerm = np.multiply(exponentialDecay, orbitalAngleCos)

    hoppingIndices = np.argwhere(couplingTerm > const.COUPLING_CUTOFF)
    hoppingValues = -const.INTER_HOPPING * couplingTerm[hoppingIndices.T[0], hoppingIndices.T[1]]
    hoppingValues = np.reshape(hoppingValues, (-1, 1))

    hoppings = np.append(hoppingIndices, hoppingValues, axis=1)

    '''
    hoppings = []

    siteNumber1 = len(cnt1.sites)
    siteNumber2 = len(cnt2.sites)

    for i in range(siteNumber1):
      site1 = cnt1.sites[i]
      siteRelative1 = site1 - cnt1.origin
      siteAxial1 = np.dot(siteRelative1, cnt1.axis) * cnt1.axis
      orbitalAxis1 = siteRelative1 - siteAxial1

      for j in range(siteNumber2):
        site2 = cnt2.sites[j]
        siteRelative2 = site2 - cnt2.origin
        siteAxial2 = np.dot(siteRelative2, cnt2.axis) * cnt2.axis
        orbitalAxis2 = siteRelative2 - siteAxial2

        orbitalAngleCos = np.abs(np.dot(orbitalAxis1, orbitalAxis2) / np.linalg.norm(orbitalAxis1) / np.linalg.norm(orbitalAxis2))
        dist = np.linalg.norm(site1 - site2)

        couplingTerm = orbitalAngleCos * np.exp((const.ALPHA-dist)/const.DELTA)

        if np.abs(couplingTerm) > const.COUPLING_CUTOFF:
          hoppings.append([i, j, -const.INTER_HOPPING*couplingTerm])
    '''

    return hoppings
