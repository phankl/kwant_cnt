# CNT Class

import constants as const
import numpy as np

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
    
    self.origin = origin
    self.axis = axis / np.linalg.norm(axis)

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
      lastCell = lastCell[np.dot(lastCell - origin, self.axis) < length]
    
    if cellNumber > 1:
      cells.append(lastCell)
      self.sites = np.concatenate(cells)
    else:
      self.sites = lastCell

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

  def intraTubeHopping(cnt1, cnt2):
    aCCMin = cnt1.radius * np.sqrt(2.0*(1-np.cos(const.A_CC/cnt1.radius)))

    hoppings = []

    siteNumber1 = len(cnt1.sites)
    siteNumber2 = len(cnt2.sites)

    for i in range(siteNumber1):
      site1 = cnt1.sites[i]
      for j in range(siteNumber2):
        site2 = cnt2.sites[j]
        dist = np.linalg.norm(site1 - site2)
        if dist - aCCMin > -const.EPS and dist - const.A_CC < const.EPS:
          #hoppings.append(0.5 * (site1 + site2))
          hoppings.append([i, j, const.INTRA_HOPPING])

    return hoppings

  def interTubeHopping(cnt1, cnt2):
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

        orbitalAngleCos = np.dot(orbitalAxis1, orbitalAxis2) / np.linalg.norm(orbitalAxis1) / np.linalg.norm(orbitalAxis2)
        dist = np.linalg.norm(site1 - site2)

        couplingTerm = orbitalAngleCos * np.exp((const.ALPHA-dist)/const.DELTA)

        if np.abs(couplingTerm) > const.COUPLING_CUTOFF:
          hoppings.append([i, j, -const.INTER_HOPPING*couplingTerm])
    
    return hoppings
