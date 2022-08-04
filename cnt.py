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

    # bin sites

    minCoords = np.amin(sites, axis=0)
    maxCoords = np.amax(sites, axis=0)

    xMin = minCoords[0]
    yMin = minCoords[1]
    zMin = minCoords[2]
    xMax = maxCoords[0]
    yMax = maxCoords[1]
    zMax = maxCoords[2]

    xBinNumber = np.floor((xMax-xMin) / const.A_CC).astype('int')
    yBinNumber = np.floor((yMax-yMin) / const.A_CC).astype('int')
    zBinNumber = np.floor((zMax-zMin) / const.A_CC).astype('int')

    if xBinNumber == 0: xBinNumber = 1
    if yBinNumber == 0: yBinNumber = 1
    if zBinNumber == 0: zBinNumber = 1

    xBins = np.linspace(xMin, xMax, xBinNumber)
    yBins = np.linspace(yMin, yMax, yBinNumber)
    zBins = np.linspace(zMin, zMax, zBinNumber)

    xIndices = np.digitize(sites[:,0], xBins).reshape((-1, 1))
    yIndices = np.digitize(sites[:,1], yBins).reshape((-1, 1))
    zIndices = np.digitize(sites[:,2], zBins).reshape((-1, 1))

    indices = np.concatenate((xIndices, yIndices, zIndices), axis=1)

    hoppings = np.array([])

    for i in range(len(sites)):
      index1 = indices[i]
      site1 = sites[i]

      nearbySiteNumbers = np.argwhere(spatial.distance.cdist([index1], indices2, lambda u, v: np.amax(np.abs(u-v))) < 2)[:,1].reshape((-1))

      if len(nearbySiteNumbers) == 0: continue

      nearbySites = sites[nearbySiteNumbers]

      dist = spatial.distance.cdist([site1], nearbySites).reshape((-1))

      cutoffMask = np.argwhere(np.logical_and(dist-aCCMin+const.EPS > 0, dist-const.A_CC-const.EPS < 0))

      if len(cutoffMask) == 0: continue

      targetIndices = nearbySiteNumbers[cutoffMask].reshape((-1, 1))
      sourceIndices = np.full((len(targetIndices), 1), i)
      hoppingValues = np.full((len(targetIndices), 1), const.INTRA_HOPPING)

      newHoppings = np.concatenate((sourceIndices, targetIndices, hoppingValues), axis=1)

      if len(hoppings) == 0: hoppings = newHoppings
      else: hoppings = np.append(hoppings, newHoppings, axis=0)

    # Hoppings between nearest neighbours

    self.hoppings = hoppings


  def intraTubeHopping(cnt1, cnt2):

    aCCMin = cnt1.radius * np.sqrt(2.0*(1-np.cos(const.A_CC/cnt1.radius)))

    dist = spatial.distance.cdist(cnt1.sites, cnt2.sites)
    hopping_indices = np.argwhere((dist > aCCMin-const.EPS) & (dist < const.A_CC+const.EPS))
    hoppings = np.zeros((np.shape(hopping_indices)[0], 3))
    hoppings[:, :-1] = hopping_indices
    hoppings[:, 2] = np.full(np.shape(hopping_indices)[0], const.INTRA_HOPPING)

    return hoppings

  def interTubeHoppingOld(cnt1, cnt2):

    cutoffDistance = const.ALPHA - const.DELTA*np.log(const.COUPLING_CUTOFF)

    # compute orbital orientation

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

    # compute hoppings

    dist = spatial.distance.cdist(cnt1.sites, cnt2.sites)
    orbitalAngleCos = np.abs(siteOrbitalNormalised1 @ siteOrbitalNormalised2.T)

    hopping_indices = np.argwhere(dist < cutoffDistance)

    dist = dist[tuple(hopping_indices.T)]
    decay = np.exp((const.ALPHA - dist) / const.DELTA)
    orbitalAngleCos = orbitalAngleCos[tuple(hopping_indices.T)]
    coupling = np.multiply(decay, orbitalAngleCos)

    cutoffMask = np.argwhere(coupling > const.COUPLING_CUTOFF).T[0]
    hopping_indices = hopping_indices[cutoffMask]
    coupling = coupling[cutoffMask]

    hoppings = np.zeros((np.shape(hopping_indices)[0], 3))
    hoppings[:, :-1] = hopping_indices
    hoppings[:, 2] = -const.INTER_HOPPING * coupling

    return hoppings

  def interTubeHopping(cnt1, cnt2):

    cutoffDistance = const.ALPHA - const.DELTA*np.log(const.COUPLING_CUTOFF)

    # compute orbital orientation

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

    siteOrbital1 = siteOrbital1 / siteOrbitalNorms1
    siteOrbital2 = siteOrbital2 / siteOrbitalNorms2

    siteOrbitalRepeat1 = np.repeat(siteOrbital1[:, np.newaxis, :], len(cnt2.sites), axis=1)
    siteOrbitalRepeat2 = np.repeat(siteOrbital2[np.newaxis, :, :], len(cnt1.sites), axis=0)

    # compute projections onto sigma and pi orbitals

    sitesRepeat1 = np.repeat(cnt1.sites[:, np.newaxis, :], len(cnt2.sites), axis=1)
    sitesRepeat2 = np.repeat(cnt2.sites[np.newaxis, :, :], len(cnt1.sites), axis=0)
    diff = sitesRepeat2 - sitesRepeat1
    dist = np.linalg.norm(diff, axis=2)
    diff = diff / np.repeat(dist[:, :, np.newaxis], 3, axis=2)

    # random vectors orthogonal to diff

    randomVectors = np.random.rand(*np.shape(diff))
    diffComponent = np.sum(randomVectors*diff, axis=2)
    diffOrth1 = randomVectors - np.repeat(diffComponent[:, :, np.newaxis], 3, axis=2) * diff
    diffOrthNorms1 = np.linalg.norm(diffOrth1, axis=2, keepdims=True)
    diffOrth1 = diffOrth1 / diffOrthNorms1
    diffOrth2 = np.cross(diff, diffOrth1)

    # compute orbital alignment terms

    sigmaAlignment = np.sum(siteOrbitalRepeat1 * diff, axis=2) * np.sum(siteOrbitalRepeat2 * diff, axis=2)

    piAlignment1 = np.sum(siteOrbitalRepeat1 * diffOrth1, axis=2) * np.sum(siteOrbitalRepeat2 * diffOrth1, axis=2)
    piAlignment2 = np.sum(siteOrbitalRepeat1 * diffOrth2, axis=2) * np.sum(siteOrbitalRepeat2 * diffOrth2, axis=2)
    piAlignment = piAlignment1 + piAlignment2

    # compute decay terms

    sigmaDecay = np.exp((const.ALPHA - dist) / const.DELTA)
    piDecay = np.exp((const.BETA - dist) / const.DELTA)

    # compute hoppings

    coupling = const.PI_HOPPING * piDecay * piAlignment - const.SIGMA_HOPPING * sigmaDecay * sigmaAlignment

    # cutoff

    hopping_indices = np.argwhere(np.abs(coupling) > const.COUPLING_CUTOFF)
    coupling = coupling[tuple(hopping_indices.T)]

    hoppings = np.zeros((np.shape(hopping_indices)[0], 3))
    hoppings[:, :-1] = hopping_indices
    hoppings[:, 2] = coupling

    return hoppings

