# CNT Class

import constants as const
import numpy as np
from scipy import spatial

import sys
import resource

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

    # Hoppings between nearest neighbours

    aCCMin = self.radius * np.sqrt(2.0*(1-np.cos(const.A_CC/self.radius)))

    chunk_size = 1000
    n_chunks = len(self.sites) // chunk_size
    if (len(self.sites) % chunk_size):
        n_chunks += 1
    chunks = np.array_split(np.arange(len(self.sites)), n_chunks)

    hoppings = []

    for i, chunk1 in enumerate(chunks):
        for j, chunk2 in enumerate(chunks[i:]):
            sites1 = self.sites[chunk1]
            sites2 = self.sites[chunk2]
            dist = spatial.distance.cdist(sites1, sites2)
            if j == 0:
                dist = np.triu(dist)

            hopping_indices = np.argwhere((dist > aCCMin-const.EPS) & (dist < const.A_CC+const.EPS))
            hopping_indices = np.array((chunk1[hopping_indices[:, 0]], chunk2[hopping_indices[:, 1]])).T

            hoppings_chunk = np.zeros((np.shape(hopping_indices)[0], 3))
            hoppings_chunk[:, :-1] = hopping_indices
            hoppings_chunk[:, 2] = np.full(np.shape(hopping_indices)[0], const.INTRA_HOPPING)

            hoppings += [hoppings_chunk]

    self.hoppings = np.concatenate(hoppings, axis=0)

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

    chunk_size = 1000
    n_chunks1 = len(cnt1.sites) // chunk_size
    if (len(cnt1.sites) % chunk_size):
        n_chunks1 += 1
    n_chunks2 = len(cnt2.sites) // chunk_size
    if (len(cnt2.sites) % chunk_size):
        n_chunks2 += 1
    chunks1 = np.array_split(np.arange(len(cnt1.sites)), n_chunks1)
    chunks2 = np.array_split(np.arange(len(cnt2.sites)), n_chunks2)

    hoppings = []

    for chunk1 in chunks1:
        for chunk2 in chunks2:
            sites1 = cnt1.sites[chunk1]
            sites2 = cnt2.sites[chunk2]
            dist = spatial.distance.cdist(sites1, sites2)

            hopping_indices = np.argwhere((dist > aCCMin-const.EPS) & (dist < const.A_CC+const.EPS))
            hopping_indices = np.array((chunk1[hopping_indices[:, 0]], chunk2[hopping_indices[:, 1]])).T

            hoppings_chunk = np.zeros((np.shape(hopping_indices)[0], 3))
            hoppings_chunk[:, :-1] = hopping_indices
            hoppings_chunk[:, 2] = np.full(np.shape(hopping_indices)[0], const.INTRA_HOPPING)

            hoppings += [hoppings_chunk]

    hoppings = np.concatenate(hoppings, axis=0)
    return hoppings

  def interTubeHopping(cnt1, cnt2):

    # split distance calculation into chunks to save memory

    chunk_size = 1000
    n_chunks1 = len(cnt1.sites) // chunk_size
    if (len(cnt1.sites) % chunk_size):
        n_chunks1 += 1
    n_chunks2 = len(cnt2.sites) // chunk_size
    if (len(cnt2.sites) % chunk_size):
        n_chunks2 += 1
    chunks1 = np.array_split(np.arange(len(cnt1.sites)), n_chunks1)
    chunks2 = np.array_split(np.arange(len(cnt2.sites)), n_chunks2)

    print(n_chunks1, n_chunks2)

    hoppings = []

    for chunk1 in chunks1:
        for chunk2 in chunks2:
            sites1 = cnt1.sites[chunk1]
            sites2 = cnt2.sites[chunk2]

            # compute orbital orientation

            siteRelative1 = sites1 - cnt1.origin
            siteRelative2 = sites2 - cnt2.origin
            siteAxialComponent1 = np.dot(siteRelative1, cnt1.axis)
            siteAxialComponent2 = np.dot(siteRelative2, cnt2.axis)
            siteAxial1 = np.tensordot(siteAxialComponent1, cnt1.axis, axes=0)
            siteAxial2 = np.tensordot(siteAxialComponent2, cnt2.axis, axes=0)

            siteOrbital1 = siteRelative1 - siteAxial1
            siteOrbital2 = siteRelative2 - siteAxial2

            siteOrbital1 = siteOrbital1 / np.linalg.norm(siteOrbital1, axis=1, keepdims=True)
            siteOrbital2 = siteOrbital2 / np.linalg.norm(siteOrbital2, axis=1, keepdims=True)

            # compute projections onto sigma and pi orbitals

            sitesRepeat1 = np.repeat(sites1[:, np.newaxis, :], len(sites2), axis=1)
            sitesRepeat2 = np.repeat(sites2[np.newaxis, :, :], len(sites1), axis=0)
            diff = sitesRepeat2 - sitesRepeat1

            dist = np.linalg.norm(diff, axis=2)
            diff = diff / np.repeat(dist[:, :, np.newaxis], 3, axis=2)

            # compute decay terms

            sigmaDecay = np.exp((const.ALPHA - dist) / const.DELTA)
            piDecay = np.exp((const.BETA - dist) / const.DELTA)

            # random vectors orthogonal to diff

            randomVectors = np.random.rand(*np.shape(diff))
            diffComponent = np.sum(randomVectors*diff, axis=2)
            diffOrth1 = randomVectors - np.repeat(diffComponent[:, :, np.newaxis], 3, axis=2) * diff

            diffOrth1 = diffOrth1 / np.linalg.norm(diffOrth1, axis=2, keepdims=True)
            diffOrth2 = np.cross(diff, diffOrth1)

            # compute orbital alignment terms

            siteOrbitalRepeat1 = np.repeat(siteOrbital1[:, np.newaxis, :], len(sites2), axis=1)
            siteOrbitalRepeat2 = np.repeat(siteOrbital2[np.newaxis, :, :], len(sites1), axis=0)

            sigmaAlignment = np.sum(siteOrbitalRepeat1 * diff, axis=2) * np.sum(siteOrbitalRepeat2 * diff, axis=2)

            piAlignment1 = np.sum(siteOrbitalRepeat1 * diffOrth1, axis=2) * np.sum(siteOrbitalRepeat2 * diffOrth1, axis=2)
            piAlignment2 = np.sum(siteOrbitalRepeat1 * diffOrth2, axis=2) * np.sum(siteOrbitalRepeat2 * diffOrth2, axis=2)

            piAlignment = piAlignment1 + piAlignment2

            # compute hoppings

            coupling = const.PI_HOPPING * piDecay * piAlignment - const.SIGMA_HOPPING * sigmaDecay * sigmaAlignment

            # cutoff

            hopping_indices = np.argwhere(np.abs(coupling) > const.COUPLING_CUTOFF)
            coupling = coupling[tuple(hopping_indices.T)]

            hopping_indices = np.array((chunk1[hopping_indices[:, 0]], chunk2[hopping_indices[:, 1]])).T
            hoppings_chunk = np.zeros((np.shape(hopping_indices)[0], 3))
            hoppings_chunk[:, :-1] = hopping_indices
            hoppings_chunk[:, 2] = coupling

            print(hoppings_chunk)
            hoppings += [hoppings_chunk]

    hoppings = np.concatenate(hoppings, axis=0)

    return hoppings

