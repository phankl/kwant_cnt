# Different CNT Systems for Quantum Electron Transport Problems with KWANT

import kwant
import matplotlib.pyplot as plt
import numpy as np

import constants as const
import cnt

class System:

  def transmission(self, i, j, energies):

    # Transmission probability as function of energy

    transmissionProbability = np.array([kwant.smatrix(self.systemFinalized, energy).transmission(i, j) for energy in energies])

    return transmissionProbability

  def plotSystem(self):
    kwant.plot(self.systemFinalized, site_size=0.1, hop_lw=0.1)

class SingleCNT(System):

  def __init__(self, n, m, cellNumber=5):

    self.n = n
    self.m = m

    deviceUnitCell = cnt.CNT(n, m, 1, axis=(1.0, 0.0, 0.0))
    offsetUnitCell = cnt.CNT(n, m, 1, axis=(1.0, 0.0, 0.0), origin=(deviceUnitCell.length, 0.0, 0.0))

    unitCellHoppings = cnt.CNT.intraTubeHopping(deviceUnitCell, offsetUnitCell)

    # Kwant setup

    latticeVector = deviceUnitCell.length * deviceUnitCell.axis

    system = kwant.Builder()

    lattice = kwant.lattice.Polyatomic([latticeVector], deviceUnitCell.sites, norbs=1)

    # Scattering region

    # Sites

    for i in range(cellNumber):
      for j in range(len(deviceUnitCell.sites)):
        system[lattice.sublattices[j](i)] = 0.0

    # Intra-cell hopping

    for i in range(cellNumber):
      for hopping in deviceUnitCell.hoppings:
        system[lattice.sublattices[hopping[0].astype('int')](i), lattice.sublattices[hopping[1].astype('int')](i)] = hopping[2]

    # Inter-cell hopping

    for i in range(cellNumber-1):
      for hopping in unitCellHoppings:
        system[lattice.sublattices[hopping[0].astype('int')](i), lattice.sublattices[hopping[1].astype('int')](i+1)] = hopping[2]



    # Left lead

    symmetryLead0 = kwant.TranslationalSymmetry(-latticeVector)
    lead0 = kwant.Builder(symmetryLead0)

    for i in range(len(deviceUnitCell.sites)):
      lead0[lattice.sublattices[i](-1)] = 0.0

    for hopping in deviceUnitCell.hoppings:
      lead0[lattice.sublattices[hopping[0].astype('int')](0), lattice.sublattices[hopping[1].astype('int')](0)] = hopping[2]

    for hopping in unitCellHoppings:
      lead0[lattice.sublattices[hopping[0].astype('int')](0), lattice.sublattices[hopping[1].astype('int')](1)] = hopping[2]

    system.attach_lead(lead0)

    # Right lead

    symmetryLead1 = kwant.TranslationalSymmetry(latticeVector)
    lead1 = kwant.Builder(symmetryLead1)

    for i in range(len(deviceUnitCell.sites)):
      lead1[lattice.sublattices[i](-1)] = 0.0

    for hopping in deviceUnitCell.hoppings:
      lead1[lattice.sublattices[hopping[0].astype('int')](0), lattice.sublattices[hopping[1].astype('int')](0)] = hopping[2]

    for hopping in unitCellHoppings:
      lead1[lattice.sublattices[hopping[0].astype('int')](0), lattice.sublattices[hopping[1].astype('int')](1)] = hopping[2]

    system.attach_lead(lead1)

    self.systemFinalized = system.finalized()
    self.leadFinalized = lead0.finalized()

  def plotBandStructure(self):
    kwant.plotter.bands(self.leadFinalized, momenta=1000, show=False)
    plt.xlabel("Momentum [m^(-1)]")
    plt.ylabel("Energy [eV]")
    plt.show()


class InfiniteContact(System):
  def __init__(self, n, m, distance, rot1=0.0, rot2=0.0, offset=0.0):

    self.n = n
    self.m = m
    self.distance = distance
    self.rot1 = rot1
    self.rot2 = rot2
    self.offset = offset

    cntUnitCell = cnt.CNT(n, m, 1)

    cutoffDistance = const.ALPHA - const.DELTA*np.log(const.COUPLING_CUTOFF) + np.linalg.norm(offset)
    cutoffCellNumber = np.ceil(cutoffDistance / cntUnitCell.length).astype('int')

    cntCell1 = cnt.CNT(n, m, cutoffCellNumber, axis=(1.0, 0.0, 0.0), rot=rot1)
    offsetCell1 = cnt.CNT(n, m, cutoffCellNumber, axis=(1.0, 0.0, 0.0), rot=rot1, origin=(cntCell1.length, 0.0, 0.0))

    cntCell2 = cnt.CNT(n, m, cutoffCellNumber, axis=(1.0, 0.0, 0.0), rot=rot2, origin=(offset, 0.0, -distance))
    offsetCell2 = cnt.CNT(n, m, cutoffCellNumber, axis=(1.0, 0.0, 0.0), rot=rot2, origin=(offset+cntCell2.length, 0.0, -distance))

    # Intra-tube hoppings

    intraCellHoppings1 = cntCell1.hoppings
    intraCellHoppings2 = cntCell2.hoppings

    interCellHoppings1 = cnt.CNT.intraTubeHopping(cntCell1, offsetCell1)
    interCellHoppings2 = cnt.CNT.intraTubeHopping(cntCell2, offsetCell2)

    # Inter-tube hoppings

    interTubeIntraCellHoppings = cnt.CNT.interTubeHopping(cntCell1, cntCell2)
    interTubeInterCellHoppings12 = cnt.CNT.interTubeHopping(cntCell1, offsetCell2)
    interTubeInterCellHoppings21 = cnt.CNT.interTubeHopping(cntCell2, offsetCell1)

    # Kwant setup

    latticeVector = cntCell1.length * cntCell1.axis

    system = kwant.Builder()

    lattice1 = kwant.lattice.Polyatomic([latticeVector], cntCell1.sites, norbs=1)
    lattice2 = kwant.lattice.Polyatomic([latticeVector], cntCell2.sites, norbs=1)

    # Scattering region

    # Sites

    for i in range(len(cntCell1.sites)):
      system[lattice1.sublattices[i](0)] = 0.0
    for i in range(len(cntCell2.sites)):
      system[lattice2.sublattices[i](0)] = 0.0

    # Intra-tube hoppings

    for hopping in intraCellHoppings1:
      system[lattice1.sublattices[hopping[0].astype('int')](0), lattice1.sublattices[hopping[1].astype('int')](0)] = hopping[2]
    for hopping in intraCellHoppings2:
      system[lattice2.sublattices[hopping[0].astype('int')](0), lattice2.sublattices[hopping[1].astype('int')](0)] = hopping[2]

    kwant.plot(system, site_size=0.1, hop_lw=0.1)

    # Inter-tube hoppings

    for hopping in interTubeIntraCellHoppings:
      system[lattice1.sublattices[hopping[0].astype('int')](0), lattice2.sublattices[hopping[1].astype('int')](0)] = hopping[2]


    # Left lead

    symmetryLead0 = kwant.TranslationalSymmetry(-latticeVector)
    lead0 = kwant.Builder(symmetryLead0)

    # Sites

    for i in range(len(cntCell1.sites)):
      lead0[lattice1.sublattices[i](-1)] = 0.0
    for i in range(len(cntCell2.sites)):
      lead0[lattice2.sublattices[i](-1)] = 0.0

    # Intra-tube hoppings inside unit cell

    for hopping in intraCellHoppings1:
      lead0[lattice1.sublattices[hopping[0].astype('int')](0), lattice1.sublattices[hopping[1].astype('int')](0)] = hopping[2]
    for hopping in intraCellHoppings2:
      lead0[lattice2.sublattices[hopping[0].astype('int')](0), lattice2.sublattices[hopping[1].astype('int')](0)] = hopping[2]

    # Intra-tube hoppings, between unit cells

    for hopping in interCellHoppings1:
      lead0[lattice1.sublattices[hopping[0].astype('int')](0), lattice1.sublattices[hopping[1].astype('int')](1)] = hopping[2]
    for hopping in interCellHoppings2:
      lead0[lattice2.sublattices[hopping[0].astype('int')](0), lattice2.sublattices[hopping[1].astype('int')](1)] = hopping[2]

    # Inter-tube hoppings, inside unit cell

    for hopping in interTubeIntraCellHoppings:
      lead0[lattice1.sublattices[hopping[0].astype('int')](0), lattice2.sublattices[hopping[1].astype('int')](0)] = hopping[2]

    # Inter-tube hoppings, between unit cells

    for hopping in interTubeInterCellHoppings12:
      lead0[lattice1.sublattices[hopping[0].astype('int')](0), lattice2.sublattices[hopping[1].astype('int')](1)] = hopping[2]
    for hopping in interTubeInterCellHoppings21:
      lead0[lattice2.sublattices[hopping[0].astype('int')](0), lattice1.sublattices[hopping[1].astype('int')](1)] = hopping[2]

    system.attach_lead(lead0)

    # Right lead

    symmetryLead1 = kwant.TranslationalSymmetry(latticeVector)
    lead1 = kwant.Builder(symmetryLead1)

    # Sites

    for i in range(len(cntCell1.sites)):
      lead1[lattice1.sublattices[i](0)] = 0.0
    for i in range(len(cntCell2.sites)):
      lead1[lattice2.sublattices[i](0)] = 0.0

    # Intra-tube hoppings inside unit cell

    for hopping in intraCellHoppings1:
      lead1[lattice1.sublattices[hopping[0].astype('int')](0), lattice1.sublattices[hopping[1].astype('int')](0)] = hopping[2]
    for hopping in intraCellHoppings2:
      lead1[lattice2.sublattices[hopping[0].astype('int')](0), lattice2.sublattices[hopping[1].astype('int')](0)] = hopping[2]

    # Intra-tube hoppings, between unit cells

    for hopping in interCellHoppings1:
      lead1[lattice1.sublattices[hopping[0].astype('int')](0), lattice1.sublattices[hopping[1].astype('int')](1)] = hopping[2]
    for hopping in interCellHoppings2:
      lead1[lattice2.sublattices[hopping[0].astype('int')](0), lattice2.sublattices[hopping[1].astype('int')](1)] = hopping[2]

    # Inter-tube hoppings, inside unit cell

    for hopping in interTubeIntraCellHoppings:
      lead1[lattice1.sublattices[hopping[0].astype('int')](0), lattice2.sublattices[hopping[1].astype('int')](0)] = hopping[2]

    # Inter-tube hoppings, between unit cells

    for hopping in interTubeInterCellHoppings12:
      lead1[lattice1.sublattices[hopping[0].astype('int')](0), lattice2.sublattices[hopping[1].astype('int')](1)] = hopping[2]
    for hopping in interTubeInterCellHoppings21:
      lead1[lattice2.sublattices[hopping[0].astype('int')](0), lattice1.sublattices[hopping[1].astype('int')](1)] = hopping[2]

    system.attach_lead(lead1)

    self.systemFinalized = system.finalized()
    self.leadFinalized = lead0.finalized()

  def plotBandStructure(self):
    kwant.plotter.bands(self.leadFinalized, momenta=np.linspace(0.6*np.pi, 0.7*np.pi, num=1000), show=False)
    #kwant.plotter.bands(self.leadFinalized, momenta=5000, show=False)
    plt.xlabel("Momentum [m^(-1)]")
    plt.ylabel("Energy [eV]")
    plt.xlim(0.6*np.pi, 0.7*np.pi)
    plt.ylim(-0.1, 0.1)
    plt.show()


class FiniteContact(System):
  def __init__(self, n1, m1, n2, m2, overlap, distance, rot1=0.0, rot2=0.0, offset=0.0, leads=(True, True, True, True)):

    self.n1 = n1
    self.m1 = m1
    self.n2 = n2
    self.m2 = m2
    self.overlap = overlap
    self.distance = distance
    self.rot1 = rot1
    self.rot2 = rot2
    self.offset = offset
    self.leads = leads

    xAxis = (1.0, 0.0, 0.0)

    cntUnitCell1 = cnt.CNT(n1, m1, 1, axis=xAxis, rot=rot1)
    cntUnitCell2 = cnt.CNT(n2, m2, 1, axis=xAxis, origin=(offset, 0.0, -distance), rot=rot2)

    cutoffDistance = const.ALPHA - const.DELTA*np.log(const.COUPLING_CUTOFF)

    overlapCellNumber1 = np.ceil(overlap/cntUnitCell1.length).astype('int')
    overlapCellNumber2 = np.ceil(overlap/cntUnitCell2.length).astype('int')
    cutoffCellNumber1 = np.ceil(cutoffDistance/cntUnitCell1.length).astype('int')
    cutoffCellNumber2 = np.ceil(cutoffDistance/cntUnitCell2.length).astype('int')

    overlapCellStartIndex1 = np.floor(offset/cntUnitCell1.length).astype('int')
    bufferCellStartIndexLeft1 = overlapCellStartIndex1 - cutoffCellNumber1
    bufferCellStartIndexRight1 = overlapCellStartIndex1 + overlapCellNumber1
    overlapCellStartIndex2 = 0
    bufferCellStartIndexLeft2 = overlapCellStartIndex2 - cutoffCellNumber2
    bufferCellStartIndexRight2 = overlapCellStartIndex2 + overlapCellNumber2

    # Construct scattering geometry

    if leads[0]:
      origin1 = bufferCellStartIndexLeft1 * cntUnitCell1.length
    else:
      origin1 = overlapCellStartIndex1 * cntUnitCell1.length

    if leads[2]:
      origin2 = bufferCellStartIndexLeft2 * cntUnitCell2.length + offset
    else:
      origin2 = overlapCellStartIndex2 * cntUnitCell2.length + offset

    totalCellNumber1 = overlapCellNumber1
    if leads[0]:
      totalCellNumber1 += cutoffCellNumber1
    if leads[1]:
      totalCellNumber1 += cutoffCellNumber1

    totalCellNumber2 = overlapCellNumber2
    if leads[2]:
      totalCellNumber2 += cutoffCellNumber2
    if leads[3]:
      totalCellNumber2 += cutoffCellNumber2

    if leads[0] and leads[1]:
      cell1 = cnt.CNT(n1, m1, totalCellNumber1, axis=xAxis, origin=(origin1, 0.0, 0.0), rot=rot1)
    elif leads[0]:
      cell1 = cnt.CNT(n1, m1, cutoffCellNumber1*cntUnitCell1.length + offset + overlap, cellMode=False, axis=xAxis, origin=(origin1, 0.0, 0.0), rot=rot1)
    elif leads[1]:
      cell1 = cnt.CNT(n1, m1, totalCellNumber1, axis=xAxis, origin=(origin1, 0.0, 0.0), rot=rot1)
      cell1.sliceStart(offset)
    else:
      cell1 = cnt.CNT(n1, m1, offset + overlap, cellMode=False, axis=xAxis, origin=(origin1, 0.0, 0.0), rot=rot1)
      cell1.sliceStart(offset)

    if leads[2] and leads[3]:
      cell2 = cnt.CNT(n2, m2, totalCellNumber2, axis=xAxis, origin=(origin2, 0.0, -distance), rot=rot2)
    elif leads[2]:
      cell2 = cnt.CNT(n2, m2, cutoffCellNumber2*cntUnitCell2.length + overlap, cellMode=False, axis=xAxis, origin=(origin2, 0.0, -distance), rot=rot2)
    elif leads[3]:
      cell2 = cnt.CNT(n2, m2, totalCellNumber2, axis=xAxis, origin=(origin2, 0.0, -distance), rot=rot2)
    else:
      cell2 = cnt.CNT(n2, m2, overlap, cellMode=False, axis=xAxis, origin=(origin2, 0.0, -distance), rot=rot2)

    intraCellHoppings1 = cell1.hoppings
    intraCellHoppings2 = cell2.hoppings

    interTubeHoppingsAll = cnt.CNT.interTubeHopping(cell1, cell2)

    # Filter for hoppings in interaction slice
    interTubeHoppings = []
    for hopping in interTubeHoppingsAll:
      site1 = cell1.sites[hopping[0].astype('int')]
      site2 = cell2.sites[hopping[1].astype('int')]

      if (site1[0] > offset-const.EPS and site1[0] < offset+overlap+const.EPS) or (site2[0] > offset-const.EPS and site2[0] < offset+overlap+const.EPS):
        interTubeHoppings.append(hopping)

    # Lead unit cells in scattering region

    if leads[0]:
      leadUnitCellLeft1 = cnt.CNT(n1, m1, 1, axis=xAxis, origin=(origin1-cntUnitCell1.length, 0.0, 0.0), rot=rot1)
      intraCellHoppingsLeft1 = leadUnitCellLeft1.hoppings
      interCellHoppingsLeft1 = cnt.CNT.intraTubeHopping(cell1, leadUnitCellLeft1)
    if leads[1]:
      leadUnitCellRight1 = cnt.CNT(n1, m1, 1, axis=xAxis, origin=(origin1+cell1.length, 0.0, 0.0), rot=rot1)
      intraCellHoppingsRight1 = leadUnitCellRight1.hoppings
      interCellHoppingsRight1 = cnt.CNT.intraTubeHopping(cell1, leadUnitCellRight1)
    if leads[2]:
      leadUnitCellLeft2 = cnt.CNT(n2, m2, 1, axis=xAxis, origin=(origin2-cntUnitCell2.length, 0.0, -distance), rot=rot2)
      intraCellHoppingsLeft2 = leadUnitCellLeft2.hoppings
      interCellHoppingsLeft2 = cnt.CNT.intraTubeHopping(cell2, leadUnitCellLeft2)
    if leads[3]:
      leadUnitCellRight2 = cnt.CNT(n2, m2, 1, axis=xAxis, origin=(origin2+cell2.length, 0.0, -distance), rot=rot2)
      intraCellHoppingsRight2 = leadUnitCellRight2.hoppings
      interCellHoppingsRight2 = cnt.CNT.intraTubeHopping(cell2, leadUnitCellRight2)

    # Lead hoppings

    if leads[0] or leads[1]:
      offsetCNTUnitCell1 = cnt.CNT(n1, m1, 1, axis=xAxis, origin=(cntUnitCell1.length, 0.0, 0.0), rot=rot1)
      intraCellLeadHoppings1 = cntUnitCell1.hoppings
      interCellLeadHoppings1 = cnt.CNT.intraTubeHopping(cntUnitCell1, offsetCNTUnitCell1)
    if leads[2] or leads[3]:
      offsetCNTUnitCell2 = cnt.CNT(n2, m2, 1, axis=xAxis, origin=(cntUnitCell2.length+offset, 0.0, -distance), rot=rot2)
      intraCellLeadHoppings2 = cntUnitCell2.hoppings
      interCellLeadHoppings2 = cnt.CNT.intraTubeHopping(cntUnitCell2, offsetCNTUnitCell2)

    # Kwant setup

    # Interacting scattering region

    system = kwant.Builder()

    latticeVectorDevice1 = cell1.length * cell1.axis
    latticeVectorDevice2 = cell2.length * cell2.axis

    if leads[0] or leads[1]:
      latticeVectorLead1 = cntUnitCell1.length * cntUnitCell1.axis
    if leads[2] or leads[3]:
      latticeVectorLead2 = cntUnitCell2.length * cntUnitCell2.axis

    latticeDevice1 = kwant.lattice.Polyatomic([latticeVectorDevice1], cell1.sites, norbs=1)
    latticeDevice2 = kwant.lattice.Polyatomic([latticeVectorDevice2], cell2.sites, norbs=1)

    # Sites
    for i in range(len(cell1.sites)):
      system[latticeDevice1.sublattices[i](0)] = 0.0
    for i in range(len(cell2.sites)):
      system[latticeDevice2.sublattices[i](0)] = 0.0

    # Intra-tube hopping
    for hopping in intraCellHoppings1:
      system[latticeDevice1.sublattices[hopping[0].astype('int')](0), latticeDevice1.sublattices[hopping[1].astype('int')](0)] = hopping[2]
    for hopping in intraCellHoppings2:
      system[latticeDevice2.sublattices[hopping[0].astype('int')](0), latticeDevice2.sublattices[hopping[1].astype('int')](0)] = hopping[2]

    # Inter-tube hopping
    for hopping in interTubeHoppings:
      system[latticeDevice1.sublattices[hopping[0].astype('int')](0), latticeDevice2.sublattices[hopping[1].astype('int')](0)] = hopping[2]

    # Leads (including non-interacting part of scattering region)

    if leads[0]:
      latticeLeadLeft1 = kwant.lattice.Polyatomic([latticeVectorLead1], leadUnitCellLeft1.sites, norbs=1)
      symmetryLeadLeft1 = kwant.TranslationalSymmetry(-latticeVectorLead1)
      leadLeft1 = kwant.Builder(symmetryLeadLeft1)

      for i in range(len(leadUnitCellLeft1.sites)):
        system[latticeLeadLeft1.sublattices[i](0)] = 0.0
        leadLeft1[latticeLeadLeft1.sublattices[i](0)] = 0.0

      for hopping in intraCellHoppingsLeft1:
        system[latticeLeadLeft1.sublattices[hopping[0].astype('int')](0), latticeLeadLeft1.sublattices[hopping[1].astype('int')](0)] = hopping[2]
      for hopping in interCellHoppingsLeft1:
        system[latticeDevice1.sublattices[hopping[0].astype('int')](0), latticeLeadLeft1.sublattices[hopping[1].astype('int')](0)] = hopping[2]
      for hopping in intraCellLeadHoppings1:
        leadLeft1[latticeLeadLeft1.sublattices[hopping[0].astype('int')](0), latticeLeadLeft1.sublattices[hopping[1].astype('int')](0)] = hopping[2]
      for hopping in interCellLeadHoppings1:
        leadLeft1[latticeLeadLeft1.sublattices[hopping[0].astype('int')](0), latticeLeadLeft1.sublattices[hopping[1].astype('int')](1)] = hopping[2]

      system.attach_lead(leadLeft1)

    if leads[1]:
      latticeLeadRight1 = kwant.lattice.Polyatomic([latticeVectorLead1], leadUnitCellRight1.sites, norbs=1)
      symmetryLeadRight1 = kwant.TranslationalSymmetry(latticeVectorLead1)
      leadRight1 = kwant.Builder(symmetryLeadRight1)

      for i in range(len(leadUnitCellRight1.sites)):
        system[latticeLeadRight1.sublattices[i](0)] = 0.0
        leadRight1[latticeLeadRight1.sublattices[i](0)] = 0.0

      for hopping in intraCellHoppingsRight1:
        system[latticeLeadRight1.sublattices[hopping[0].astype('int')](0), latticeLeadRight1.sublattices[hopping[1].astype('int')](0)] = hopping[2]
      for hopping in interCellHoppingsRight1:
        system[latticeDevice1.sublattices[hopping[0].astype('int')](0), latticeLeadRight1.sublattices[hopping[1].astype('int')](0)] = hopping[2]
      for hopping in intraCellLeadHoppings1:
        leadRight1[latticeLeadRight1.sublattices[hopping[0].astype('int')](0), latticeLeadRight1.sublattices[hopping[1].astype('int')](0)] = hopping[2]
      for hopping in interCellLeadHoppings1:
        leadRight1[latticeLeadRight1.sublattices[hopping[0].astype('int')](0), latticeLeadRight1.sublattices[hopping[1].astype('int')](1)] = hopping[2]

      system.attach_lead(leadRight1)

    if leads[2]:
      latticeLeadLeft2 = kwant.lattice.Polyatomic([latticeVectorLead2], leadUnitCellLeft2.sites, norbs=1)
      symmetryLeadLeft2 = kwant.TranslationalSymmetry(-latticeVectorLead2)
      leadLeft2 = kwant.Builder(symmetryLeadLeft2)

      for i in range(len(leadUnitCellLeft2.sites)):
        system[latticeLeadLeft2.sublattices[i](0)] = 0.0
        leadLeft2[latticeLeadLeft2.sublattices[i](0)] = 0.0

      for hopping in intraCellHoppingsLeft2:
        system[latticeLeadLeft2.sublattices[hopping[0].astype('int')](0), latticeLeadLeft2.sublattices[hopping[1].astype('int')](0)] = hopping[2]
      for hopping in interCellHoppingsLeft2:
        system[latticeDevice2.sublattices[hopping[0].astype('int')](0), latticeLeadLeft2.sublattices[hopping[1].astype('int')](0)] = hopping[2]
      for hopping in intraCellLeadHoppings2:
        leadLeft2[latticeLeadLeft2.sublattices[hopping[0].astype('int')](0), latticeLeadLeft2.sublattices[hopping[1].astype('int')](0)] = hopping[2]
      for hopping in interCellLeadHoppings2:
        leadLeft2[latticeLeadLeft2.sublattices[hopping[0].astype('int')](0), latticeLeadLeft2.sublattices[hopping[1].astype('int')](1)] = hopping[2]

      system.attach_lead(leadLeft2)

    if leads[3]:
      latticeLeadRight2 = kwant.lattice.Polyatomic([latticeVectorLead2], leadUnitCellRight2.sites, norbs=1)
      symmetryLeadRight2 = kwant.TranslationalSymmetry(latticeVectorLead2)
      leadRight2 = kwant.Builder(symmetryLeadRight2)

      for i in range(len(leadUnitCellRight2.sites)):
        system[latticeLeadRight2.sublattices[i](0)] = 0.0
        leadRight2[latticeLeadRight2.sublattices[i](0)] = 0.0

      for hopping in intraCellHoppingsRight2:
        system[latticeLeadRight2.sublattices[hopping[0].astype('int')](0), latticeLeadRight2.sublattices[hopping[1].astype('int')](0)] = hopping[2]
      for hopping in interCellHoppingsRight2:
        system[latticeDevice2.sublattices[hopping[0].astype('int')](0), latticeLeadRight2.sublattices[hopping[1].astype('int')](0)] = hopping[2]
      for hopping in intraCellLeadHoppings2:
        leadRight2[latticeLeadRight2.sublattices[hopping[0].astype('int')](0), latticeLeadRight2.sublattices[hopping[1].astype('int')](0)] = hopping[2]
      for hopping in interCellLeadHoppings2:
        leadRight2[latticeLeadRight2.sublattices[hopping[0].astype('int')](0), latticeLeadRight2.sublattices[hopping[1].astype('int')](1)] = hopping[2]

      system.attach_lead(leadRight2)

      self.systemFinalized = system.finalized()

class SlidingContact(System):
  def __init__(self, n1, m1, n2, m2, overlap, distance, rot1=0.0, rot2=0.0):

    self.n1 = n1
    self.m1 = m1
    self.n2 = n2
    self.m2 = m2
    self.overlap = overlap
    self.distance = distance
    self.rot1 = rot1
    self.rot2 = rot2

    xAxis = (1.0, 0.0, 0.0)

    cntUnitCell1 = cnt.CNT(n1, m1, 1, axis=xAxis, rot=rot1)
    cntUnitCell2 = cnt.CNT(n2, m2, 1, axis=xAxis, rot=rot2)

    cutoffDistance = const.ALPHA - const.DELTA*np.log(const.COUPLING_CUTOFF)

    overlapCellNumber1 = np.ceil(overlap/cntUnitCell1.length).astype('int')
    overlapCellNumber2 = np.ceil(overlap/cntUnitCell2.length).astype('int')
    cutoffCellNumber1 = np.ceil(cutoffDistance/cntUnitCell1.length).astype('int')
    cutoffCellNumber2 = np.ceil(cutoffDistance/cntUnitCell2.length).astype('int')
    totalCellNumber1 = overlapCellNumber1 + cutoffCellNumber1
    totalCellNumber2 = overlapCellNumber2 + cutoffCellNumber2

    # Construct scattering geometry

    unitCellGap1 = cntUnitCell1.length - np.amax(cntUnitCell1.sites, axis=0)[0]

    length1 = totalCellNumber1 * cntUnitCell1.length
    origin1 = unitCellGap1 + overlap - length1

    cell1 = cnt.CNT(n1, m1, totalCellNumber1, axis=xAxis, origin=(origin1, 0.0, 0.0), rot=rot1)
    cell2 = cnt.CNT(n2, m2, totalCellNumber2, axis=xAxis, origin=(0.0, 0.0, -distance), rot=rot2)

    leadUnitCell1 = cnt.CNT(n1, m1, 1, axis=xAxis, origin=(origin1-cntUnitCell1.length, 0.0, 0.0), rot=rot1)
    leadUnitCell2 = cnt.CNT(n2, m2, 1, axis=xAxis, origin=(totalCellNumber2*cntUnitCell2.length, 0.0, -distance), rot=rot2)

    # Scattering region hoppings

    intraCellHoppings1 = cell1.hoppings
    intraCellHoppings2 = cell2.hoppings

    interCellHoppings1 = cnt.CNT.intraTubeHopping(cell1, leadUnitCell1)
    interCellHoppings2 = cnt.CNT.intraTubeHopping(cell2, leadUnitCell2)

    interTubeHoppings = cnt.CNT.interTubeHopping(cell1, cell2)

    # Lead hoppings

    offsetUnitCell1 = cnt.CNT(n1, m1, 1, axis=xAxis, origin=(cntUnitCell1.length, 0.0, 0.0), rot=rot1)
    offsetUnitCell2 = cnt.CNT(n2, m2, 1, axis=xAxis, origin=(cntUnitCell2.length, 0.0, 0.0), rot=rot2)

    intraCellLeadHoppings1 = cntUnitCell1.hoppings
    intraCellLeadHoppings2 = cntUnitCell2.hoppings

    interCellLeadHoppings1 = cnt.CNT.intraTubeHopping(cntUnitCell1, offsetUnitCell1)
    interCellLeadHoppings2 = cnt.CNT.intraTubeHopping(cntUnitCell2, offsetUnitCell2)

    # Kwant setup

    # Scattering region

    system = kwant.Builder()

    latticeVectorDevice1 = cell1.length * cell1.axis
    latticeVectorDevice2 = cell2.length * cell2.axis

    latticeVectorLead1 = cntUnitCell1.length * cntUnitCell1.axis
    latticeVectorLead2 = cntUnitCell2.length * cntUnitCell2.axis

    latticeDevice1 = kwant.lattice.Polyatomic([latticeVectorDevice1], cell1.sites, norbs=1)
    latticeDevice2 = kwant.lattice.Polyatomic([latticeVectorDevice2], cell2.sites, norbs=1)

    # Sites
    system[(latticeDevice1.sublattices[i](0) for i in range(len(cell1.sites)))] = 0.0
    system[(latticeDevice2.sublattices[i](0) for i in range(len(cell2.sites)))] = 0.0


    # Intra-tube hopping
    intraCellHoppingKind1 = [((0,), latticeDevice1.sublattices[hopping[0].astype('int')], latticeDevice1.sublattices[hopping[1].astype('int')]) for hopping in intraCellHoppings1]
    intraCellHoppingKind2 = [((0,), latticeDevice2.sublattices[hopping[0].astype('int')], latticeDevice2.sublattices[hopping[1].astype('int')]) for hopping in intraCellHoppings2]

    system[[kwant.builder.HoppingKind(*hopping) for hopping in intraCellHoppingKind1]] = const.INTRA_HOPPING
    system[[kwant.builder.HoppingKind(*hopping) for hopping in intraCellHoppingKind2]] = const.INTRA_HOPPING

    # Inter-tube hopping
    for hopping in interTubeHoppings:
      system[latticeDevice1.sublattices[hopping[0].astype('int')](0), latticeDevice2.sublattices[hopping[1].astype('int')](0)] = hopping[2]

    # Leads (including non-interacting part of scattering region)

    latticeLead1 = kwant.lattice.Polyatomic([latticeVectorLead1], leadUnitCell1.sites, norbs=1)
    symmetryLead1 = kwant.TranslationalSymmetry(-latticeVectorLead1)
    lead1 = kwant.Builder(symmetryLead1)

    latticeLead2 = kwant.lattice.Polyatomic([latticeVectorLead2], leadUnitCell2.sites, norbs=1)
    symmetryLead2 = kwant.TranslationalSymmetry(latticeVectorLead2)
    lead2 = kwant.Builder(symmetryLead2)

    # Lead sites
    system[(latticeLead1.sublattices[i](0) for i in range(len(leadUnitCell1.sites)))] = 0.0
    lead1[(latticeLead1.sublattices[i](0) for i in range(len(leadUnitCell1.sites)))] = 0.0

    system[(latticeLead2.sublattices[i](0) for i in range(len(leadUnitCell2.sites)))] = 0.0
    lead2[(latticeLead2.sublattices[i](0) for i in range(len(leadUnitCell2.sites)))] = 0.0

    # Intra-cell hoppings
    intraCellLeadHoppingKind1 = [((0,), latticeLead1.sublattices[hopping[0].astype('int')], latticeLead1.sublattices[hopping[1].astype('int')]) for hopping in intraCellLeadHoppings1]
    intraCellLeadHoppingKind2 = [((0,), latticeLead2.sublattices[hopping[0].astype('int')], latticeLead2.sublattices[hopping[1].astype('int')]) for hopping in intraCellLeadHoppings2]

    system[[kwant.builder.HoppingKind(*hopping) for hopping in intraCellLeadHoppingKind1]] = const.INTRA_HOPPING
    lead1[[kwant.builder.HoppingKind(*hopping) for hopping in intraCellLeadHoppingKind1]] = const.INTRA_HOPPING
    system[[kwant.builder.HoppingKind(*hopping) for hopping in intraCellLeadHoppingKind2]] = const.INTRA_HOPPING
    lead2[[kwant.builder.HoppingKind(*hopping) for hopping in intraCellLeadHoppingKind2]] = const.INTRA_HOPPING

    # Inter-cell hoppings (part of scattering region)
    interCellHoppingKind1 = [((0,), latticeDevice1.sublattices[hopping[0].astype('int')], latticeLead1.sublattices[hopping[1].astype('int')]) for hopping in interCellHoppings1]
    interCellHoppingKind2 = [((0,), latticeDevice2.sublattices[hopping[0].astype('int')], latticeLead2.sublattices[hopping[1].astype('int')]) for hopping in interCellHoppings2]

    system[[kwant.builder.HoppingKind(*hopping) for hopping in interCellHoppingKind1]] = const.INTRA_HOPPING
    system[[kwant.builder.HoppingKind(*hopping) for hopping in interCellHoppingKind2]] = const.INTRA_HOPPING

    # Inter-cell hoppings (part of proper leads)
    interCellLeadHoppingKind1 = [((-1,), latticeLead1.sublattices[hopping[0].astype('int')], latticeLead1.sublattices[hopping[1].astype('int')]) for hopping in interCellLeadHoppings1]
    interCellLeadHoppingKind2 = [((-1,), latticeLead2.sublattices[hopping[0].astype('int')], latticeLead2.sublattices[hopping[1].astype('int')]) for hopping in interCellLeadHoppings2]

    lead1[[kwant.builder.HoppingKind(*hopping) for hopping in interCellLeadHoppingKind1]] = const.INTRA_HOPPING
    lead2[[kwant.builder.HoppingKind(*hopping) for hopping in interCellLeadHoppingKind2]] = const.INTRA_HOPPING

    system.attach_lead(lead1)
    system.attach_lead(lead2)

    self.systemFinalized = system.finalized()

class angledContact(System):
  def __init__(self, n1, m1, n2, m2, angle, distance, rot1=0.0, offset1=0.0, rot2=0.0, offset2=0.0):

    self.n1 = n1
    self.m1 = m1
    self.n2 = n2
    self.m2 = m2
    self.distance = distance
    self.rot1 = rot1
    self.rot2 = rot2

    angle %= np.pi
    self.angle = angle

    xAxis = (1.0, 0.0, 0.0)
    axis1 = (1.0, 0.0, 0.0)
    axis2 = (np.cos(angle), np.sin(angle), 0.0)

    cntUnitCell1 = cnt.CNT(n1, m1, 1, axis=xAxis, rot=rot1)
    cntUnitCell2 = cnt.CNT(n2, m2, 1, axis=xAxis, rot=rot2)

    radius1 = cntUnitCell1.radius
    radius2 = cntUnitCell2.radius
    offset1 %= cntUnitCell1.length
    offset2 %= cntUnitCell2.length

    self.offset1 = offset1
    self.offset2 = offset2

    surfaceDistance = distance - radius1 - radius2
    cutoffDistance = np.sqrt((const.ALPHA - const.DELTA*np.log(const.COUPLING_CUTOFF))**2 - surfaceDistance**2)

    overlap1 = np.abs((radius1 + cutoffDistance)/np.tan(angle)) + radius2/np.sin(angle) + cutoffDistance
    overlap2 = np.abs((radius2 + cutoffDistance)/np.tan(angle)) + radius1/np.sin(angle) + cutoffDistance

    overlapCellNumber1 = 2 * np.ceil(overlap1/cntUnitCell1.length).astype('int') + 1
    overlapCellNumber2 = 2 * np.ceil(overlap2/cntUnitCell2.length).astype('int') + 1

    # Construct scattering geometry

    length1 = overlapCellNumber1 * cntUnitCell1.length
    origin1 = offset1 - 0.5*length1
    length2 = overlapCellNumber2 * cntUnitCell2.length
    origin2 = offset2 - 0.5*length2

    cell1 = cnt.CNT(n1, m1, overlapCellNumber1, axis=axis1, origin=(origin1, 0.0, 0.0), rot=rot1)
    cell2 = cnt.CNT(n2, m2, overlapCellNumber2, axis=axis2, origin=(origin2*np.cos(angle), origin2*np.sin(angle), -distance), rot=rot2)

    leadUnitCellLeft1 = cnt.CNT(n1, m1, 1, axis=axis1, origin=(origin1-cntUnitCell1.length, 0.0, 0.0), rot=rot1)
    leadUnitCellRight1 = cnt.CNT(n1, m1, 1, axis=axis1, origin=(origin1+cell1.length, 0.0, 0.0), rot=rot1)
    leadUnitCellLeft2 = cnt.CNT(n2, m2, 1, axis=axis2, origin=((origin2-cntUnitCell2.length)*np.cos(angle), (origin2-cntUnitCell2.length)*np.sin(angle), -distance), rot=rot2)
    leadUnitCellRight2 = cnt.CNT(n2, m2, 1, axis=axis2, origin=((origin2+cell2.length)*np.cos(angle), (origin2+cell2.length)*np.sin(angle), -distance), rot=rot2)

    # Scattering region hoppings

    intraCellHoppings1 = cell1.hoppings
    intraCellHoppings2 = cell2.hoppings

    interTubeHoppings = cnt.CNT.interTubeHopping(cell1, cell2)

    # Lead unit cells in scattering region

    intraCellHoppingsLeft1 = leadUnitCellLeft1.hoppings
    interCellHoppingsLeft1 = cnt.CNT.intraTubeHopping(cell1, leadUnitCellLeft1)

    intraCellHoppingsRight1 = leadUnitCellRight1.hoppings
    interCellHoppingsRight1 = cnt.CNT.intraTubeHopping(cell1, leadUnitCellRight1)

    intraCellHoppingsLeft2 = leadUnitCellLeft2.hoppings
    interCellHoppingsLeft2 = cnt.CNT.intraTubeHopping(cell2, leadUnitCellLeft2)

    intraCellHoppingsRight2 = leadUnitCellRight2.hoppings
    interCellHoppingsRight2 = cnt.CNT.intraTubeHopping(cell2, leadUnitCellRight2)

    # Lead hoppings

    offsetUnitCell1 = cnt.CNT(n1, m1, 1, axis=xAxis, origin=(cntUnitCell1.length, 0.0, 0.0), rot=rot1)
    offsetUnitCell2 = cnt.CNT(n2, m2, 1, axis=xAxis, origin=(cntUnitCell2.length, 0.0, 0.0), rot=rot2)

    intraCellLeadHoppings1 = cntUnitCell1.hoppings
    intraCellLeadHoppings2 = cntUnitCell2.hoppings

    interCellLeadHoppings1 = cnt.CNT.intraTubeHopping(cntUnitCell1, offsetUnitCell1)
    interCellLeadHoppings2 = cnt.CNT.intraTubeHopping(cntUnitCell2, offsetUnitCell2)

    # Kwant setup

    # Scattering region

    system = kwant.Builder()

    latticeVectorDevice1 = cell1.length * cell1.axis
    latticeVectorDevice2 = cell2.length * cell2.axis

    latticeVectorLead1 = cntUnitCell1.length * cell1.axis
    latticeVectorLead2 = cntUnitCell2.length * cell2.axis

    latticeDevice1 = kwant.lattice.Polyatomic([latticeVectorDevice1], cell1.sites, norbs=1)
    latticeDevice2 = kwant.lattice.Polyatomic([latticeVectorDevice2], cell2.sites, norbs=1)

    # Sites
    system[(latticeDevice1.sublattices[i](0) for i in range(len(cell1.sites)))] = 0.0
    system[(latticeDevice2.sublattices[i](0) for i in range(len(cell2.sites)))] = 0.0

    # Intra-tube hopping
    intraCellHoppingKind1 = [((0,), latticeDevice1.sublattices[hopping[0].astype('int')], latticeDevice1.sublattices[hopping[1].astype('int')]) for hopping in intraCellHoppings1]
    intraCellHoppingKind2 = [((0,), latticeDevice2.sublattices[hopping[0].astype('int')], latticeDevice2.sublattices[hopping[1].astype('int')]) for hopping in intraCellHoppings2]

    system[[kwant.builder.HoppingKind(*hopping) for hopping in intraCellHoppingKind1]] = const.INTRA_HOPPING
    system[[kwant.builder.HoppingKind(*hopping) for hopping in intraCellHoppingKind2]] = const.INTRA_HOPPING

    # Inter-tube hopping
    for hopping in interTubeHoppings:
      system[latticeDevice1.sublattices[hopping[0].astype('int')](0), latticeDevice2.sublattices[hopping[1].astype('int')](0)] = hopping[2]

    # Leads (including non-interacting part of scattering region)

    # Define lattices, symmetries and leads
    latticeLeadLeft1 = kwant.lattice.Polyatomic([latticeVectorLead1], leadUnitCellLeft1.sites, norbs=1)
    symmetryLeadLeft1 = kwant.TranslationalSymmetry(-latticeVectorLead1)
    leadLeft1 = kwant.Builder(symmetryLeadLeft1)

    latticeLeadRight1 = kwant.lattice.Polyatomic([latticeVectorLead1], leadUnitCellRight1.sites, norbs=1)
    symmetryLeadRight1 = kwant.TranslationalSymmetry(latticeVectorLead1)
    leadRight1 = kwant.Builder(symmetryLeadRight1)

    latticeLeadLeft2 = kwant.lattice.Polyatomic([latticeVectorLead2], leadUnitCellLeft2.sites, norbs=1)
    symmetryLeadLeft2 = kwant.TranslationalSymmetry(-latticeVectorLead2)
    leadLeft2 = kwant.Builder(symmetryLeadLeft2)

    latticeLeadRight2 = kwant.lattice.Polyatomic([latticeVectorLead2], leadUnitCellRight2.sites, norbs=1)
    symmetryLeadRight2 = kwant.TranslationalSymmetry(latticeVectorLead2)
    leadRight2 = kwant.Builder(symmetryLeadRight2)

    # Sites
    system[(latticeLeadLeft1.sublattices[i](0) for i in range(len(leadUnitCellLeft1.sites)))] = 0.0
    leadLeft1[(latticeLeadLeft1.sublattices[i](0) for i in range(len(leadUnitCellLeft1.sites)))] = 0.0

    system[(latticeLeadRight1.sublattices[i](0) for i in range(len(leadUnitCellRight1.sites)))] = 0.0
    leadRight1[(latticeLeadRight1.sublattices[i](0) for i in range(len(leadUnitCellRight1.sites)))] = 0.0

    system[(latticeLeadLeft2.sublattices[i](0) for i in range(len(leadUnitCellLeft2.sites)))] = 0.0
    leadLeft2[(latticeLeadLeft2.sublattices[i](0) for i in range(len(leadUnitCellLeft2.sites)))] = 0.0

    system[(latticeLeadRight2.sublattices[i](0) for i in range(len(leadUnitCellRight2.sites)))] = 0.0
    leadRight2[(latticeLeadRight2.sublattices[i](0) for i in range(len(leadUnitCellRight2.sites)))] = 0.0

    # Hoppings

    for hopping in intraCellHoppingsLeft1:
      system[latticeLeadLeft1.sublattices[hopping[0].astype('int')](0), latticeLeadLeft1.sublattices[hopping[1].astype('int')](0)] = hopping[2]
    for hopping in interCellHoppingsLeft1:
      system[latticeDevice1.sublattices[hopping[0].astype('int')](0), latticeLeadLeft1.sublattices[hopping[1].astype('int')](0)] = hopping[2]
    for hopping in intraCellLeadHoppings1:
      leadLeft1[latticeLeadLeft1.sublattices[hopping[0].astype('int')](0), latticeLeadLeft1.sublattices[hopping[1].astype('int')](0)] = hopping[2]
    for hopping in interCellLeadHoppings1:
      leadLeft1[latticeLeadLeft1.sublattices[hopping[0].astype('int')](0), latticeLeadLeft1.sublattices[hopping[1].astype('int')](1)] = hopping[2]

    for hopping in intraCellHoppingsRight1:
      system[latticeLeadRight1.sublattices[hopping[0].astype('int')](0), latticeLeadRight1.sublattices[hopping[1].astype('int')](0)] = hopping[2]
    for hopping in interCellHoppingsRight1:
      system[latticeDevice1.sublattices[hopping[0].astype('int')](0), latticeLeadRight1.sublattices[hopping[1].astype('int')](0)] = hopping[2]
    for hopping in intraCellLeadHoppings1:
      leadRight1[latticeLeadRight1.sublattices[hopping[0].astype('int')](0), latticeLeadRight1.sublattices[hopping[1].astype('int')](0)] = hopping[2]
    for hopping in interCellLeadHoppings1:
      leadRight1[latticeLeadRight1.sublattices[hopping[0].astype('int')](0), latticeLeadRight1.sublattices[hopping[1].astype('int')](1)] = hopping[2]

    for hopping in intraCellHoppingsLeft2:
      system[latticeLeadLeft2.sublattices[hopping[0].astype('int')](0), latticeLeadLeft2.sublattices[hopping[1].astype('int')](0)] = hopping[2]
    for hopping in interCellHoppingsLeft2:
      system[latticeDevice2.sublattices[hopping[0].astype('int')](0), latticeLeadLeft2.sublattices[hopping[1].astype('int')](0)] = hopping[2]
    for hopping in intraCellLeadHoppings2:
      leadLeft2[latticeLeadLeft2.sublattices[hopping[0].astype('int')](0), latticeLeadLeft2.sublattices[hopping[1].astype('int')](0)] = hopping[2]
    for hopping in interCellLeadHoppings2:
      leadLeft2[latticeLeadLeft2.sublattices[hopping[0].astype('int')](0), latticeLeadLeft2.sublattices[hopping[1].astype('int')](1)] = hopping[2]

    for hopping in intraCellHoppingsRight2:
      system[latticeLeadRight2.sublattices[hopping[0].astype('int')](0), latticeLeadRight2.sublattices[hopping[1].astype('int')](0)] = hopping[2]
    for hopping in interCellHoppingsRight2:
      system[latticeDevice2.sublattices[hopping[0].astype('int')](0), latticeLeadRight2.sublattices[hopping[1].astype('int')](0)] = hopping[2]
    for hopping in intraCellLeadHoppings2:
      leadRight2[latticeLeadRight2.sublattices[hopping[0].astype('int')](0), latticeLeadRight2.sublattices[hopping[1].astype('int')](0)] = hopping[2]
    for hopping in interCellLeadHoppings2:
      leadRight2[latticeLeadRight2.sublattices[hopping[0].astype('int')](0), latticeLeadRight2.sublattices[hopping[1].astype('int')](1)] = hopping[2]

    system.attach_lead(leadLeft1)
    system.attach_lead(leadRight1)
    system.attach_lead(leadLeft2)
    system.attach_lead(leadRight2)

    self.systemFinalized = system.finalized()

class Chain(System):
  def __init__(self, overlap, spacing=1.0, site=1.0, hopping=0.5):
    xVector = spacing * np.array([1.0, 0.0])
    yVector = spacing * np.array([0.0, 1.0])

    lattice = kwant.lattice.square(spacing)

    system = kwant.Builder()

    system[(lattice(i, j) for i in range(overlap) for j in range(-1,1))] = site
    system[lattice.neighbors()] = hopping

    leftLead1 = kwant.Builder(kwant.TranslationalSymmetry((-spacing, 0.0)))
    leftLead1[lattice(0, 0)] = site
    leftLead1[lattice.neighbors()] = hopping

    leftLead2 = kwant.Builder(kwant.TranslationalSymmetry((-spacing, 0.0)))
    leftLead2[lattice(0, -1)] = site
    leftLead2[lattice.neighbors()] = hopping

    system.attach_lead(leftLead1)
    system.attach_lead(leftLead1.reversed())
    system.attach_lead(leftLead2)
    system.attach_lead(leftLead2.reversed())

    self.systemFinalized = system.finalized()
