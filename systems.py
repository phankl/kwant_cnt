# Different CNT Systems for Quantum Electron Transport Problems with KWANT

import kwant
import matplotlib.pyplot as plt
import numpy as np

import constants as const
import cnt

class System:
  
  def transmission(self, i, j):
    
    # Transmission probability as function of energy
    
    energies = np.linspace(0.0, -3.0*const.INTRA_HOPPING, 1000)
    transmissionProbability = np.array([kwant.smatrix(self.systemFinalized, energy).transmission(i, j) for energy in energies])

    return energies, transmissionProbability
  
  def plotSystem(self):
    kwant.plot(self.systemFinalized, site_size=0.1, hop_lw=0.01)

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
        system[lattice.sublattices[hopping[0]](i), lattice.sublattices[hopping[1]](i)] = hopping[2]

    # Inter-cell hopping

    for i in range(cellNumber-1):
      for hopping in unitCellHoppings:
        system[lattice.sublattices[hopping[0]](i), lattice.sublattices[hopping[1]](i+1)] = hopping[2]



    # Left lead

    symmetryLead0 = kwant.TranslationalSymmetry(-latticeVector)
    lead0 = kwant.Builder(symmetryLead0)

    for i in range(len(deviceUnitCell.sites)):
      lead0[lattice.sublattices[i](-1)] = 0.0

    for hopping in deviceUnitCell.hoppings:
      lead0[lattice.sublattices[hopping[0]](0), lattice.sublattices[hopping[1]](0)] = hopping[2]

    for hopping in unitCellHoppings:
      lead0[lattice.sublattices[hopping[0]](0), lattice.sublattices[hopping[1]](1)] = hopping[2]

    system.attach_lead(lead0)

    # Right lead

    symmetryLead1 = kwant.TranslationalSymmetry(latticeVector)
    lead1 = kwant.Builder(symmetryLead1)

    for i in range(len(deviceUnitCell.sites)):
      lead1[lattice.sublattices[i](-1)] = 0.0

    for hopping in deviceUnitCell.hoppings:
      lead1[lattice.sublattices[hopping[0]](0), lattice.sublattices[hopping[1]](0)] = hopping[2]

    for hopping in unitCellHoppings:
      lead1[lattice.sublattices[hopping[0]](0), lattice.sublattices[hopping[1]](1)] = hopping[2]

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
      system[lattice1.sublattices[hopping[0]](0), lattice1.sublattices[hopping[1]](0)] = hopping[2]
    for hopping in intraCellHoppings2:
      system[lattice2.sublattices[hopping[0]](0), lattice2.sublattices[hopping[1]](0)] = hopping[2]

    kwant.plot(system, site_size=0.1, hop_lw=0.1)

    # Inter-tube hoppings

    for hopping in interTubeIntraCellHoppings:
      system[lattice1.sublattices[hopping[0]](0), lattice2.sublattices[hopping[1]](0)] = hopping[2]


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
      lead0[lattice1.sublattices[hopping[0]](0), lattice1.sublattices[hopping[1]](0)] = hopping[2]
    for hopping in intraCellHoppings2:
      lead0[lattice2.sublattices[hopping[0]](0), lattice2.sublattices[hopping[1]](0)] = hopping[2]
    
    # Intra-tube hoppings, between unit cells

    for hopping in interCellHoppings1:
      lead0[lattice1.sublattices[hopping[0]](0), lattice1.sublattices[hopping[1]](1)] = hopping[2]
    for hopping in interCellHoppings2:
      lead0[lattice2.sublattices[hopping[0]](0), lattice2.sublattices[hopping[1]](1)] = hopping[2]
 
    # Inter-tube hoppings, inside unit cell

    for hopping in interTubeIntraCellHoppings:
      lead0[lattice1.sublattices[hopping[0]](0), lattice2.sublattices[hopping[1]](0)] = hopping[2]

    # Inter-tube hoppings, between unit cells
    
    for hopping in interTubeInterCellHoppings12:
      lead0[lattice1.sublattices[hopping[0]](0), lattice2.sublattices[hopping[1]](1)] = hopping[2]
    for hopping in interTubeInterCellHoppings21:
      lead0[lattice2.sublattices[hopping[0]](0), lattice1.sublattices[hopping[1]](1)] = hopping[2]

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
      lead1[lattice1.sublattices[hopping[0]](0), lattice1.sublattices[hopping[1]](0)] = hopping[2]
    for hopping in intraCellHoppings2:
      lead1[lattice2.sublattices[hopping[0]](0), lattice2.sublattices[hopping[1]](0)] = hopping[2]
    
    # Intra-tube hoppings, between unit cells

    for hopping in interCellHoppings1:
      lead1[lattice1.sublattices[hopping[0]](0), lattice1.sublattices[hopping[1]](1)] = hopping[2]
    for hopping in interCellHoppings2:
      lead1[lattice2.sublattices[hopping[0]](0), lattice2.sublattices[hopping[1]](1)] = hopping[2]
 
    # Inter-tube hoppings, inside unit cell

    for hopping in interTubeIntraCellHoppings:
      lead1[lattice1.sublattices[hopping[0]](0), lattice2.sublattices[hopping[1]](0)] = hopping[2]

    # Inter-tube hoppings, between unit cells
    
    for hopping in interTubeInterCellHoppings12:
      lead1[lattice1.sublattices[hopping[0]](0), lattice2.sublattices[hopping[1]](1)] = hopping[2]
    for hopping in interTubeInterCellHoppings21:
      lead1[lattice2.sublattices[hopping[0]](0), lattice1.sublattices[hopping[1]](1)] = hopping[2]
    
    system.attach_lead(lead1)

    self.systemFinalized = system.finalized()
    self.leadFinalized = lead0.finalized()
  
  def plotBandStructure(self):
    kwant.plotter.bands(self.leadFinalized, momenta=1000, show=False)
    plt.xlabel("Momentum [m^(-1)]")
    plt.ylabel("Energy [eV]")
    plt.show()
