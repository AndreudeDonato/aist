"""Free-energy contributions of one molecule: the core of aist.

Attach your own results to an ASE Atoms object (here, a water molecule with its experimental vibrational
frequencies and an illustrative energy), then ask for the terms you want on the temperatures you want.

    python examples/water_contributions.py
"""
import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from aist import ThermoTerms, contributions, free_energy

water = Atoms("H2O", positions=[(0.0, 0.757, -0.471), (0.0, -0.757, -0.471), (0.0, 0.0, 0.118)])
water.calc = SinglePointCalculator(water, energy=-14.22)   # your DFT energy (eV)
water.info["frequencies"] = [0.0] * 6 + [1595.0, 3657.0, 3756.0]   # all 3N modes (cm⁻¹), rigid ones first

T = np.array([300.0, 600.0, 1000.0])
p = 1e5  # Pa

# Each term separately (eV): the ideal-gas treatment of a molecule switches every term on.
parts = contributions(water, T, p, ThermoTerms.gas(), multiplicity=1)
print("T (K):        " + "".join(f"{t:10.0f}" for t in T))
for name, values in parts.items():
    print(f"{name:13s} " + "".join(f"{v:10.4f}" for v in np.broadcast_to(values, T.shape)))

# The free energy with the terms you choose: here, a water molecule held in place (no translation, no rotation).
bound = free_energy(water, T, terms=ThermoTerms(vibrations=True, zpe=True), multiplicity=1)
print("G, gas      " + "".join(f"{v:10.4f}" for v in free_energy(water, T, p, ThermoTerms.gas(), multiplicity=1)))
print("G, held     " + "".join(f"{v:10.4f}" for v in bound))
