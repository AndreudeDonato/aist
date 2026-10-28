"""A canonical ensemble of isomers, built by hand from the per-structure free energies.

Isomers of Cu4Ag2 (random starts relaxed with ASE's EMT potential, a toy potential that ships with ASE) are given
their harmonic vibrations; their populations versus temperature follow from aist's free energies and a
Boltzmann weighting with the chirality degeneracy. The ensemble is yours to define: which isomers, which terms.

    python examples/isomers_canonical.py      # a few seconds
"""
import os
import tempfile

import numpy as np
from ase import Atoms
from ase.calculators.emt import EMT
from ase.optimize import BFGS
from ase.units import invcm
from ase.vibrations import Vibrations

from aist import ThermoTerms, boltzmann_populations, free_energy
from aist.ensembles import chirality_degeneracy

rng = np.random.default_rng(3)
isomers = []
with tempfile.TemporaryDirectory() as workdir:
    for k in range(16):
        atoms = Atoms("Cu4Ag2", positions=rng.normal(scale=1.5, size=(6, 3)), calculator=EMT())
        BFGS(atoms, logfile=None).run(fmax=0.005, steps=500)
        if any(abs(atoms.get_potential_energy() - a.get_potential_energy()) < 1e-3 for a in isomers):
            continue  # same minimum again
        vib = Vibrations(atoms, name=os.path.join(workdir, f"vib{k}"), delta=0.01)
        vib.run()
        e = vib.get_energies()
        atoms.info["frequencies"] = list(np.sort(np.where(np.abs(e.imag) > 0, -np.abs(e.imag), e.real) / invcm))
        atoms.info["total_spin"] = {"J": 1}
        isomers.append(atoms)

T = np.linspace(100.0, 1200.0, 6)
terms = ThermoTerms(vibrations=True, zpe=True)          # your choice of terms
G = np.array([free_energy(a, T, terms=terms) for a in isomers])
P = boltzmann_populations(G, T, [chirality_degeneracy(a) for a in isomers])

print(f"{len(isomers)} Cu4Ag2 isomers; populations (%) vs T (K):")
print("ΔE (eV)  " + "".join(f"{t:8.0f}" for t in T))
E0 = min(a.get_potential_energy() for a in isomers)
for a, row in sorted(zip(isomers, P), key=lambda x: x[0].get_potential_energy()):
    print(f"{a.get_potential_energy() - E0:7.3f}  " + "".join(f"{100 * x:8.1f}" for x in row))
