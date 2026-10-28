"""Grand-canonical ensemble of small Cu-O clusters in O2 (one chemical potential), on ASE objects.

Random Cu3Ox and Cu4Ox structures are relaxed and vibrated with ASE's EMT potential, then weighted on a
(T, p_O2) grid. EMT is a toy potential, used here only because it ships with ASE and runs in seconds: for
science, attach DFT energies, frequencies and spin multiplicities instead (``atoms.info``).

    python examples/cu_oxide_emt.py      # about 20 s
"""
import os
import tempfile

import numpy as np
from ase import Atoms
from ase.calculators.emt import EMT
from ase.optimize import BFGS
from ase.units import invcm
from ase.vibrations import Vibrations

from aist import GrandCanonicalEnsemble, Reference, Reservoir, ThermoTerms


def relax(atoms):
    atoms.calc = EMT()
    BFGS(atoms, logfile=None).run(fmax=0.01, steps=300)
    return atoms


def attach_vibrations(atoms, workdir, multiplicity=1):
    """Harmonic frequencies (cm⁻¹, all 3N, imaginary ones negative), ZPE and spin in ``atoms.info``."""
    vib = Vibrations(atoms, name=os.path.join(workdir, f"vib{id(atoms)}"), delta=0.01)
    vib.run()
    energies = vib.get_energies()
    nu = np.sort(np.where(np.abs(energies.imag) > 0, -np.abs(energies.imag), energies.real) / invcm)
    atoms.info["frequencies"] = list(nu)
    atoms.info["zpe"] = 0.5 * float(np.sum(energies.real[nu > 0]))
    atoms.info["total_spin"] = {"J": multiplicity}
    return atoms


rng = np.random.default_rng(1)
with tempfile.TemporaryDirectory() as workdir:
    clusters = []
    for n_cu in (3, 4):
        for n_o in range(2 * n_cu + 1):
            for _ in range(3):  # three random starts per composition
                atoms = relax(Atoms(f"Cu{n_cu}O{n_o}", positions=rng.normal(scale=1.3, size=(n_cu + n_o, 3))))
                d = atoms.get_all_distances()
                np.fill_diagonal(d, np.inf)
                if d.min() > 0.9:
                    clusters.append(attach_vibrations(atoms, workdir))
    o2 = attach_vibrations(relax(Atoms("O2", positions=[(0, 0, 0), (0, 0, 1.2)])), workdir, multiplicity=3)
cu = Atoms("Cu", calculator=EMT())

# The terms are your choice: by default only the electronic contribution (energy and spin degeneracy) enters;
# here the clusters also get their harmonic vibrations and ZPE. O2 gets its full ideal-gas chemical potential.
ensemble = GrandCanonicalEnsemble(clusters, references=[Reference(cu)],
                                  reservoirs=[Reservoir(o2, multiplicity=3)],
                                  terms=ThermoTerms(vibrations=True, zpe=True))
T = np.linspace(200.0, 1000.0, 5)
p = np.logspace(-10, 5, 4)
result = ensemble.evaluate(T=T, p={"O2": p})

print(f"{len(clusters)} clusters; <n_O/n_Cu> on the grid (rows p_O2 in Pa, columns T in K):")
print("p \\ T     " + "".join(f"{t:8.0f}" for t in T))
for pi, row in zip(p, result.composition("O")):
    print(f"{pi:9.1e} " + "".join(f"{x:8.2f}" for x in row))
i, j = result.index(O2=1e5, T=200.0)
best = result.most_stable()[i, j]
print(f"most stable at 1e5 Pa, 200 K: {result.formulas[best]} (population {result.populations[best, i, j]:.2f})")
