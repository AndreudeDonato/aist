"""Small synthetic datasets for the tests (no project data): metal-oxide clusters, gas molecules, references."""
import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

CM_TO_EV = 1.239841984e-4


def with_data(atoms, energy, *, J=None, freqs=None, charges=True, gap=None):
    """Attach an energy and the ``info`` fields aist reads (frequencies with the rigid modes first)."""
    atoms.calc = SinglePointCalculator(atoms, energy=energy)
    n_rigid = 0 if len(atoms) == 1 else (5 if len(atoms) == 2 else 6)
    if freqs is None and len(atoms) > 1:
        rng = np.random.default_rng(len(atoms) * 7 + int(abs(energy)) % 97)
        freqs = np.sort(rng.uniform(80.0, 900.0, 3 * len(atoms) - n_rigid))
    if freqs is not None:
        atoms.info["frequencies"] = [0.0] * n_rigid + [float(f) for f in freqs]
        atoms.info["zpe"] = 0.5 * float(np.sum(freqs)) * CM_TO_EV
    if J is not None:
        atoms.info["total_spin"] = {"N": J - 1, "S": (J - 1) / 2, "J": J}
    if charges:
        q = np.where(np.array(atoms.get_chemical_symbols()) == "O", -0.3, 0.4)
        atoms.info["hirshfeld"] = {"charges": list(q), "spins": [0.0] * len(atoms), "volumes": [12.0] * len(atoms)}
    if gap is not None:
        atoms.info["HOMO-LUMO"] = gap
    return atoms


def _geometry(n_metal, n_other, seed):
    """Compact, non-symmetric (chiral, C1) cluster geometry: metal core plus outer ligands."""
    rng = np.random.default_rng(seed)
    pos = [(0.0, 0.0, 0.0)]
    for k in range(1, n_metal):
        pos.append((2.4 * k, 0.3 * k * k, 0.1 * k))
    for k in range(n_other):
        theta = 2.2 * k + 0.4
        pos.append((1.2 * (n_metal - 1) + 2.0 * np.cos(theta), 2.0 * np.sin(theta), 1.4 + 0.5 * k))
    return np.array(pos) + rng.normal(scale=0.05, size=(len(pos), 3))


def metal_oxide_set(metal="Ir", ligand="O", e_metal=-100.0, e_ligand=-5.0, seed=0):
    """Two sizes (M2, M3) and several ligand counts, with formation energies that make oxidation compete."""
    rows = []
    for n_m, n_l, extra, J in ((2, 0, 0.0, 3.0), (2, 1, 0.2, 1.0), (2, 2, -0.1, 1.0), (2, 2, 0.15, 3.0),
                               (2, 3, 0.4, 2.0), (3, 2, 0.0, 2.0), (3, 4, 0.1, 1.0), (3, 4, 0.35, 1.0)):
        e = n_m * e_metal + n_l * (e_ligand - 1.1) + extra
        a = Atoms(metal * n_m + ligand * n_l, positions=_geometry(n_m, n_l, seed + 10 * n_m + n_l))
        rows.append(with_data(a, e, J=J, gap=1.0 + 0.2 * n_l))
    return rows


def metal_atom(metal="Ir", energy=-100.0):
    return with_data(Atoms(metal, positions=[(0.0, 0.0, 0.0)]), energy, J=4.0)


def o2(energy=-9.86, with_spin=False):
    a = Atoms("O2", positions=[(0, 0, -0.605), (0, 0, 0.605)])
    return with_data(a, energy, J=3.0 if with_spin else None, freqs=[1580.0], charges=False)


def h2(energy=-6.77):
    a = Atoms("H2", positions=[(0, 0, -0.37), (0, 0, 0.37)])
    return with_data(a, energy, freqs=[4401.0], charges=False)


def h2o(energy=-14.22):
    a = Atoms("H2O", positions=[(0.0, 0.757, -0.471), (0.0, -0.757, -0.471), (0.0, 0.0, 0.118)])
    return with_data(a, energy, freqs=[1595.0, 3657.0, 3756.0], charges=False)


def hydroxide_set(metal="Ir"):
    """M2 clusters with O and H (oxides, hydroxides, adsorbed water) for the two-reservoir tests."""
    rows = []
    for formula, e in (("Ir2O2", -219.0), ("Ir2O2H2", -226.5), ("Ir2O3H2", -232.6), ("Ir2O4", -230.1),
                       ("Ir2O4H2", -236.9), ("Ir2O3", -224.8)):
        symbols = Atoms(formula.replace("Ir", metal)).get_chemical_symbols()
        n_m = symbols.count(metal)
        a = Atoms(symbols, positions=_geometry(n_m, len(symbols) - n_m, len(symbols)))
        rows.append(with_data(a, e, J=1.0))
    return rows
