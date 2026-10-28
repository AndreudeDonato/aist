"""Chemical potential of a gas-phase reservoir molecule on a (T, p) grid."""
from __future__ import annotations

import numpy as np
from ase.io import read

from aist_v01.composedthermo import compute_thermo_contributions
from aist_v01.utils import get_energy

O2_ELECTRONIC_STATES = ((0.0, 3),)   # triplet ground state


def reference_molecule_energy(path: str) -> float:
    """Bare DFT energy (eV) of the reservoir molecule in ``path``."""
    return float(get_energy(read(path)))


def reservoir_chemical_potential_grid(
    path: str,
    Ts: np.ndarray,
    Ps: np.ndarray,
    *,
    novib: bool = True,
    relative_to_molecule: bool = False,
    elec_states=O2_ELECTRONIC_STATES,
) -> np.ndarray:
    """``μ(T, p)`` of the molecule in ``path`` on the grid; shape ``(n_P, n_T)`` in eV.

    Translation, rotation and the electronic term always enter; vibrations and ZPE unless ``novib``.
    With ``relative_to_molecule=True`` the bare DFT energy is subtracted.
    """
    structure = read(path)
    Tg, Pg = np.meshgrid(np.asarray(Ts, dtype=float), np.asarray(Ps, dtype=float), indexing="xy")
    mu, _ = compute_thermo_contributions(
        structure, Tg, Pg, elec_states=list(elec_states),
        include_v=not novib, include_zpe=not novib, include_r=True, include_t=True,
    )
    mu = np.asarray(mu, dtype=np.float64)
    if relative_to_molecule:
        mu = mu - reference_molecule_energy(path)
    return mu
