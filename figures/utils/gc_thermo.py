"""O₂ chemical potential on the grand-canonical (T, p) grid."""
from __future__ import annotations

import numpy as np

from aist.composedthermo import compute_thermo_contributions
from aist.utils import get_energy


def o2_molecule_energy(o2_path: str) -> float:
    """Bare DFT energy of the O₂ reference (eV)."""
    from ase.io import read

    return float(get_energy(read(o2_path)))


def o2_chemical_potential_grid(
    o2_path: str,
    Ts: np.ndarray,
    Ps: np.ndarray,
    *,
    novib: bool = True,
    relative_to_molecule: bool = False,
) -> np.ndarray:
    """``μ_O₂(T,p)`` on the GC grid; shape ``(n_P, n_T)`` in eV.

    With ``relative_to_molecule=True``, returns ``μ_O₂ − E_O₂`` using the
  DFT energy from the same O₂ reference file (PBE or PBE0).
    """
    from ase.io import read

    structure = read(o2_path)
    Tg, Pg = np.meshgrid(np.asarray(Ts, dtype=float), np.asarray(Ps, dtype=float), indexing="xy")
    include_v = not novib
    include_zpe = not novib
    mu, _ = compute_thermo_contributions(
        structure,
        Tg,
        Pg,
        elec_states=[(0.0, 3)],
        include_v=include_v,
        include_zpe=include_zpe,
        include_r=True,
        include_t=True,
    )
    mu = np.asarray(mu, dtype=np.float64)
    if relative_to_molecule:
        mu = mu - o2_molecule_energy(o2_path)
    return mu
