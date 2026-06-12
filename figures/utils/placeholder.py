"""Prepare placeholder trajectories in cache (synthetic energies, etc.)."""
from __future__ import annotations

import os

from ase.io import read, write


def stamp_placeholder_energies(
    source_traj: str,
    dest_traj: str,
    *,
    e_per_atom: float = -400.0,
) -> str:
    """Copy ``source_traj`` to ``dest_traj``, adding ``info['energy']`` if missing."""
    os.makedirs(os.path.dirname(dest_traj) or ".", exist_ok=True)
    structures = list(read(source_traj, ":"))
    for i, atoms in enumerate(structures):
        if atoms.info.get("energy") is None:
            n = len(atoms)
            atoms.info = dict(atoms.info)
            atoms.info["energy"] = float(e_per_atom * n + 0.01 * i)
    write(dest_traj, structures)
    print(f"[placeholder] staged {len(structures)} frames -> {dest_traj}")
    return dest_traj
