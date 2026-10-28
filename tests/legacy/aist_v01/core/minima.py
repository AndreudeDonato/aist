"""Stoichiometry minima selection and inertia orientation (from analysis/plotminima.py)."""
from __future__ import annotations

import re
from typing import Dict, Iterable, Tuple

import numpy as np
from ase import Atoms

from aist_v01.thermo import ThermoPartitionFunctions
from aist_v01.utils import get_energy


def parse_group_formula(group_formula: str) -> Dict[str, int]:
    tokens = re.findall(r"([A-Z][a-z]*)(\d*)", group_formula.strip())
    if not tokens:
        raise ValueError(f"Invalid atom-group formula '{group_formula}'")
    coeffs: Dict[str, int] = {}
    for element, count_str in tokens:
        coeffs[element] = coeffs.get(element, 0) + (int(count_str) if count_str else 1)
    return coeffs


def weighted_group_count(symbols: Iterable[str], group_coeffs: Dict[str, int]) -> int:
    counts: Dict[str, int] = {}
    for symbol in symbols:
        counts[symbol] = counts.get(symbol, 0) + 1
    return sum(group_coeffs[k] * counts.get(k, 0) for k in group_coeffs)


def orient_by_inertia_axes(atoms: Atoms, tp: ThermoPartitionFunctions | None = None) -> Atoms:
    tp = tp or ThermoPartitionFunctions()
    atoms_oriented = atoms.copy()
    atoms_oriented.set_positions(
        atoms_oriented.get_positions() - atoms_oriented.get_center_of_mass()
    )
    inertia = tp.get_inertia_tensor(atoms_oriented)
    _, eigvecs = np.linalg.eigh(inertia)
    basis = np.column_stack((eigvecs[:, 0], eigvecs[:, 1], eigvecs[:, 2]))
    atoms_oriented.set_positions(atoms_oriented.get_positions() @ basis)
    return atoms_oriented


def select_stoichiometry_minima(
    structures: list[Atoms],
    *,
    x_group_formula: str = "Ir",
    y_group_formula: str = "O",
    orient: bool = True,
) -> list[Tuple[int, int, Atoms]]:
    """Return (n_x, n_y, oriented_atoms) for lowest-energy isomer per stoichiometry."""
    return [
        (x, y, a)
        for x, y, a, _ in select_stoichiometry_minima_indexed(
            structures, x_group_formula=x_group_formula, y_group_formula=y_group_formula, orient=orient,
        )
    ]


def select_stoichiometry_minima_indexed(
    structures: list[Atoms],
    *,
    x_group_formula: str = "Ir",
    y_group_formula: str = "O",
    orient: bool = True,
) -> list[Tuple[int, int, Atoms, int]]:
    """Like :func:`select_stoichiometry_minima` but also returns the traj frame index."""
    x_group = parse_group_formula(x_group_formula)
    y_group = parse_group_formula(y_group_formula)
    tp = ThermoPartitionFunctions()

    minima: dict[Tuple[int, int], Tuple[float, int]] = {}
    for idx, atoms in enumerate(structures):
        syms = atoms.get_chemical_symbols()
        key = (
            weighted_group_count(syms, x_group),
            weighted_group_count(syms, y_group),
        )
        e = float(get_energy(atoms))
        if key not in minima or e < minima[key][0]:
            minima[key] = (e, idx)

    selected = []
    for (x_val, y_val), (_e, idx) in sorted(minima.items()):
        a = structures[idx]
        if orient:
            a = orient_by_inertia_axes(a, tp)
        selected.append((x_val, y_val, a, idx))
    return selected
