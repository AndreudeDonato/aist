"""Exact stoichiometric balance of structures against references and reservoir molecules.

Structure *i* with element counts ``n_i`` is written as ``n_i = Σ_s x_is c_s`` over the composition vectors
``c_s`` of the references and reservoirs. The coefficients are solved in rational arithmetic, so the floats that
enter the free energies are correctly rounded (and equal to those of ``aist.stoichiometry.balance_reaction``).
"""
from __future__ import annotations

from collections import Counter
from fractions import Fraction

import numpy as np


class BalanceError(ValueError):
    """The references and reservoirs cannot balance a structure, or do not define a unique balance."""


def element_counts(structures) -> tuple[list[str], np.ndarray]:
    """Sorted element list and the integer count matrix (structures × elements)."""
    counters = [Counter(atoms.get_chemical_symbols()) for atoms in structures]
    elements = sorted({e for c in counters for e in c})
    counts = np.array([[c.get(e, 0) for e in elements] for c in counters], dtype=int).reshape(len(counters), -1)
    return elements, counts


def _row_reduce(rows: list[list[Fraction]]):
    """Reduced row echelon form; returns (rows, pivot columns)."""
    rows = [list(r) for r in rows]
    n_rows, n_cols = len(rows), (len(rows[0]) if rows else 0)
    pivots, r = [], 0
    for col in range(n_cols):
        pivot = next((i for i in range(r, n_rows) if rows[i][col] != 0), None)
        if pivot is None:
            continue
        rows[r], rows[pivot] = rows[pivot], rows[r]
        lead = rows[r][col]
        rows[r] = [v / lead for v in rows[r]]
        for i in range(n_rows):
            if i != r and rows[i][col] != 0:
                f = rows[i][col]
                rows[i] = [a - f * b for a, b in zip(rows[i], rows[r])]
        pivots.append(col)
        r += 1
        if r == n_rows:
            break
    return rows, pivots


def _dependency(matrix: list[list[Fraction]], n_species: int) -> list[Fraction]:
    """One non-zero vector v with matrix · v = 0 (matrix is elements × species)."""
    reduced, pivots = _row_reduce(matrix)
    free = next(c for c in range(n_species) if c not in pivots)
    v = [Fraction(0)] * n_species
    v[free] = Fraction(1)
    for row, col in zip(reduced, pivots):
        v[col] = -row[free]
    return v


def _format_dependency(names, v) -> str:
    left = [f"{abs(c)}·{n}" for n, c in zip(names, v) if c > 0]
    right = [f"{abs(c)}·{n}" for n, c in zip(names, v) if c < 0]
    return f"{' + '.join(left)} = {' + '.join(right)}"


def balance_coefficients(structures, species: list[tuple[str, dict]]) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Exact coefficients ``x_is`` of every structure on ``species`` = ``[(name, composition), ...]``.

    Returns ``(elements, counts, X)`` with ``X`` an object array of ``Fraction`` (structures × species).
    Raises ``BalanceError`` when the species are linearly dependent (e.g. O2, H2 and H2O together) or when a
    structure holds an element that the species cannot supply.
    """
    names = [name for name, _ in species]
    elements, counts = element_counts(structures)
    supplied = {e for _, comp in species for e in comp}
    missing = [e for e in elements if e not in supplied]
    if missing:
        i = int(np.nonzero(counts[:, elements.index(missing[0])])[0][0])
        raise BalanceError(
            f"{', '.join(missing)} appear(s) in the structures (e.g. row {i}, "
            f"{structures[i].get_chemical_formula()}) but in no reference or reservoir: add a species that supplies it"
        )
    all_elements = sorted(set(elements) | supplied)
    matrix = [[Fraction(comp.get(e, 0)) for _, comp in species] for e in all_elements]
    _, pivots = _row_reduce(matrix)
    if len(pivots) < len(species):
        v = _dependency(matrix, len(species))
        raise BalanceError(
            f"The references and reservoirs {names} are linearly dependent ({_format_dependency(names, v)}), so the "
            "balance is not unique; drop one of them (at equilibrium its chemical potential follows from the others)"
        )
    # an invertible square block: one independent element row per species
    basis_rows: list[int] = []
    for k in range(len(all_elements)):
        trial = [matrix[j] for j in basis_rows + [k]]
        if len(_row_reduce(trial)[1]) == len(basis_rows) + 1:
            basis_rows.append(k)
        if len(basis_rows) == len(species):
            break
    block = [matrix[k] for k in basis_rows]
    n_sp = len(species)
    augmented = [row + [Fraction(int(i == j)) for j in range(n_sp)] for i, row in enumerate(block)]
    inverse = [row[n_sp:] for row in _row_reduce(augmented)[0]]

    col = {e: j for j, e in enumerate(elements)}
    X = np.empty((len(structures), n_sp), dtype=object)
    for i in range(len(structures)):
        target = [Fraction(int(counts[i, col[e]])) if e in col else Fraction(0) for e in all_elements]
        rhs = [target[k] for k in basis_rows]
        x = [sum(inverse[a][b] * rhs[b] for b in range(n_sp)) for a in range(n_sp)]
        for k, e in enumerate(all_elements):
            if sum(matrix[k][s] * x[s] for s in range(n_sp)) != target[k]:
                raise BalanceError(
                    f"Row {i} ({structures[i].get_chemical_formula()}) cannot be balanced on {names}: element {e} "
                    "does not close"
                )
        X[i] = x
    return elements, counts, X
