"""HOMO–LUMO gap helpers for GC ensemble maps."""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from figures.utils.join import ensemble_weighted_mean_map

HOMO_LUMO_KEY = "HOMO-LUMO"
HOMO_LUMO_CMAP = "Greens"


def homo_lumo_gaps(structures) -> np.ndarray:
    """Per-structure gap (eV) from ``atoms.info['HOMO-LUMO']``; NaN if missing."""
    gaps = np.full(len(structures), np.nan, dtype=float)
    for i, atoms in enumerate(structures):
        val = atoms.info.get(HOMO_LUMO_KEY)
        if val is not None:
            gaps[i] = float(val)
    return gaps


def homo_lumo_gap_map(Pgi: np.ndarray, gaps: np.ndarray) -> np.ndarray:
    """Ensemble ``⟨HOMO–LUMO⟩`` field on the GC grid."""
    n = min(gaps.size, Pgi.shape[0])
    return ensemble_weighted_mean_map(Pgi[:n], gaps[:n])


def homo_lumo_gap_minmax_vs_oxidation(structures) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Min/max HOMO–LUMO gap at each stoichiometry (grouped by n_Ir, n_O)."""
    by_stoich: dict[tuple[int, int], list[float]] = defaultdict(list)
    for atoms in structures:
        syms = atoms.get_chemical_symbols()
        n_ir = sum(s == "Ir" for s in syms)
        n_o = sum(s == "O" for s in syms)
        gap = atoms.info.get(HOMO_LUMO_KEY)
        if gap is None or n_ir == 0:
            continue
        by_stoich[(n_ir, n_o)].append(float(gap))

    rows: list[tuple[float, float, float]] = []
    for (n_ir, n_o), gs in by_stoich.items():
        x = n_o / n_ir
        rows.append((x, min(gs), max(gs)))
    rows.sort(key=lambda r: r[0])
    xs = np.asarray([r[0] for r in rows], dtype=float)
    ymin = np.asarray([r[1] for r in rows], dtype=float)
    ymax = np.asarray([r[2] for r in rows], dtype=float)
    return xs, ymin, ymax
