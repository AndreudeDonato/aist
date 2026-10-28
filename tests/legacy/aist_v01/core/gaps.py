"""Per-structure electronic gaps and their ensemble projections (HOMO–LUMO from ``atoms.info``)."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d

from aist_v01.ace.join import ensemble_weighted_mean_map

HOMO_LUMO_KEY = "HOMO-LUMO"
POP_OX_N_BINS = 100
POP_OX_SMOOTH_SIGMA = 1.0


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
    if gaps.size != Pgi.shape[0]:
        raise ValueError(f"{gaps.size} gaps but {Pgi.shape[0]} GC rows")
    return ensemble_weighted_mean_map(Pgi, gaps)


def oxidation_ratio_xgrid(ratios: np.ndarray, *, n_bins: int = POP_OX_N_BINS) -> np.ndarray:
    """Regular ``n_O/n_Ir`` grid spanning the finite ratios."""
    ratio = np.asarray(ratios, dtype=float)
    finite = ratio[np.isfinite(ratio)]
    if finite.size == 0:
        return np.array([0.0])
    x_min, x_max = float(finite.min()), float(finite.max())
    if x_max <= x_min:
        return np.array([x_min])
    return np.linspace(x_min, x_max, int(n_bins))


def structure_oxidation_ratios_and_gaps(structures):
    """``(n_O/n_Ir, gap, n_Ir)`` arrays over the structures that carry a gap."""
    ratios, gaps, n_ir_vals = [], [], []
    for atoms in structures:
        syms = atoms.get_chemical_symbols()
        n_ir = sum(s == "Ir" for s in syms)
        n_o = sum(s == "O" for s in syms)
        gap = atoms.info.get(HOMO_LUMO_KEY)
        if gap is None or n_ir == 0:
            continue
        ratios.append(n_o / n_ir)
        gaps.append(float(gap))
        n_ir_vals.append(n_ir)
    return np.asarray(ratios, dtype=float), np.asarray(gaps, dtype=float), np.asarray(n_ir_vals, dtype=int)


def structure_oxidation_ratios(structures) -> np.ndarray:
    """Per-structure ``n_O / n_Ir`` for structures with a gap."""
    return structure_oxidation_ratios_and_gaps(structures)[0]


def kernel_smoothed_mean_vs_oxidation(
    ratios, values, x_grid, *, bandwidth: float | None = None, smooth_sigma: float = POP_OX_SMOOTH_SIGMA,
) -> np.ndarray:
    """Gaussian-kernel mean of ``values`` on ``x_grid``, then a 1D Gaussian smoothing of the curve."""
    ratio = np.asarray(ratios, dtype=float)
    values = np.asarray(values, dtype=float)
    x_grid = np.asarray(x_grid, dtype=float)
    mask = np.isfinite(ratio) & np.isfinite(values)
    ratio, values = ratio[mask], values[mask]
    if ratio.size == 0 or x_grid.size == 0:
        return np.full(x_grid.shape, np.nan, dtype=float)
    x_min, x_max = float(x_grid[0]), float(x_grid[-1])
    span = max(x_max - x_min, 1e-9)
    bw = float(bandwidth) if bandwidth is not None else span / max(len(x_grid) * 0.55, 4.0)
    y = np.full(len(x_grid), np.nan, dtype=float)
    for gi, xi in enumerate(x_grid):
        w = np.exp(-0.5 * ((ratio - xi) / bw) ** 2)
        wsum = w.sum()
        if wsum > 0:
            y[gi] = float(np.dot(w, values) / wsum)
    if smooth_sigma > 0 and y.size > 3:
        valid = np.isfinite(y).astype(float)
        num = gaussian_filter1d(np.nan_to_num(y, nan=0.0), sigma=float(smooth_sigma), mode="nearest")
        den = gaussian_filter1d(valid, sigma=float(smooth_sigma), mode="nearest")
        y = np.divide(num, den, out=np.full_like(y, np.nan), where=den > 0)
    return y


def homo_lumo_gap_lines_by_ir_size(structures, ir_counts, x_grid, *, smooth_sigma: float = POP_OX_SMOOTH_SIGMA):
    """Per-``n_Ir`` kernel-smoothed mean gap vs ``x_grid``."""
    ratios, gaps, n_ir_vals = structure_oxidation_ratios_and_gaps(structures)
    lines: dict[int, np.ndarray] = {}
    for n_ir in ir_counts:
        mask = n_ir_vals == int(n_ir)
        if not np.any(mask):
            continue
        lines[int(n_ir)] = kernel_smoothed_mean_vs_oxidation(ratios[mask], gaps[mask], x_grid, smooth_sigma=smooth_sigma)
    return lines
