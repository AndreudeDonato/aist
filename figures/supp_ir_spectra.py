"""Supplementary – Population-weighted IR spectra by Ir size at the four standard conditions, PBE | PBE0.

Rows: THERMO_IR_COUNTS; columns: PBE | PBE0. In each panel one spectrum per (T, pO₂) of ``STANDARD_CONDITIONS``
(colours; nearest grid point), S(ν) = Σ_i P_i Σ_k I_ik g(ν − ν_ik): grand-canonical populations of that Ir size
with vibrations (full), harmonic frequencies (raw where stored) and FHI-aims IR intensities of every structure,
the five/six rigid modes left out, Gaussian broadening of ``FWHM_CM`` cm⁻¹. Each spectrum is normalised to its
maximum and offset, so the panels compare band positions and shapes, not absolute intensities.

Output: ``figures/outputs/supp_ir_spectra.svg``
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from aist.core.grandcanonical import pressure_grid_index, temperature_grid_index
from aist.thermo import ThermoPartitionFunctions

from figures.config import DPI, OUTPUT_DIR, STANDARD_CONDITIONS, THERMO_IR_COUNTS
from figures.utils.cache import ensure_for_supplementary
from figures.utils.datasets import DATASETS, dataset_inputs, gc_structures
from figures.utils.io import ensure_dirs, load_gc_npz
from figures.utils.plotting import (
    AXIS_LABEL_FONTSIZE,
    LEGEND_FONTSIZE,
    TICK_FONTSIZE,
    TITLE_FONTSIZE,
    save_fig,
)

OUTPUT_PATH = os.path.join(OUTPUT_DIR, "supp_ir_spectra.svg")
FIG_W_MM = 180
ROW_H_MM = 52
NU_GRID = np.arange(100.0, 1300.0, 1.0)   # cm⁻¹
FWHM_CM = 20.0
OFFSET = 1.15


def _modes(atoms, tp: ThermoPartitionFunctions) -> tuple[np.ndarray, np.ndarray]:
    """Vibrational frequencies (cm⁻¹) and IR intensities (D²/Å²) of one structure, rigid modes left out."""
    nu = np.asarray(atoms.info.get("frequencies_raw", atoms.info["frequencies"]), dtype=float)
    ir = np.asarray(atoms.info["ir_intensities"], dtype=float)
    if nu.size != ir.size:
        raise ValueError(f"{atoms.get_chemical_formula()}: {nu.size} frequencies but {ir.size} IR intensities")
    n_rigid = 5 if tp.is_linear(atoms) else 6
    nu, ir = nu[n_rigid:], np.nan_to_num(ir[n_rigid:], nan=0.0)
    keep = nu > 0
    return nu[keep], ir[keep]


def _structure_spectra(structures) -> np.ndarray:
    """Broadened IR spectrum of every structure on ``NU_GRID`` (rows = structures)."""
    tp = ThermoPartitionFunctions()
    sigma = FWHM_CM / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    out = np.zeros((len(structures), NU_GRID.size))
    for i, atoms in enumerate(structures):
        nu, ir = _modes(atoms, tp)
        g = np.exp(-0.5 * ((NU_GRID[None, :] - nu[:, None]) / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))
        out[i] = ir @ g
    return out


def _dataset_spectra(label: str) -> dict:
    """{(n_Ir, condition index): ensemble spectrum} of one functional."""
    inp = dataset_inputs(label)
    gc = load_gc_npz(inp.gc_npz, light=True)
    structures = gc_structures(label)
    if len(structures) != gc["Pgi"].shape[0]:
        raise ValueError(f"{label}: {len(structures)} structures but {gc['Pgi'].shape[0]} GC rows")
    spectra = _structure_spectra(structures)
    n_ir = np.array([a.get_chemical_symbols().count("Ir") for a in structures])
    out = {}
    for c, cond in enumerate(STANDARD_CONDITIONS):
        p_idx, _ = pressure_grid_index(gc["Ps"], cond["P"])
        t_idx, _ = temperature_grid_index(gc["Ts"], cond["T"])
        for n in THERMO_IR_COUNTS:
            w = np.asarray(gc["Pgi"][n_ir == n, p_idx, t_idx], dtype=float)
            out[(n, c)] = (w / w.sum()) @ spectra[n_ir == n]
    return out


def main() -> None:
    ensure_for_supplementary(["supp_ir_spectra"])
    ensure_dirs(OUTPUT_DIR)
    data = {label: _dataset_spectra(label) for label in DATASETS}

    n_rows = len(THERMO_IR_COUNTS)
    fig, axes = plt.subplots(n_rows, len(DATASETS), figsize=(FIG_W_MM / 25.4, n_rows * ROW_H_MM / 25.4),
                             dpi=DPI, sharex=True, sharey=True, squeeze=False)
    for row, n in enumerate(THERMO_IR_COUNTS):
        for col, label in enumerate(DATASETS):
            ax = axes[row, col]
            for c, cond in enumerate(STANDARD_CONDITIONS):
                s = data[label][(n, c)]
                ax.plot(NU_GRID, s / s.max() + c * OFFSET, color=cond["color"], linewidth=0.9)
            ax.set_title(f"{label} — Ir$_{{{n}}}$", fontsize=TITLE_FONTSIZE)
            ax.set_yticks([])
            ax.tick_params(axis="x", labelsize=TICK_FONTSIZE)
            ax.set_xlim(NU_GRID[0], NU_GRID[-1])
            if row == n_rows - 1:
                ax.set_xlabel(r"Wavenumber (cm$^{-1}$)", fontsize=AXIS_LABEL_FONTSIZE)
            if col == 0:
                ax.set_ylabel("IR intensity (normalised, offset)", fontsize=AXIS_LABEL_FONTSIZE)

    handles = [Line2D([], [], color=cond["color"], linewidth=1.5, label=cond["label"]) for cond in STANDARD_CONDITIONS]
    fig.legend(handles=handles, fontsize=LEGEND_FONTSIZE, frameon=False, ncol=len(handles),
               loc="upper center", bbox_to_anchor=(0.5, 1.0))
    fig.subplots_adjust(left=0.06, right=0.99, top=0.93, bottom=0.08, hspace=0.30, wspace=0.05)
    save_fig(fig, OUTPUT_PATH, dpi=DPI)
    print(f"[supp] {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
