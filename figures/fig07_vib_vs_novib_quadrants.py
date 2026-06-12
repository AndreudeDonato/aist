"""Figure 7 – Vibrational vs. non-vibrational phase diagrams.

2×2 quadrant layout (PBE — full vib/ZPE thermo from ``IryOxPBE.traj``):
  (0,0) Full (vib + ensemble):         existing x_per_non_o
  (0,1) No vib/ZPE, with ensemble:     x_per_non_o recomputed without vib
  (1,0) With vib, no ensemble:         single-minimum stoichiometry (full G)
  (1,1) No vib/ZPE, no ensemble:       single-minimum stoichiometry (novib G)

Requires figures/data/gc_variants.npz (produced by
figures/data/compute_gc_variants.py on first run).

Output: figures/outputs/fig07_vib_vs_novib_quadrants.svg
"""
from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in [_ROOT, os.path.join(_ROOT, "analysis")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from figures.config import GC_NPZ, GC_VARIANTS_NPZ, OUTPUT_DIR, DPI
from figures.utils.io import ensure_dirs, load_gc_npz
from figures.utils.cache import ensure_for_figures
from figures.utils.plotting import (
    draw_global_minimum_phase_overlay,
    save_fig,
    stoichiometry_overlay_labels,
    style_phase_ax,
)

OUTPUT_PATH = os.path.join(OUTPUT_DIR, "fig07_vib_vs_novib_quadrants.svg")
PHASE_CMAP = "coolwarm"

TITLES = {
    (0, 0): "Full (vib + ensemble)",
    (0, 1): "No vib/ZPE, with ensemble",
    (1, 0): "With vib, single minimum",
    (1, 1): "No vib/ZPE, single minimum",
}


def _phase_map_panel(
    ax,
    Ts,
    Ps,
    z_map,
    phase_index,
    overlay_labels,
    title,
    vmin,
    vmax,
):
    norm = Normalize(vmin=vmin, vmax=vmax)
    mesh = ax.pcolormesh(
        Ts, Ps, z_map, shading="auto",
        cmap=PHASE_CMAP, norm=norm, rasterized=True,
    )
    draw_global_minimum_phase_overlay(
        ax, Ts, Ps, phase_index, overlay_labels,
    )
    style_phase_ax(ax)
    ax.set_title(title, fontsize=7)
    return mesh


def main() -> None:
    ensure_for_figures([7])
    ensure_dirs(OUTPUT_DIR)

    vd = np.load(GC_VARIANTS_NPZ, allow_pickle=True)
    Ts = vd["Ts"]
    Ps = vd["Ps"]

    gc = load_gc_npz(GC_NPZ)
    overlay_labels = stoichiometry_overlay_labels(gc["formulas"])

    stoich = vd["stoich"]
    vmin = float(stoich.min())
    vmax = float(stoich.max())

    panels = {
        (0, 0): (vd["full_x_per_non_o"], vd["full_phase_index"]),
        (0, 1): (vd["novib_x_per_non_o"], vd["novib_phase_index"]),
        (1, 0): (vd["full_stoich_map"], vd["full_phase_index"]),
        (1, 1): (vd["novib_stoich_map"], vd["novib_phase_index"]),
    }

    fig, axes = plt.subplots(2, 2, figsize=(180 / 25.4, 120 / 25.4), dpi=DPI)
    mesh_ref = None
    for (row, col), (z_map, phase_idx) in panels.items():
        ax = axes[row, col]
        mesh = _phase_map_panel(
            ax, Ts, Ps, z_map, phase_idx, overlay_labels,
            TITLES[(row, col)], vmin, vmax,
        )
        if mesh_ref is None:
            mesh_ref = mesh

    fig.tight_layout(pad=6 / 25.4, rect=[0, 0, 0.84, 1])
    if mesh_ref is not None:
        cbar = fig.colorbar(
            mesh_ref,
            ax=axes,
            location="right",
            pad=0.06,
            fraction=0.04,
            shrink=0.92,
            label=r"$n_O / n_{Ir}$",
        )
        cbar.ax.tick_params(labelsize=7)
    save_fig(fig, OUTPUT_PATH, dpi=DPI)
    print(f"\n[fig07] {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
