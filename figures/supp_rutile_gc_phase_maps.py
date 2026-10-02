"""Supplementary – IrO₂ rutile vs gas oxides grand-canonical phase diagrams.

Three-species ensemble per panel: IrO₂(rutile), IrO₂(g), IrO₃(g).
Dedicated cache under ``figures/cache/rutile_gc/`` (not the main cluster GC pool).
Grid: T = 100–1500 K, 500×500; fully novib; gas *P* = ``RUTILE_GC_GAS_P`` (1 Pa).

Layout: 1×2 — PBE | PBE0, shared axes, ⟨n_O/n_Ir⟩ with global-minimum stoichiometry labels.

Output: ``figures/outputs/supp_rutile_gc_phase_maps.svg``
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

from figures.config import (
    AMBIENT_PO2_PA,
    DPI,
    OUTPUT_DIR,
    PHASE_CMAP,
    RUTILE_GC_CALIBRATION_T_K,
    RUTILE_GC_NPZ,
)
from figures.utils.io import ensure_dirs, load_gc_npz
from figures.utils.plotting import (
    TITLE_FONTSIZE,
    add_phase_colorbar,
    draw_global_minimum_phase_overlay,
    margin_colorbar_cax,
    save_fig,
    stoichiometry_overlay_labels,
    style_phase_ax,
)
from figures.utils.rutile_gc_cache import ensure_rutile_gc

OUTPUT_PATH = os.path.join(OUTPUT_DIR, "supp_rutile_gc_phase_maps.svg")
FIG_W_MM = 180
FIG_H_MM = 70
LAYOUT_RECT = [0, 0, 0.88, 1]
CBAR_GAP = 0.012
CBAR_WIDTH = 0.015


def _panel(ax, gc: dict, overlay_labels: list[str], vmin: float, vmax: float):
    Ts, Ps = gc["Ts"], gc["Ps"]
    Z = np.asarray(gc["x_per_non_o"], dtype=float)
    mesh = ax.pcolormesh(
        Ts, Ps, Z, shading="auto",
        cmap=PHASE_CMAP,
        norm=Normalize(vmin=vmin, vmax=vmax),
        rasterized=True,
    )
    draw_global_minimum_phase_overlay(
        ax, Ts, Ps, gc["phase_index"], overlay_labels,
    )
    ax.plot(
        RUTILE_GC_CALIBRATION_T_K,
        AMBIENT_PO2_PA,
        marker="x",
        color="black",
        markersize=4,
        markeredgewidth=0.8,
        linestyle="none",
        zorder=10,
    )
    style_phase_ax(ax, apply_gc_t_ticks=False)
    return mesh


def main() -> None:
    ensure_rutile_gc()
    ensure_dirs(OUTPUT_DIR)

    panels: dict[str, dict] = {}
    for label, path in RUTILE_GC_NPZ.items():
        panels[label] = load_gc_npz(path, light=True, frames=False)

    formulas = panels["PBE"]["formulas"]
    overlay_labels = stoichiometry_overlay_labels(formulas)
    print(f"[supp rutile] species: {[str(f) for f in formulas]}")

    vmax = max(float(gc["x_per_non_o"].max()) for gc in panels.values())
    vmin = 0.0
    if vmax <= vmin:
        vmax = 3.0

    fig, axes = plt.subplots(
        1, 2,
        figsize=(FIG_W_MM / 25.4, FIG_H_MM / 25.4),
        dpi=DPI,
        sharex=True,
        sharey=True,
    )

    mesh_ref = None
    for col, label in enumerate(("PBE", "PBE0")):
        mesh = _panel(axes[col], panels[label], overlay_labels, vmin, vmax)
        axes[col].set_title(label, fontsize=TITLE_FONTSIZE)
        if col == 1:
            axes[col].set_ylabel("")
            axes[col].tick_params(labelleft=False)
        if mesh_ref is None:
            mesh_ref = mesh

    fig.tight_layout(pad=6 / 25.4, rect=LAYOUT_RECT)
    if mesh_ref is not None:
        add_phase_colorbar(
            fig, mesh_ref,
            cax=margin_colorbar_cax(fig, axes[0], axes[1], gap=CBAR_GAP, width=CBAR_WIDTH),
            label=r"$n_O/n_{Ir}$",
        )

    save_fig(fig, OUTPUT_PATH, dpi=DPI)
    print(f"[supp] {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
