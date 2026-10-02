"""Supplementary – Vibrational vs. non-vibrational phase diagrams, PBE | PBE0.

Former Figure 7. Pooled Ir₄–Ir₆ grand-canonical ensembles of each functional, 2×4 layout
(columns PBE full | PBE no vib | PBE0 full | PBE0 no vib):
  row 0  with ensemble:     x_per_non_o with vib/ZPE | recomputed without vib
  row 1  single minimum:    single-minimum stoichiometry with vib (full G) | without (novib G)

Requires figures/cache/gc_variants.npz and gc_variants_PBE0.npz.

Output: figures/outputs/supp_vib_vs_novib_quadrants.svg
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

from figures.config import OUTPUT_DIR, DPI
from figures.utils.datasets import DATASETS, dataset_inputs
from figures.utils.io import ensure_dirs, load_gc_npz, load_variants_like
from figures.utils.cache import ensure_for_supplementary
from figures.utils.plotting import (
    TITLE_FONTSIZE,
    add_phase_colorbar,
    draw_global_minimum_phase_overlay,
    margin_colorbar_cax,
    save_fig,
    stoichiometry_overlay_labels,
    style_phase_ax,
)

OUTPUT_PATH = os.path.join(OUTPUT_DIR, "supp_vib_vs_novib_quadrants.svg")
PHASE_CMAP = "coolwarm"

TITLES = {
    (0, 0): "full (vib + ensemble)",
    (0, 1): "no vib/ZPE, ensemble",
    (1, 0): "with vib, single minimum",
    (1, 1): "no vib/ZPE, single minimum",
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
    ax.set_title(title, fontsize=TITLE_FONTSIZE)
    return mesh


def main() -> None:
    ensure_for_supplementary(["supp_vib_vs_novib_quadrants"])
    ensure_dirs(OUTPUT_DIR)

    data = {}
    for label in DATASETS:
        inp = dataset_inputs(label)
        gc = load_gc_npz(inp.gc_npz, light=True)
        vd = load_variants_like(inp.variants_npz, gc, (
            "Ts", "Ps", "stoich", "full_x_per_non_o", "full_phase_index", "novib_x_per_non_o",
            "novib_phase_index", "full_stoich_map", "novib_stoich_map",
        ))
        data[label] = {
            "Ts": vd["Ts"],
            "Ps": vd["Ps"],
            "stoich": np.asarray(vd["stoich"], dtype=float),
            "labels": stoichiometry_overlay_labels(gc["formulas"]),
            "panels": {
                (0, 0): (vd["full_x_per_non_o"], vd["full_phase_index"]),
                (0, 1): (vd["novib_x_per_non_o"], vd["novib_phase_index"]),
                (1, 0): (vd["full_stoich_map"], vd["full_phase_index"]),
                (1, 1): (vd["novib_stoich_map"], vd["novib_phase_index"]),
            },
        }
    vmin = min(float(d["stoich"].min()) for d in data.values())
    vmax = max(float(d["stoich"].max()) for d in data.values())

    fig, axes = plt.subplots(2, 2 * len(DATASETS), figsize=(180 / 25.4, 100 / 25.4), dpi=DPI)
    mesh_ref = None
    for i, label in enumerate(DATASETS):
        d = data[label]
        for (row, col), (z_map, phase_idx) in d["panels"].items():
            ax = axes[row, 2 * i + col]
            mesh = _phase_map_panel(
                ax, d["Ts"], d["Ps"], z_map, phase_idx, d["labels"],
                f"{label}\n{TITLES[(row, col)]}", vmin, vmax,
            )
            if mesh_ref is None:
                mesh_ref = mesh

    fig.tight_layout(pad=6 / 25.4, rect=[0, 0, 0.93, 1])
    if mesh_ref is not None:
        add_phase_colorbar(
            fig, mesh_ref,
            cax=margin_colorbar_cax(fig, axes[0, 0], axes[1, -1]),
            label=r"$n_O / n_{Ir}$",
        )
    save_fig(fig, OUTPUT_PATH, dpi=DPI)
    print(f"\n[supp] {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
