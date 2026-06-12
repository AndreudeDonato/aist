"""Figure 2 – Atom motif classification.

2 × 3 layout (180 mm wide), same column order as Figure 4:
  Row 1 (Ir): UMAP scatter | PBE populations | PBE0 populations
  Row 2 (O):  UMAP scatter | PBE populations | PBE0 populations

Output: ``figures/outputs/fig02_atom_motifs.svg``
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

from aist.ace.common import discrete_palette
from aist.ace.datasets import load_dataset
from aist.ace.io import atom_fit_from_npz
from aist.ace.plotting import atom_population_stack_vs_oxidation

from figures.config import (
    ACE_ATOM_IR_NPZ,
    ACE_ATOM_O_NPZ,
    ACE_FEATURES,
    DPI,
    IR_CMAP,
    O_CMAP,
    OUTPUT_DIR,
    TRAJ_PBE,
    TRAJ_PBE0,
)
from figures.utils.cache import ensure_for_figures
from figures.utils.io import ensure_dirs
from figures.utils.plotting import draw_population_stack

OUTPUT_PATH = os.path.join(OUTPUT_DIR, "fig02_atom_motifs.svg")
FIG_W_MM = 180
ROW_H_MM = 60  # match Figure 3 row height
FIG_H_MM = 2 * ROW_H_MM
N_BINS = 100
SMOOTH_SIGMA = 1.0


def _draw_atom_pop(ax, d, species: str, k: int):
    key = "per_struct_ir_cluster" if species == "Ir" else "per_struct_o_cluster"
    cmap_name = IR_CMAP if species == "Ir" else O_CMAP
    x_grid, Y = atom_population_stack_vs_oxidation(
        d, k, key, n_bins=N_BINS, smooth_sigma=SMOOTH_SIGMA,
    )
    colors = discrete_palette(cmap_name, k)
    labels = [f"{species}-A{c}" for c in range(k)]
    draw_population_stack(
        ax, x_grid, Y, colors,
        labels=labels,
        xlabel=r"$n_O / n_{Ir}$",
        ylabel=f"{species} motif pop. (%)",
        title=f"{d['label']} — {species} motifs vs oxidation",
        legend_ncol=2,
    )
    ax.tick_params(axis="both", labelsize=6)


def _draw_paper_atom_scatter(ax, fit, species: str, cmap_name: str, *, fontsize: int = 7) -> None:
    """UMAP scatter with paper labels: ``Ir-A0``, title ``{species} atoms - UMAP``."""
    k = int(fit["k"])
    Xp = np.asarray(fit["X_plot"])
    labels = np.asarray(fit["labels"], dtype=int)
    colors = discrete_palette(cmap_name, k) if k > 0 else []
    ax_labels = fit.get("plot_axes", ("UMAP-1", "UMAP-2"))

    for c in range(k):
        mask = labels == c
        if not mask.any():
            continue
        ax.scatter(
            Xp[mask, 0], Xp[mask, 1],
            c=[colors[c]], s=5, alpha=0.65, edgecolors="none",
            label=f"{species}-A{c}",
        )
    ax.set_xlabel(ax_labels[0], fontsize=fontsize)
    ax.set_ylabel(ax_labels[1], fontsize=fontsize)
    ax.set_title(f"{species} atoms - UMAP", fontsize=fontsize)
    ax.tick_params(axis="both", labelsize=fontsize - 1)
    ax.legend(fontsize=fontsize - 1, frameon=False, loc="best", markerscale=2, ncol=2)


def main() -> None:
    ensure_for_figures([2])
    ensure_dirs(OUTPUT_DIR)

    ir_fit = atom_fit_from_npz(ACE_ATOM_IR_NPZ)
    o_fit = atom_fit_from_npz(ACE_ATOM_O_NPZ)

    datasets = []
    for label, traj_path, csv_path in [
        ("PBE", TRAJ_PBE, ACE_FEATURES["PBE"]),
        ("PBE0", TRAJ_PBE0, ACE_FEATURES["PBE0"]),
    ]:
        if not os.path.exists(traj_path):
            print(f"[skip] {label}: {traj_path}")
            continue
        datasets.append(load_dataset(label, traj_path, csv_path))

    if not datasets:
        raise SystemExit("[fig02] No datasets loaded.")

    k_Ir = int(ir_fit["k"])
    k_O = int(o_fit["k"])

    fig, axes = plt.subplots(2, 3, figsize=(FIG_W_MM / 25.4, FIG_H_MM / 25.4), dpi=DPI)

    _draw_paper_atom_scatter(axes[0, 0], ir_fit, "Ir", IR_CMAP, fontsize=7)
    _draw_paper_atom_scatter(axes[1, 0], o_fit, "O", O_CMAP, fontsize=7)

    for col_idx, d in enumerate(datasets[:2]):
        _draw_atom_pop(axes[0, col_idx + 1], d, "Ir", k_Ir)
        _draw_atom_pop(axes[1, col_idx + 1], d, "O", k_O)

    fig.tight_layout(pad=6 / 25.4)
    fig.savefig(OUTPUT_PATH, format="svg", dpi=DPI, bbox_inches=None, pad_inches=0)
    plt.close(fig)
    print(f"[save] {OUTPUT_PATH}")
    print(f"\n[fig02] {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
