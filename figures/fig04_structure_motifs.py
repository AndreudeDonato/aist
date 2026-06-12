"""Figure 4 – Structure-level motif classification.

1 × 3 layout (180 mm wide), same column order as Figure 2 atom panel:
  Structure PCA scatter | PBE populations | PBE0 populations

Output: ``figures/outputs/fig04_structure_motifs.svg``
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from aist.ace.common import struct_cluster_palette
from aist.ace.datasets import load_dataset
from aist.ace.plotting import struct_population_stack_vs_oxidation

from figures.config import (
    ACE_CENTROIDS,
    ACE_FEATURES,
    ACE_STRUCT_NPZ,
    DATASET_COLORS,
    DPI,
    MOTIF_JSON,
    OUTPUT_DIR,
    TRAJ_PBE,
    TRAJ_PBE0,
    VISUALIZE_SCRIPT,
)
from figures.utils.cache import ensure_for_figures
from figures.utils.io import ensure_dirs, load_npz
from figures.utils.plotting import draw_population_stack

OUTPUT_PATH = os.path.join(OUTPUT_DIR, "fig04_structure_motifs.svg")
FIG_W_MM = 180
ROW_H_MM = 60  # match Figure 2 row height
N_BINS = 100
SMOOTH_SIGMA = 1.0

plt.rcParams["svg.fonttype"] = "none"


def _draw_paper_struct_pca(ax, struct_cache: dict, *, fontsize: int = 7) -> None:
    """Pooled structure PCA: legend ``C0``, …; title ``structures - PCA``."""
    k = int(struct_cache["k"])
    Xp = np.asarray(struct_cache["X_plot"])
    labels = np.asarray(struct_cache["labels"], dtype=int)
    ds_idx = np.asarray(struct_cache["dataset_index"], dtype=int)
    ds_names = [str(x) for x in np.asarray(struct_cache["dataset_names"]).tolist()]
    cmap = struct_cluster_palette(max(k, 1))
    ax_labels = struct_cache.get("plot_axes", ("PC1", "PC2"))
    markers = {"PBE": "o", "PBE0": "s"}

    for di, name in enumerate(ds_names):
        sel = ds_idx == di
        if not sel.any():
            continue
        mk = markers.get(name, "o")
        ds_color = DATASET_COLORS.get(name, "black")
        for c in range(k):
            mask = sel & (labels == c)
            if not mask.any():
                continue
            ax.scatter(
                Xp[mask, 0], Xp[mask, 1],
                c=[cmap(c % cmap.N)], s=16, alpha=0.75,
                marker=mk, edgecolors=ds_color, linewidths=0.35,
            )

    cluster_handles = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor=cmap(c % cmap.N),
               markersize=6, label=f"C{c}")
        for c in range(k)
    ]
    ds_handles = [
        Line2D([0], [0], marker=markers.get(name, "o"), color="w",
               markerfacecolor="none", markeredgecolor=DATASET_COLORS.get(name, "k"),
               markersize=6, label=name)
        for name in ds_names
    ]
    ax.set_xlabel(ax_labels[0], fontsize=fontsize)
    ax.set_ylabel(ax_labels[1], fontsize=fontsize)
    ax.set_title("structures - PCA", fontsize=fontsize)
    ax.tick_params(axis="both", labelsize=fontsize - 1)
    ax.legend(
        handles=cluster_handles + ds_handles,
        fontsize=fontsize - 1, frameon=False, loc="best", ncol=2,
    )


def main() -> None:
    ensure_for_figures([4])
    ensure_dirs(OUTPUT_DIR)

    struct_cache = load_npz(ACE_STRUCT_NPZ)

    datasets = []
    for label, traj_path, csv_path in [
        ("PBE", TRAJ_PBE, ACE_FEATURES["PBE"]),
        ("PBE0", TRAJ_PBE0, ACE_FEATURES["PBE0"]),
    ]:
        if not os.path.exists(traj_path):
            print(f"[skip] {label}")
            continue
        datasets.append(load_dataset(label, traj_path, csv_path))

    if not datasets:
        raise SystemExit("[fig04] No datasets found.")

    k = int(struct_cache["k"])
    cmap = struct_cluster_palette(k)
    colors = [cmap(c % cmap.N) for c in range(k)]

    fig, axes = plt.subplots(1, 3, figsize=(FIG_W_MM / 25.4, ROW_H_MM / 25.4), dpi=DPI)

    _draw_paper_struct_pca(axes[0], struct_cache, fontsize=7)

    for col_idx, d in enumerate(datasets[:2]):
        ax = axes[col_idx + 1]
        x_grid, Y = struct_population_stack_vs_oxidation(
            d, d["struct_labels"], k,
            n_bins=N_BINS, smooth_sigma=SMOOTH_SIGMA,
        )
        labels = [f"C{c}" for c in range(k)]
        draw_population_stack(
            ax, x_grid, Y, colors,
            labels=labels,
            xlabel=r"$n_O / n_{Ir}$",
            ylabel="Structure cluster pop. (%)",
            title=f"{d['label']} — structure clusters vs oxidation",
            legend_ncol=2,
        )
        ax.tick_params(axis="both", labelsize=6)

    fig.tight_layout(pad=6 / 25.4)
    fig.savefig(OUTPUT_PATH, format="svg", dpi=DPI, bbox_inches=None, pad_inches=0)
    plt.close(fig)
    print(f"[save] {OUTPUT_PATH}")
    print(f"\n[fig04] {OUTPUT_PATH}")
    print(
        f"  Tip: run 'python {VISUALIZE_SCRIPT} "
        f"{ACE_CENTROIDS['pooled']} {MOTIF_JSON['pooled']}' "
        f"to pick representative cluster structures."
    )


if __name__ == "__main__":
    main()
