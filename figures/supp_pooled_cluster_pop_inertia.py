"""Supplementary – cluster population vs n_Ir + inertia shape triangle.

88 mm wide:
  Row 1 — PBE | PBE0 structure-cluster population (%) vs n_Ir (phase stacks, pooled C*)
  Row 2 — Principal moment-ratio triangle (S–L–P only), all structures coloured by n_Ir

Output: ``figures/outputs/supp_pooled_cluster_pop_inertia.svg``
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize

from aist.ace.common import struct_cluster_palette
from aist.ace.datasets import load_dataset
from aist.inertia import inertia_shape_extent_features
from aist.thermo import ThermoPartitionFunctions
from aist.ace.constants import ACE_EXTENT_BLEND, ACE_EXTENT_NORMALIZATION

from figures.config import (
    ACE_FEATURES,
    ACE_STRUCT_NPZ,
    DPI,
    OUTPUT_DIR,
    SUPP_IR_COUNTS,
    TRAJ_PBE,
    TRAJ_PBE0,
)
from figures.utils.cache import ensure_for_supplementary
from figures.utils.inertia_triangle import style_bare_inertia_triangle
from figures.utils.io import ensure_dirs, load_npz
from figures.utils.plotting import (
    AXIS_LABEL_FONTSIZE,
    TICK_FONTSIZE,
    TITLE_FONTSIZE,
    discrete_category_colorbar,
    margin_colorbar_cax,
    save_fig,
)

OUTPUT_PATH = os.path.join(OUTPUT_DIR, "supp_pooled_cluster_pop_inertia.svg")
FIG_W_MM = 88
ROW_H_MM = 50
FIG_H_MM = 2 * ROW_H_MM
CBAR_GAP = 0.012
CBAR_WIDTH = 0.025
LAYOUT_RECT = [0, 0, 0.88, 1]
EXTENT_NORMALIZATION = ACE_EXTENT_NORMALIZATION
EXTENT_BLEND = ACE_EXTENT_BLEND
IR_CMAP = "winter"
MARKER_SCALE = 0.10
SCATTER_S = 8.0
SCATTER_ALPHA = 0.55
POP_YLABEL = "pop %"

DATASETS = [
    ("PBE", TRAJ_PBE, ACE_FEATURES["PBE"]),
    ("PBE0", TRAJ_PBE0, ACE_FEATURES["PBE0"]),
]


def struct_population_at_n_ir(
    n_ir: np.ndarray,
    labels: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Population (%) of each cluster at each discrete n_Ir value."""
    sizes = sorted(np.unique(n_ir))
    Y = np.zeros((k, len(sizes)), dtype=float)
    labels = np.asarray(labels, dtype=int)
    for j, ni in enumerate(sizes):
        mask = n_ir == ni
        counts = np.bincount(labels[mask], minlength=k)[:k]
        total = counts.sum()
        if total > 0:
            Y[:, j] = 100.0 * counts / total
    return np.asarray(sizes, dtype=float), Y


def _load_datasets() -> list[dict]:
    out: list[dict] = []
    for label, traj_path, csv_path in DATASETS:
        if not os.path.exists(traj_path):
            print(f"[skip] {label}: {traj_path}")
            continue
        out.append(load_dataset(label, traj_path, csv_path))
    if not out:
        raise SystemExit("[supp] No datasets loaded.")
    return out


def _load_inertia_points(datasets: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tp = ThermoPartitionFunctions()
    xs, ys, n_ir_vals = [], [], []
    for d in datasets:
        for atoms, ni in zip(d["structures"], d["n_Ir"], strict=True):
            x_s, y_s, _ = inertia_shape_extent_features(
                atoms, tp,
                extent_normalization=EXTENT_NORMALIZATION,
                extent_blend=EXTENT_BLEND,
            )
            xs.append(x_s)
            ys.append(y_s)
            n_ir_vals.append(int(ni))
    return (
        np.asarray(xs, dtype=float),
        np.asarray(ys, dtype=float),
        np.asarray(n_ir_vals, dtype=int),
    )


def _ir_size_colors() -> tuple[list, list[str], ListedColormap, BoundaryNorm]:
    """Discrete winter colours for Ir₃…Ir₉ (matches fig. 8 / fig. 12)."""
    cont_norm = Normalize(vmin=float(min(SUPP_IR_COUNTS)), vmax=float(max(SUPP_IR_COUNTS)))
    base = plt.get_cmap(IR_CMAP)
    labels = [f"Ir{n}" for n in SUPP_IR_COUNTS]
    colors = [base(cont_norm(float(n))) for n in SUPP_IR_COUNTS]
    bounds = np.arange(min(SUPP_IR_COUNTS) - 0.5, max(SUPP_IR_COUNTS) + 1.5, 1.0)
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(bounds, cmap.N)
    return colors, labels, cmap, norm


def _draw_population_phase(
    ax,
    n_ir: np.ndarray,
    labels: np.ndarray,
    k: int,
    colors: list,
    *,
    title: str = "",
    show_ylabel: bool = True,
    show_xlabel: bool = True,
) -> None:
    x_grid, Y = struct_population_at_n_ir(n_ir, labels, k)
    if Y.size == 0 or not np.any(Y.sum(axis=0)):
        ax.text(
            0.5, 0.5, "No data", ha="center", va="center",
            fontsize=AXIS_LABEL_FONTSIZE, transform=ax.transAxes,
        )
        return

    bottom = np.zeros(len(x_grid), dtype=float)
    for c in range(k):
        ax.fill_between(
            x_grid, bottom, bottom + Y[c],
            step="mid",
            color=colors[c], alpha=0.85,
            edgecolor="none", linewidth=0.0, zorder=2,
        )
        bottom += Y[c]

    ax.set_xlim(float(x_grid.min()), float(x_grid.max()))
    ax.set_ylim(0, 100)
    ax.set_ymargin(0)
    ax.margins(y=0)
    ax.set_xticks(x_grid.astype(int))
    if show_xlabel:
        ax.set_xlabel(r"$n_{\mathrm{Ir}}$", fontsize=AXIS_LABEL_FONTSIZE)
    else:
        ax.tick_params(labelbottom=False)
        ax.set_xlabel("")
    if show_ylabel:
        ax.set_ylabel(POP_YLABEL, fontsize=AXIS_LABEL_FONTSIZE)
    else:
        ax.tick_params(labelleft=False)
        ax.set_ylabel("")
    if title:
        ax.set_title(title, fontsize=TITLE_FONTSIZE, pad=3)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)


def _draw_inertia_triangle(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    n_ir: np.ndarray,
    *,
    cmap: ListedColormap,
    norm: BoundaryNorm,
) -> None:
    style_bare_inertia_triangle(ax, marker_scale=MARKER_SCALE)
    ax.scatter(
        x, y, c=n_ir, cmap=cmap, norm=norm, s=SCATTER_S, alpha=SCATTER_ALPHA,
        edgecolors="none", zorder=2,
    )


def main() -> None:
    ensure_for_supplementary(["supp_pooled_cluster_pop_inertia"])
    ensure_dirs(OUTPUT_DIR)

    datasets = _load_datasets()
    struct_cache = load_npz(ACE_STRUCT_NPZ)
    k = int(struct_cache["k"])
    cmap = struct_cluster_palette(k)
    colors = [cmap(c % cmap.N) for c in range(k)]

    ix, iy, in_ir = _load_inertia_points(datasets)

    fig = plt.figure(figsize=(FIG_W_MM / 25.4, FIG_H_MM / 25.4), dpi=DPI)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.38, wspace=0.12)
    ax_pbe = fig.add_subplot(gs[0, 0])
    ax_pbe0 = fig.add_subplot(gs[0, 1], sharex=ax_pbe, sharey=ax_pbe)
    ax_inertia = fig.add_subplot(gs[1, :])

    for d, ax, title in zip(datasets, (ax_pbe, ax_pbe0), ("PBE", "PBE0"), strict=True):
        _draw_population_phase(
            ax, d["n_Ir"], d["struct_labels"], k, colors,
            title=title,
            show_ylabel=(ax is ax_pbe),
            show_xlabel=True,
        )

    ir_colors, ir_labels, ir_cmap, ir_norm = _ir_size_colors()
    _draw_inertia_triangle(ax_inertia, ix, iy, in_ir, cmap=ir_cmap, norm=ir_norm)

    fig.tight_layout(pad=6 / 25.4, rect=LAYOUT_RECT)
    cluster_labels = [f"C{c}" for c in range(k)]
    discrete_category_colorbar(
        colors, cluster_labels,
        cax=margin_colorbar_cax(
            fig, ax_pbe, ax_pbe0,
            gap=CBAR_GAP, width=CBAR_WIDTH,
        ),
        cbar_label="Cluster",
    )
    discrete_category_colorbar(
        ir_colors, ir_labels,
        cax=margin_colorbar_cax(
            fig, ax_inertia, ax_inertia,
            gap=CBAR_GAP, width=CBAR_WIDTH,
        ),
        cbar_label=r"$n_{\mathrm{Ir}}$",
    )

    n_struct = sum(len(d["n_Ir"]) for d in datasets)

    save_fig(fig, OUTPUT_PATH, dpi=DPI)
    print(
        f"[supp] {OUTPUT_PATH}  "
        f"({len(datasets)} datasets, {n_struct} structures, {k} clusters)"
    )


if __name__ == "__main__":
    main()
