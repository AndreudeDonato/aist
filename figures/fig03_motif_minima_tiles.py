"""Figure 3 – Pooled ACE centroids with per-atom motif colours.

Uses ``ace_centroids_pooled.traj`` (one frame per structure cluster) plus
``motif_atoms_pooled.json`` (per-atom Ir/O motif colours via ``frame_index``).

Layout: 88 × 200 mm — one row per centroid, 2 columns:
  left: cluster tag (coloured square + ``C*``)
  right: structure tile

Output: ``figures/outputs/fig03_motif_minima_tiles.svg``
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from ase.io import read
from ase.visualize.plot import plot_atoms
from matplotlib.offsetbox import AnchoredOffsetbox, DrawingArea, HPacker, TextArea
from matplotlib.patches import Rectangle

from aist.ace.common import struct_cluster_palette
from aist.ace.io import atom_rgbs_for_motif_tile, load_motif_atoms_json
from aist.core.minima import orient_by_inertia_axes
from aist.thermo import ThermoPartitionFunctions

from figures.config import (
    ACE_CENTROIDS,
    DPI,
    MOTIF_JSON,
    OUTPUT_DIR,
)
from figures.fig01_minima import (
    MM,
    ROTATION,
    _half_span,
)
from figures.utils.cache import ensure_for_figures
from figures.utils.io import ensure_dirs

FIG_W_MM = 88
FIG_H_MM = 200
LABEL_COL_W_MM = 10
STRUCT_COL_W_MM = FIG_W_MM - LABEL_COL_W_MM
LABEL_FONTSIZE = 9
TAG_PAD_PT = 3
TAG_SQ_PT = 8

# ``ase.visualize.plot.plot_atoms`` → matplotlib Circle patches (black edges).
OUTLINE_WIDTH = 0.3
HALF_SPAN_ZOOM = 1.08
TILE_BOX_SCALE = 1.95  # scale about row centre; may overlap, row spacing unchanged

OUTPUT_PATH = os.path.join(OUTPUT_DIR, "fig03_motif_minima_tiles.svg")
CENTROID_TRAJ = ACE_CENTROIDS["pooled"]
MOTIF_JSON_POOLED = MOTIF_JSON["pooled"]

plt.rcParams["svg.fonttype"] = "none"


def _load_centroids(traj_path: str) -> list[dict]:
    """One entry per pooled cluster centroid (sorted by cluster id)."""
    tp = ThermoPartitionFunctions()
    structures = list(read(traj_path, ":"))
    out = []
    for atoms in structures:
        info = atoms.info
        frame_idx = int(info.get("frame_index", len(out)))
        cluster = int(info.get("cluster", 0))
        oriented = orient_by_inertia_axes(atoms, tp)
        out.append({
            "atoms": oriented,
            "frame_index": frame_idx,
            "cluster": cluster,
        })
    out.sort(key=lambda e: e["cluster"])
    print(f"[fig03] {len(out)} pooled centroids from {traj_path}")
    return out


def _motif_labels(motif_json: dict, frame_idx: int, n_atoms: int) -> list[str]:
    labels = motif_json.get("frames", {}).get(str(frame_idx))
    if labels is None or len(labels) != n_atoms:
        return ["_"] * n_atoms
    return labels


def _scale_rows_in_place(axes, scale: float) -> None:
    """Enlarge label + structure cells about each row centre (overlap ok; GridSpec gaps fixed)."""
    if scale == 1.0:
        return
    for i in range(axes.shape[0]):
        for col in (0, 1):
            ax = axes[i, col]
            pos = ax.get_position()
            cx = pos.x0 + 0.5 * pos.width
            cy = pos.y0 + 0.5 * pos.height
            nw = pos.width * scale
            nh = pos.height * scale
            ax.set_position([cx - 0.5 * nw, cy - 0.5 * nh, nw, nh])


def _style_atom_plot(ax, atoms, colors) -> None:
    plot_atoms(
        atoms, ax=ax, rotation=ROTATION, radii=None,
        colors=colors, show_unit_cell=0,
    )
    for col in ax.collections:
        col.set_clip_on(False)
        try:
            col.set_linewidths(OUTLINE_WIDTH)
        except Exception:
            pass
    for patch in ax.patches:
        patch.set_clip_on(False)
        try:
            patch.set_linewidth(OUTLINE_WIDTH)
        except Exception:
            pass


def _draw_cluster_tag(ax, cluster: int, struct_cmap) -> None:
    """Centred square + ``C*`` in the label column cell."""
    color = struct_cmap(cluster % struct_cmap.N)
    sq = TAG_SQ_PT
    square = DrawingArea(sq, sq, 0, 0)
    square.add_artist(
        Rectangle(
            (0, 0), sq, sq,
            facecolor=color,
            edgecolor="black",
            linewidth=0.4,
        )
    )
    label = TextArea(
        f"C{cluster}",
        textprops={"fontsize": LABEL_FONTSIZE, "color": "black"},
    )
    box = HPacker(children=[square, label], align="center", pad=0, sep=TAG_PAD_PT)
    ax.set_axis_off()
    ax.add_artist(
        AnchoredOffsetbox(
            loc="center",
            pad=0,
            borderpad=0,
            child=box,
            frameon=False,
        )
    )


def main() -> None:
    ensure_for_figures([3])
    ensure_dirs(OUTPUT_DIR)

    if not os.path.exists(CENTROID_TRAJ):
        raise SystemExit(f"[fig03] missing centroid traj: {CENTROID_TRAJ}")
    if not os.path.exists(MOTIF_JSON_POOLED):
        raise SystemExit(f"[fig03] missing motif JSON: {MOTIF_JSON_POOLED}")

    entries = _load_centroids(CENTROID_TRAJ)
    motif_json = load_motif_atoms_json(MOTIF_JSON_POOLED)
    k_struct = max(int(e["cluster"]) for e in entries) + 1
    struct_cmap = struct_cluster_palette(k_struct)

    combined = [(0, 0, e["atoms"]) for e in entries]
    shared_half_span = _half_span(combined) * HALF_SPAN_ZOOM
    print(f"[fig03] shared half_span: {shared_half_span:.4f} (×{HALF_SPAN_ZOOM})")

    n_tiles = len(entries)
    tile_h_mm = FIG_H_MM / n_tiles
    fig = plt.figure(figsize=(FIG_W_MM * MM, FIG_H_MM * MM), dpi=DPI)
    outer = fig.add_gridspec(
        n_tiles,
        2,
        width_ratios=[LABEL_COL_W_MM, STRUCT_COL_W_MM],
        height_ratios=[tile_h_mm] * n_tiles,
        wspace=0.04,
        hspace=0.01,
    )
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)

    axes = np.empty((n_tiles, 2), dtype=object)
    for i, entry in enumerate(entries):
        ax_lbl = fig.add_subplot(outer[i, 0])
        ax_img = fig.add_subplot(outer[i, 1])
        axes[i, 0] = ax_lbl
        axes[i, 1] = ax_img

        ax_img.set_axis_off()
        ax_img.set_facecolor((1.0, 1.0, 1.0, 0.0))
        ax_img.set_clip_on(False)

        atoms = entry["atoms"]
        cluster = int(entry["cluster"])
        frame_idx = int(entry["frame_index"])
        symbols = atoms.get_chemical_symbols()
        motif_labels = _motif_labels(motif_json, frame_idx, len(symbols))
        struct_color = struct_cmap(cluster % struct_cmap.N)
        colors = atom_rgbs_for_motif_tile(
            symbols, motif_labels, motif_json,
            mode="atom", struct_cluster=cluster, struct_color=struct_color,
        )
        _style_atom_plot(ax_img, atoms, colors)

        xc = 0.5 * sum(ax_img.get_xlim())
        yc = 0.5 * sum(ax_img.get_ylim())
        ax_img.set_xlim(xc - shared_half_span, xc + shared_half_span)
        ax_img.set_ylim(yc - shared_half_span, yc + shared_half_span)
        ax_img.set_aspect("equal", adjustable="box")

    _scale_rows_in_place(axes, TILE_BOX_SCALE)
    fig.canvas.draw()

    for i, entry in enumerate(entries):
        _draw_cluster_tag(axes[i, 0], int(entry["cluster"]), struct_cmap)

    fig.savefig(OUTPUT_PATH, format="svg", dpi=DPI, bbox_inches=None, pad_inches=0)
    plt.close(fig)
    print(f"[save] {OUTPUT_PATH}")
    print(
        f"\n[fig03] {OUTPUT_PATH}  ({FIG_W_MM}×{FIG_H_MM} mm, "
        f"label {LABEL_COL_W_MM} | structure {STRUCT_COL_W_MM} mm)"
    )


if __name__ == "__main__":
    main()
