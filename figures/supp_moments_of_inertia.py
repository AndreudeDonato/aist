"""Supplementary – moments of inertia shape + extent (3×2).

Column 1 — 3D shape × extent panels (same view/markers as fig. 2 moment-ratio column).
Column 2 — S/L/P floor triangle only (perpendicular to extent), same colouring.

Rows:
  1. O atoms — parent-structure O motif composition (RGB mix)
  2. Ir atoms — parent-structure Ir motif composition (RGB mix)
  3. Structures — pooled structure cluster (C*)

Output: ``figures/outputs/supp_momentsofinertia.svg``

Interactive viewer (rotate 3D panels, synced; press ``p`` to print elev/azim):

    python figures/supp_moments_of_inertia.py --interactive
"""
from __future__ import annotations

import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np
from ase.io import read
from matplotlib.colors import to_rgb

from aist.ace.common import struct_cluster_palette
from aist.inertia import inertia_shape_extent_features
from aist.thermo import ThermoPartitionFunctions

from figures.config import (
    ACE_FEATURES,
    DPI,
    OUTPUT_DIR,
    TRAJ_PBE,
    TRAJ_PBE0,
)
from figures.utils.cache import ensure_for_supplementary
from figures.utils.inertia_triangle import (
    MOMENT_RATIO_ZORDER_DATA,
    overlay_moment_ratio_corner_guides_3d,
    style_bare_inertia_triangle,
    style_inertia_shape_extent_3d,
)
from figures.utils.io import ensure_dirs
from figures.utils.plotting import ir_motif_colors, o_motif_colors, save_fig

OUTPUT_PATH = os.path.join(OUTPUT_DIR, "supp_momentsofinertia.svg")
FIG_W_MM = 180
FIG_H_MM = 220  # compact vs prior 300 mm; keep rows visually separated
COL_WSPACE = 0.06
EXTENT_NORMALIZATION = "partial"
EXTENT_BLEND = 0.1
# Trim empty Axes3D margins only; keep 2D gaps positive so triangle labels do not collide.
PANEL_HSPACE = -0.18
PANEL_HSPACE_2D = 0.08
PANEL_GRID_LEFT = 0.0
PANEL_GRID_RIGHT = 1.0
# Expand axes box into margins after layout (embedded multipanel figures).
MOMENT_RATIO_AXES_LEFT_EXPAND = 0.12
MOMENT_RATIO_AXES_RIGHT_EXPAND = 0.04
# Fig. 2 moment-ratio column (shared with ``fig02_motif_clustering_panel``).
MOMENT_RATIO_EMBED_MARKER_SCALE = 0.13
MOMENT_RATIO_EMBED_MARKER_S = 11.0
MOMENT_RATIO_EMBED_LEFT_EXPAND = 0.22
MOMENT_RATIO_EMBED_RIGHT_EXPAND = 0.05
MOMENT_RATIO_EMBED_AZIM = -50.0
FLOOR_MARKER_SCALE = 0.10
MARKER_SCALE = MOMENT_RATIO_EMBED_MARKER_SCALE
MARKER_S = MOMENT_RATIO_EMBED_MARKER_S
ATOM_ALPHA = 0.35
INTERACTIVE_MAX_POINTS = 5000
INTERACTIVE_MARKER_S = 6.0
INTERACTIVE_STRUCT_MARKER_S = 10.0
INTERACTIVE_DPI = 100
INTERACTIVE_FIG_W_MM = 240
INTERACTIVE_FIG_H_MM = 400
INTERACTIVE_MARKER_SCALE = 0.09
PANEL_TITLES = ("O atoms", "Ir atoms", "Structures")

DATASETS = [
    ("PBE", TRAJ_PBE),
    ("PBE0", TRAJ_PBE0),
]
DATASET_MARKERS = {"PBE": "o", "PBE0": "s"}
DATASET_NAMES = [label for label, _ in DATASETS]


def _motif_frac_cols(row: dict, prefix: str) -> list[str]:
    return sorted(k for k in row if k.startswith(prefix) and k.endswith("_frac"))


def _composition_rgb(frac: np.ndarray, colors: list) -> np.ndarray:
    """RGB mix of motif colours weighted by per-structure fractions (fig03 style)."""
    frac = np.asarray(frac, dtype=float)
    rgb = np.array([to_rgb(c) for c in colors], dtype=float)
    return frac @ rgb


def _extent_limits(z: np.ndarray, pad_frac: float = 0.04) -> tuple[float, float]:
    z = np.asarray(z, dtype=float)
    if z.size == 0:
        return 0.0, 1.0
    z_min, z_max = float(z.min()), float(z.max())
    if z_max <= z_min:
        return z_min - 0.05, z_max + 0.05
    pad = pad_frac * (z_max - z_min)
    return z_min - pad, z_max + pad


def _scatter_rgb_by_dataset_3d(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    rgb: np.ndarray,
    ds: np.ndarray,
    *,
    sizes: float,
    alpha: float,
) -> None:
    ds = np.asarray(ds, dtype=int)
    for di, name in enumerate(DATASET_NAMES):
        mask = ds == di
        if not mask.any():
            continue
        ax.scatter(
            x[mask], y[mask], z[mask], c=rgb[mask], s=sizes, alpha=alpha,
            marker=DATASET_MARKERS[name],
            edgecolors="none", depthshade=False, zorder=MOMENT_RATIO_ZORDER_DATA,
        )


def _scatter_clusters_by_dataset_3d(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    labels: np.ndarray,
    colors: list,
    ds: np.ndarray,
    *,
    sizes: float,
    alpha: float,
) -> None:
    labels = np.asarray(labels, dtype=int)
    ds = np.asarray(ds, dtype=int)
    for di, name in enumerate(DATASET_NAMES):
        ds_mask = ds == di
        if not ds_mask.any():
            continue
        mk = DATASET_MARKERS[name]
        for c in np.unique(labels[ds_mask]):
            mask = ds_mask & (labels == c)
            if not mask.any():
                continue
            col = colors[int(c) % len(colors)]
            ax.scatter(
                x[mask], y[mask], z[mask], c=[col], s=sizes, alpha=alpha,
                marker=mk, edgecolors="none", depthshade=False,
                zorder=MOMENT_RATIO_ZORDER_DATA,
            )


def _scatter_rgb_by_dataset_2d(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    rgb: np.ndarray,
    ds: np.ndarray,
    *,
    sizes: float,
    alpha: float,
) -> None:
    ds = np.asarray(ds, dtype=int)
    for di, name in enumerate(DATASET_NAMES):
        mask = ds == di
        if not mask.any():
            continue
        ax.scatter(
            x[mask], y[mask], c=rgb[mask], s=sizes, alpha=alpha,
            marker=DATASET_MARKERS[name],
            edgecolors="none", zorder=MOMENT_RATIO_ZORDER_DATA,
        )


def _scatter_clusters_by_dataset_2d(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    labels: np.ndarray,
    colors: list,
    ds: np.ndarray,
    *,
    sizes: float,
    alpha: float,
) -> None:
    labels = np.asarray(labels, dtype=int)
    ds = np.asarray(ds, dtype=int)
    for di, name in enumerate(DATASET_NAMES):
        ds_mask = ds == di
        if not ds_mask.any():
            continue
        mk = DATASET_MARKERS[name]
        for c in np.unique(labels[ds_mask]):
            mask = ds_mask & (labels == c)
            if not mask.any():
                continue
            col = colors[int(c) % len(colors)]
            ax.scatter(
                x[mask], y[mask], c=[col], s=sizes, alpha=alpha,
                marker=mk, edgecolors="none", zorder=MOMENT_RATIO_ZORDER_DATA,
            )


def load_pooled(
    extent_normalization: str = EXTENT_NORMALIZATION,
    extent_blend: float = EXTENT_BLEND,
):
    tp = ThermoPartitionFunctions()
    struct_x, struct_y, struct_z, struct_cluster, struct_ds = [], [], [], [], []
    o_x, o_y, o_z, o_rgb, o_ds = [], [], [], [], []
    ir_x, ir_y, ir_z, ir_rgb, ir_ds = [], [], [], [], []

    for di, (label, traj_path) in enumerate(DATASETS):
        if not os.path.exists(traj_path):
            continue
        structures = list(read(traj_path, ":"))
        csv_path = ACE_FEATURES[label]
        with open(csv_path, newline="") as f:
            rows = list(csv.DictReader(f))

        if not rows:
            continue
        o_cols = _motif_frac_cols(rows[0], "O-")
        ir_cols = _motif_frac_cols(rows[0], "Ir-")
        o_colors = o_motif_colors(len(o_cols))
        ir_colors = ir_motif_colors(len(ir_cols))

        for si, atoms in enumerate(structures):
            if si >= len(rows):
                break
            row = rows[si]
            x_s, y_s, z_s = inertia_shape_extent_features(
                atoms, tp,
                extent_normalization=extent_normalization,
                extent_blend=extent_blend,
            )
            cluster = int(row["cluster"])
            struct_x.append(x_s)
            struct_y.append(y_s)
            struct_z.append(z_s)
            struct_cluster.append(cluster)
            struct_ds.append(di)

            o_frac = np.array([float(row[c]) for c in o_cols], dtype=float)
            ir_frac = np.array([float(row[c]) for c in ir_cols], dtype=float)
            o_color = _composition_rgb(o_frac, o_colors)
            ir_color = _composition_rgb(ir_frac, ir_colors)

            for sym in atoms.get_chemical_symbols():
                if sym == "O":
                    o_x.append(x_s)
                    o_y.append(y_s)
                    o_z.append(z_s)
                    o_rgb.append(o_color)
                    o_ds.append(di)
                elif sym == "Ir":
                    ir_x.append(x_s)
                    ir_y.append(y_s)
                    ir_z.append(z_s)
                    ir_rgb.append(ir_color)
                    ir_ds.append(di)

    return {
        "struct": (
            np.asarray(struct_x), np.asarray(struct_y), np.asarray(struct_z),
            np.asarray(struct_cluster, dtype=int),
            np.asarray(struct_ds, dtype=int),
        ),
        "o": (
            np.asarray(o_x), np.asarray(o_y), np.asarray(o_z),
            np.asarray(o_rgb, dtype=float) if o_rgb else np.empty((0, 3)),
            np.asarray(o_ds, dtype=int) if o_ds else np.empty(0, dtype=int),
        ),
        "ir": (
            np.asarray(ir_x), np.asarray(ir_y), np.asarray(ir_z),
            np.asarray(ir_rgb, dtype=float) if ir_rgb else np.empty((0, 3)),
            np.asarray(ir_ds, dtype=int) if ir_ds else np.empty(0, dtype=int),
        ),
    }


def _subsample_arrays(max_n: int, *arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    """Random subsample aligned rows for faster interactive rendering."""
    n = len(arrays[0])
    if n <= max_n:
        return arrays
    idx = np.random.default_rng(0).choice(n, max_n, replace=False)
    return tuple(np.asarray(a)[idx] for a in arrays)


def _sync_views(source_ax, axes: list, fig) -> None:
    for ax in axes:
        if ax is not source_ax:
            ax.view_init(elev=source_ax.elev, azim=source_ax.azim)
    fig.canvas.draw_idle()


def _attach_interactive_viewer(fig, axes: list, titles: tuple[str, ...]) -> None:
    """Sync rotation across panels; ``p`` prints elev/azim for all three."""
    def _print_views() -> None:
        parts = [
            f"{title}: elev={ax.elev:.1f} azim={ax.azim:.1f}"
            for ax, title in zip(axes, titles)
        ]
        print("[view]", "  |  ".join(parts))
        print("  -> style_inertia_shape_extent_3d(elev=..., azim=...)")

    def on_key(event) -> None:
        if event.key == "p":
            _print_views()

    def on_release(event) -> None:
        if event.inaxes in axes:
            _sync_views(event.inaxes, axes, fig)

    fig.canvas.mpl_connect("key_press_event", on_key)
    for ax in axes:
        fig.canvas.mpl_connect("button_release_event", on_release)

    print(
        "[interactive] Drag to rotate (all panels stay synced). "
        "Press ``p`` to print elev/azim."
    )
    _print_views()


def draw_moment_ratio_inertia_column(
    fig,
    gs,
    data: dict,
    *,
    marker_scale: float = MARKER_SCALE,
    atom_marker_s: float = MARKER_S,
    struct_marker_s: float = MARKER_S,
    atom_alpha: float = ATOM_ALPHA,
    pad_frac: float = 0.02,
    z_lim: tuple[float, float] | None = None,
    elev: float = 28.0,
    azim: float = -67.0,
) -> tuple[list, tuple[float, float]]:
    """Draw O / Ir / structure moment-ratio 3D panels on a 3-row gridspec slice."""
    sx, sy, sz, sc, sd = data["struct"]
    ox, oy, oz, o_rgb, od = data["o"]
    ix, iy, iz, ir_rgb, id_ = data["ir"]
    k_c = int(sc.max()) + 1 if sc.size else 1
    struct_colors = [
        struct_cluster_palette(k_c)(c % struct_cluster_palette(k_c).N) for c in range(k_c)
    ]

    if z_lim is None:
        z_lim = _extent_limits(sz, pad_frac=pad_frac)

    def _draw_panel(ax, x, y, z, scatter_fn, *scatter_args, **scatter_kw) -> None:
        style_inertia_shape_extent_3d(
            ax, z_lim, marker_scale=marker_scale, bare=True, elev=elev, azim=azim,
        )
        scatter_fn(ax, x, y, z, *scatter_args, **scatter_kw)
        overlay_moment_ratio_corner_guides_3d(
            ax, z_lim, pillars=True, corner_markers=False, labels=True, extent_label=True,
        )
        z0, z1 = z_lim
        z_span = max(z1 - z0, 1e-9)
        ax.set_zlim(z0, z1 + 0.07 * z_span)

    ax0 = fig.add_subplot(gs[0], projection="3d")
    _draw_panel(
        ax0, ox, oy, oz, _scatter_rgb_by_dataset_3d, o_rgb, od,
        sizes=atom_marker_s, alpha=atom_alpha,
    )

    ax1 = fig.add_subplot(gs[1], projection="3d")
    _draw_panel(
        ax1, ix, iy, iz, _scatter_rgb_by_dataset_3d, ir_rgb, id_,
        sizes=atom_marker_s, alpha=atom_alpha,
    )

    ax2 = fig.add_subplot(gs[2], projection="3d")
    _draw_panel(
        ax2, sx, sy, sz, _scatter_clusters_by_dataset_3d, sc, struct_colors, sd,
        sizes=struct_marker_s, alpha=0.85,
    )

    return [ax0, ax1, ax2], z_lim


def draw_moment_ratio_floor_column(
    fig,
    gs,
    data: dict,
    *,
    marker_scale: float = FLOOR_MARKER_SCALE,
    atom_marker_s: float = MARKER_S,
    struct_marker_s: float = MARKER_S,
    atom_alpha: float = ATOM_ALPHA,
) -> list:
    """Draw O / Ir / structure floor-triangle panels (S/L/P only, no extent axis)."""
    sx, sy, _, sc, sd = data["struct"]
    ox, oy, _, o_rgb, od = data["o"]
    ix, iy, _, ir_rgb, id_ = data["ir"]
    k_c = int(sc.max()) + 1 if sc.size else 1
    struct_colors = [
        struct_cluster_palette(k_c)(c % struct_cluster_palette(k_c).N) for c in range(k_c)
    ]

    def _draw_panel(ax, x, y, scatter_fn, *scatter_args, **scatter_kw) -> None:
        style_bare_inertia_triangle(ax, marker_scale=marker_scale)
        scatter_fn(ax, x, y, *scatter_args, **scatter_kw)

    ax0 = fig.add_subplot(gs[0])
    _draw_panel(
        ax0, ox, oy, _scatter_rgb_by_dataset_2d, o_rgb, od,
        sizes=atom_marker_s, alpha=atom_alpha,
    )

    ax1 = fig.add_subplot(gs[1])
    _draw_panel(
        ax1, ix, iy, _scatter_rgb_by_dataset_2d, ir_rgb, id_,
        sizes=atom_marker_s, alpha=atom_alpha,
    )

    ax2 = fig.add_subplot(gs[2])
    _draw_panel(
        ax2, sx, sy, _scatter_clusters_by_dataset_2d, sc, struct_colors, sd,
        sizes=struct_marker_s, alpha=0.85,
    )

    return [ax0, ax1, ax2]


def adjust_moment_ratio_panel_positions(
    axes,
    *,
    left_expand: float = MOMENT_RATIO_AXES_LEFT_EXPAND,
    right_expand: float = MOMENT_RATIO_AXES_RIGHT_EXPAND,
) -> None:
    """Expand moment-ratio 3D axes into side margins to trim in-panel whitespace."""
    for ax in axes:
        pos = ax.get_position()
        dl = pos.width * left_expand
        dr = pos.width * right_expand
        ax.set_position([pos.x0 - dl, pos.y0, pos.width + dl + dr, pos.height])


def _clear_axes_background(axes) -> None:
    """Disable opaque axes patches so packed rows cannot paint white over neighbours."""
    for ax in axes:
        ax.set_facecolor("none")
        ax.patch.set_alpha(0.0)


def build_figure(data, *, interactive: bool = False):
    if interactive:
        ox, oy, oz, o_rgb, od = data["o"]
        ix, iy, iz, ir_rgb, id_ = data["ir"]
        sx, sy, sz, sc, sd = data["struct"]
        ox, oy, oz, o_rgb, od = _subsample_arrays(
            INTERACTIVE_MAX_POINTS, ox, oy, oz, o_rgb, od,
        )
        ix, iy, iz, ir_rgb, id_ = _subsample_arrays(
            INTERACTIVE_MAX_POINTS, ix, iy, iz, ir_rgb, id_,
        )
        sx, sy, sz, sc, sd = _subsample_arrays(
            INTERACTIVE_MAX_POINTS, sx, sy, sz, sc, sd,
        )
        data = {
            "struct": (sx, sy, sz, sc, sd),
            "o": (ox, oy, oz, o_rgb, od),
            "ir": (ix, iy, iz, ir_rgb, id_),
        }

    sx, sy, sz, sc, sd = data["struct"]
    ox, oy, oz, o_rgb, od = data["o"]
    ix, iy, iz, ir_rgb, id_ = data["ir"]
    k_c = int(sc.max()) + 1 if sc.size else 1

    fig_w = INTERACTIVE_FIG_W_MM if interactive else FIG_W_MM
    fig_h = INTERACTIVE_FIG_H_MM if interactive else FIG_H_MM
    fig_dpi = INTERACTIVE_DPI if interactive else DPI
    atom_marker_s = INTERACTIVE_MARKER_S if interactive else MARKER_S
    struct_marker_s = INTERACTIVE_STRUCT_MARKER_S if interactive else MARKER_S
    marker_scale = INTERACTIVE_MARKER_SCALE if interactive else MARKER_SCALE

    fig = plt.figure(figsize=(fig_w / 25.4, fig_h / 25.4), dpi=fig_dpi)
    plt.rcParams["svg.fonttype"] = "none"

    # 1×2 outer grid so column subgridspecs own the full height (no outer row gaps).
    outer = fig.add_gridspec(
        1, 2,
        left=PANEL_GRID_LEFT if not interactive else 0.0,
        right=PANEL_GRID_RIGHT if not interactive else 1.0,
        top=1.0, bottom=0.0,
        wspace=COL_WSPACE if not interactive else 0.08,
    )
    gs_3d = outer[0, 0].subgridspec(
        3, 1,
        hspace=PANEL_HSPACE if not interactive else 0.04,
    )
    gs_2d = outer[0, 1].subgridspec(
        3, 1,
        hspace=PANEL_HSPACE_2D if not interactive else 0.04,
    )

    axes_3d, z_lim = draw_moment_ratio_inertia_column(
        fig, gs_3d, data,
        marker_scale=marker_scale,
        atom_marker_s=atom_marker_s,
        struct_marker_s=struct_marker_s,
        azim=MOMENT_RATIO_EMBED_AZIM,
    )
    axes_2d = draw_moment_ratio_floor_column(
        fig, gs_2d, data,
        marker_scale=FLOOR_MARKER_SCALE,
        atom_marker_s=atom_marker_s,
        struct_marker_s=struct_marker_s,
    )
    _clear_axes_background(axes_3d + axes_2d)

    if not interactive:
        adjust_moment_ratio_panel_positions(
            axes_3d,
            left_expand=MOMENT_RATIO_EMBED_LEFT_EXPAND,
            right_expand=MOMENT_RATIO_EMBED_RIGHT_EXPAND,
        )
        adjust_moment_ratio_panel_positions(
            axes_2d,
            left_expand=MOMENT_RATIO_AXES_LEFT_EXPAND,
            right_expand=MOMENT_RATIO_AXES_RIGHT_EXPAND,
        )

    return fig, axes_3d, axes_2d, z_lim, (sx.size, ox.size, ix.size, k_c)


def _use_interactive_backend() -> str:
    """Pick a GUI backend for ``plt.show()`` (first match wins)."""
    import matplotlib

    for backend in ("Qt5Agg", "QtAgg", "GTK3Agg", "TkAgg", "WXAgg"):
        try:
            matplotlib.use(backend, force=True)
            return backend
        except ImportError:
            continue
    raise RuntimeError(
        "No interactive matplotlib backend available. "
        "Install PyQt5 or python3-tk, then rerun with --interactive."
    )


def main(*, interactive: bool = False) -> None:
    if interactive:
        backend = _use_interactive_backend()
        print(f"[interactive] matplotlib backend: {backend}")

    ensure_for_supplementary(["supp_moments_of_inertia"])
    ensure_dirs(OUTPUT_DIR)

    data = load_pooled()
    fig, axes_3d, axes_2d, z_lim, counts = build_figure(data, interactive=interactive)
    n_struct, n_o, n_ir, k_c = counts

    if interactive:
        _attach_interactive_viewer(fig, axes_3d, PANEL_TITLES)
        plt.show()
        return

    # Fixed canvas (180 mm wide): do not use bbox_inches='tight'.
    save_fig(fig, OUTPUT_PATH, dpi=DPI, bbox_inches=None, pad_inches=0)
    print(f"\n[supp] {OUTPUT_PATH}")
    print(f"  canvas={FIG_W_MM}×{FIG_H_MM} mm")
    print(f"  structures={n_struct}  O atoms={n_o}  Ir atoms={n_ir}  clusters={k_c}")
    print(f"  extent range: [{z_lim[0]:.3f}, {z_lim[1]:.3f}]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Supplementary inertia shape + extent figure")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Open interactive 3D viewer (rotate panels; press p to print elev/azim)",
    )
    cli = parser.parse_args()
    main(interactive=cli.interactive)
