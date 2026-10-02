"""Figure 2 – Combined motif clustering (figures 2 + 4 + 3).

Layout (180 × 164 mm, fixed canvas):
  Top — UMAP/PCA | PBE/PBE0 populations | principal moment ratios (3×1 3D, as supp_moments_of_inertia)
  Bottom — ASE centroid strip

Output: ``figures/outputs/fig02_motif_clustering_panel.svg``
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from ase.io import read
from ase.visualize.plot import plot_atoms
from matplotlib.lines import Line2D

from aist.ace.common import ace_atom_display_label, discrete_palette, struct_cluster_palette
from aist.ace.datasets import load_dataset
from aist.ace.io import atom_fit_from_npz, atom_rgbs_for_motif_tile, load_motif_atoms_json
from aist.ace.plotting import atom_population_stack_vs_oxidation, struct_population_stack_vs_oxidation
from aist.thermo import ThermoPartitionFunctions

from figures.config import (
    ACE_ATOM_IR_NPZ,
    ACE_ATOM_O_NPZ,
    ACE_CENTROIDS,
    ACE_FEATURES,
    ACE_STRUCT_NPZ,
    DPI,
    IR_CMAP,
    MOTIF_JSON,
    O_CMAP,
    OUTPUT_DIR,
    TRAJ_PBE,
    TRAJ_PBE0,
)
from figures.fig01_minima import MM, ROTATION, _half_span
from figures.supp_moments_of_inertia import (
    ATOM_ALPHA as MOMENT_RATIO_ATOM_ALPHA,
    MOMENT_RATIO_EMBED_AZIM,
    draw_moment_ratio_inertia_column,
    load_pooled as load_moment_ratio_inertia_data,
)
from figures.utils.cache import ensure_for_figures
from figures.utils.io import ensure_dirs, load_npz
from figures.utils.plotting import (
    AXIS_LABEL_FONTSIZE,
    LEGEND_FONTSIZE,
    TICK_FONTSIZE,
    TITLE_FONTSIZE,
    draw_population_stack,
)
from figures.utils.inertia_triangle import MOMENT_RATIO_EXTENT_BOX_SCALE

OUTPUT_PATH = os.path.join(OUTPUT_DIR, "fig02_motif_clustering_panel.svg")

# --- Canvas (mm); FIG_H = TOP block + centroid strip ---
FIG_W_MM = 180
ROW_H_MM = 43
TOP_H_MM = 3 * ROW_H_MM
BOTTOM_STRIP_MM = 35
FIG_H_MM = TOP_H_MM + BOTTOM_STRIP_MM
BOTTOM_H_MM = BOTTOM_STRIP_MM

# --- Figure margins & vertical gaps ---
SUBPLOT_LEFT = 0.036
SUBPLOT_RIGHT = 0.975
SUBPLOT_TOP = 0.955
SUBPLOT_BOTTOM = 0.030
TOP_BOTTOM_GAP = 0.048
TOP_ROW_HSPACE = 0.12
TOP_COL_WSPACE = 0.02
GROUP_COL_GAP = 0.03

# --- Scatter / pop square panels (moment-ratio panel size is independent) ---
SCATTER_LEFT_SHIFT = 0.0085
SCATTER_PBE_EDGE_GAP = 0.082
PBE_PBE0_EDGE_GAP = 0.028
PANEL_SQUARE_FRAC = 1.0
PANEL_SIDE_SCALE = 0.78
POP_AXES_LEFT_PAD = 0.035
POP_YLABEL_COORDS = (-0.22, 0.5)

# --- Moment-ratio column ---
MOMENT_RATIO_POP_GAP = 0.002
MOMENT_RATIO_RIGHT_ANCHOR = SUBPLOT_RIGHT
MOMENT_RATIO_HEIGHT_FRAC = 0.95
MOMENT_RATIO_TRIANGLE_WIDTH_ASPECT = 1.42
MOMENT_RATIO_LEFT_OVERLAP = 0.065
MOMENT_RATIO_FLOOR_X_STRETCH = 1.55
MOMENT_RATIO_FLOOR_Y_STRETCH = 1.55
MOMENT_RATIO_MARKER_SCALE = 0.092
MOMENT_RATIO_MARKER_S = 7.0
MOMENT_RATIO_ZOOM_LEFT = 0.58
MOMENT_RATIO_ZOOM_RIGHT = 0.26
MOMENT_RATIO_ZOOM_TOP = 0.42
MOMENT_RATIO_ZOOM_BOT = 0.42

# --- Bottom centroid strip ---
BOTTOM_STRIP_IMG_FRAC = 0.92
BOTTOM_STRIP_HSPACE = 0.02
BOTTOM_STRIP_WSPACE = 0.10
BOTTOM_LABEL_GAP_FRAC = 0.105
BOTTOM_LABEL_ROW_FRAC = 0.055
CENTROID_OUTLINE_WIDTH = 0.12
CENTROID_HALF_SPAN_ZOOM = 0.96
CENTROID_TILE_SCALE = 1.35
TAG_PAD_PT = 2
TAG_SQ_PT = 5

# --- Plot content ---
N_BINS = 100
SMOOTH_SIGMA = 1.0
POP_YLABEL = "pop %"
POP_OX_XMAX = 3.3  # n_O/n_Ir upper limit on O | Ir | struct population stacks
SCATTER_ROW_YLABELS = ("O atoms", "Ir atoms", "Structures")
SCATTER_COL_TITLE = "UMAP/PCA"
UMAP_LEGEND_PAD_FRAC = 0.04
UMAP_LEGEND_MARKERSIZE = 3
UMAP_LEGEND_MARKERSCALE = 1.05

plt.rcParams["svg.fonttype"] = "none"


def _orient_centroid_atoms(atoms, tp: ThermoPartitionFunctions) -> Atoms:
    """Like ``orient_by_inertia_axes``, but swap the two smallest principal axes."""
    atoms_oriented = atoms.copy()
    atoms_oriented.set_positions(
        atoms_oriented.get_positions() - atoms_oriented.get_center_of_mass()
    )
    inertia = tp.get_inertia_tensor(atoms_oriented)
    _, eigvecs = np.linalg.eigh(inertia)
    # fig01: I1→x, I2→y; swapped: I2→x, I1→y, I3→z
    basis = np.column_stack((eigvecs[:, 1], eigvecs[:, 0], eigvecs[:, 2]))
    atoms_oriented.set_positions(atoms_oriented.get_positions() @ basis)
    return atoms_oriented


def _load_centroids(traj_path: str) -> list[dict]:
    """Orient for centroid strip (fig01 recipe with I1/I2 swap)."""
    tp = ThermoPartitionFunctions()
    structures = list(read(traj_path, ":"))
    out = []
    for atoms in structures:
        info = atoms.info
        frame_idx = int(info.get("frame_index", len(out)))
        cluster = int(info.get("cluster", 0))
        oriented = _orient_centroid_atoms(atoms, tp)
        out.append({
            "atoms": oriented,
            "frame_index": frame_idx,
            "cluster": cluster,
        })
    out.sort(key=lambda e: e["cluster"])
    return out


def _motif_labels(motif_json: dict, frame_idx: int, n_atoms: int) -> list[str]:
    labels = motif_json.get("frames", {}).get(str(frame_idx))
    if labels is None or len(labels) != n_atoms:
        return ["_"] * n_atoms
    return labels


def _style_atom_plot(ax, atoms, colors) -> None:
    plot_atoms(
        atoms, ax=ax, rotation=ROTATION, radii=None,
        colors=colors, show_unit_cell=0,
    )
    for col in ax.collections:
        col.set_clip_on(False)
        try:
            col.set_linewidths(CENTROID_OUTLINE_WIDTH)
        except Exception:
            pass
    for patch in ax.patches:
        patch.set_clip_on(False)
        try:
            patch.set_linewidth(CENTROID_OUTLINE_WIDTH)
        except Exception:
            pass


def _draw_cluster_tag(ax, cluster: int, struct_cmap) -> None:
    from matplotlib.offsetbox import AnchoredOffsetbox, DrawingArea, HPacker, TextArea
    from matplotlib.patches import Rectangle

    color = struct_cmap(cluster % struct_cmap.N)
    sq = TAG_SQ_PT
    square = DrawingArea(sq, sq, 0, 0)
    square.add_artist(
        Rectangle((0, 0), sq, sq, facecolor=color, edgecolor="black", linewidth=0.4),
    )
    label = TextArea(f"C{cluster}", textprops={"fontsize": AXIS_LABEL_FONTSIZE, "color": "black"})
    box = HPacker(children=[square, label], align="center", pad=0, sep=TAG_PAD_PT)
    ax.set_axis_off()
    ax.add_artist(
        AnchoredOffsetbox(loc="center", pad=0, borderpad=0, child=box, frameon=False),
    )


def _layout_square_clustering_panels(axes: np.ndarray) -> None:
    """Scatter + PBE + PBE0: equal squares with explicit edge gaps (not grid-cell centering)."""
    for row in range(3):
        scatter_ax = axes[row, 0]
        pbe_ax = axes[row, 1]
        pbe0_ax = axes[row, 2]
        sp = scatter_ax.get_position()
        pp = pbe_ax.get_position()
        p0 = pbe0_ax.get_position()
        side_base = min(
            pp.height * PANEL_SQUARE_FRAC,
            sp.width,
            pp.width,
            p0.width,
        )
        side = side_base * PANEL_SIDE_SCALE
        cy = pp.y0 + 0.5 * pp.height
        y0 = cy - 0.5 * side
        scatter_left = sp.x0 + 0.5 * sp.width - 0.5 * side - SCATTER_LEFT_SHIFT
        scatter_right = scatter_left + side
        pbe_left = scatter_right + SCATTER_PBE_EDGE_GAP
        pbe_right = pbe_left + side
        pbe0_left = pbe_right + PBE_PBE0_EDGE_GAP
        scatter_ax.set_position([scatter_left, y0, side, side])
        pbe_ax.set_position([pbe_left, y0, side, side])
        pbe0_ax.set_position([pbe0_left, y0, side, side])


def _layout_moment_ratio_in_rows(
    axes: np.ndarray,
    *,
    row_heights: list[float] | None = None,
    right: float = MOMENT_RATIO_RIGHT_ANCHOR,
    gap: float = MOMENT_RATIO_POP_GAP,
) -> None:
    """Place moment-ratio panels in the right column, anchored to the right edge."""
    right = min(right, SUBPLOT_RIGHT)
    for row in range(3):
        pbe0 = axes[row, 2]
        moment_ratio_ax = axes[row, 3]
        p0 = pbe0.get_position()
        row_h = row_heights[row] if row_heights is not None else p0.height
        h = row_h * MOMENT_RATIO_HEIGHT_FRAC
        x_left_limit = p0.x1 + gap - MOMENT_RATIO_LEFT_OVERLAP
        w = min(h * MOMENT_RATIO_TRIANGLE_WIDTH_ASPECT, right - x_left_limit)
        w = max(w, 0.01)
        x0 = right - w
        cy = p0.y0 + 0.5 * p0.height
        moment_ratio_ax.set_position([x0, cy - 0.5 * h, w, h])


def _zoom_moment_ratio_panels(
    axes: np.ndarray,
    *,
    left: float = MOMENT_RATIO_ZOOM_LEFT,
    right: float = MOMENT_RATIO_ZOOM_RIGHT,
    top: float = MOMENT_RATIO_ZOOM_TOP,
    bot: float = MOMENT_RATIO_ZOOM_BOT,
) -> None:
    """Expand moment-ratio 3D axes beyond their cell on all sides.

    Because the 3D renderer reserves internal padding, making the *axes box*
    larger than the visible cell effectively zooms the projection in on the
    data triangle — the same trick as PANEL_GRID_LEFT in the supplementary.
    """
    for row in range(3):
        ax = axes[row, 3]
        pos = ax.get_position()
        dl = pos.width  * left
        dr = pos.width  * right
        dt = pos.height * top
        db = pos.height * bot
        ax.set_position([
            pos.x0 - dl,
            pos.y0 - db,
            pos.width  + dl + dr,
            pos.height + dt + db,
        ])


def _anchor_moment_ratio_right(
    axes: np.ndarray,
    *,
    right: float = MOMENT_RATIO_RIGHT_ANCHOR,
) -> None:
    """Keep moment-ratio right edge fixed fixed after zoom expansion (panels sit further right)."""
    right = min(right, SUBPLOT_RIGHT)
    for row in range(3):
        ax = axes[row, 3]
        pos = ax.get_position()
        ax.set_position([right - pos.width, pos.y0, pos.width, pos.height])


def _stack_axes_above_moment_ratio(axes: np.ndarray) -> None:
    """Keep moment-ratio panels under scatter/pop so overlap hides them, not the other panels."""
    for row in range(3):
        axes[row, 3].set_zorder(0)
        axes[row, 3].patch.set_visible(False)
        for col in range(3):
            axes[row, col].set_zorder(2)


def _stretch_moment_ratio_floor_aspect(axes: np.ndarray) -> None:
    """Tune moment-ratio floor triangle aspect in 3D (x / y stretch via set_box_aspect)."""
    x_st = MOMENT_RATIO_FLOOR_X_STRETCH
    y_st = MOMENT_RATIO_FLOOR_Y_STRETCH
    if x_st == 1.0 and y_st == 1.0:
        return
    for row in range(3):
        ax = axes[row, 3]
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        zlim = ax.get_zlim()
        x_span = xlim[1] - xlim[0]
        y_span = ylim[1] - ylim[0]
        z0 = float(zlim[0])
        z_span = max(float(zlim[1]) - z0, 1e-9)
        try:
            ax.set_box_aspect((
                x_span * x_st,
                y_span * y_st,
                MOMENT_RATIO_EXTENT_BOX_SCALE * z_span,
            ))
        except AttributeError:
            pass


def _draw_moment_ratio_column(
    fig, moment_ratio_gs, moment_ratio_data: dict, axes: np.ndarray,
) -> None:
    """Draw moment-ratio column after scatter/pop layout (last draw, lowest z-order)."""
    moment_ratio_axes, _ = draw_moment_ratio_inertia_column(
        fig, moment_ratio_gs, moment_ratio_data,
        marker_scale=MOMENT_RATIO_MARKER_SCALE,
        atom_marker_s=MOMENT_RATIO_MARKER_S,
        struct_marker_s=MOMENT_RATIO_MARKER_S,
        atom_alpha=MOMENT_RATIO_ATOM_ALPHA,
        azim=MOMENT_RATIO_EMBED_AZIM,
    )
    for row, ax in enumerate(moment_ratio_axes):
        axes[row, 3] = ax

    for row in range(3):
        axes[row, 3].set_title("")
        axes[row, 3].set_ylabel("")
        axes[row, 3].tick_params(labelbottom=False, labelleft=False)
        axes[row, 3].set_xlabel("")


def _nudge_pop_axes_right(axes: np.ndarray, pad_frac: float) -> None:
    """Reserve left margin in pop panels for the shared y-axis label."""
    for row in range(3):
        for col in (1, 2):
            ax = axes[row, col]
            pos = ax.get_position()
            pad = pos.width * pad_frac
            ax.set_position([pos.x0 + pad, pos.y0, pos.width - pad, pos.height])


def _layout_pop_ylabel(axes: np.ndarray) -> None:
    for row in range(3):
        ax = axes[row, 1]
        ax.set_ylabel(POP_YLABEL, fontsize=AXIS_LABEL_FONTSIZE)
        ax.yaxis.set_label_coords(*POP_YLABEL_COORDS)


def _scale_tile_axes(axes, scale: float) -> None:
    if scale == 1.0:
        return
    for ax in axes.ravel():
        if ax is None:
            continue
        pos = ax.get_position()
        cx = pos.x0 + 0.5 * pos.width
        cy = pos.y0 + 0.5 * pos.height
        nw = pos.width * scale
        nh = pos.height * scale
        ax.set_position([cx - 0.5 * nw, cy - 0.5 * nh, nw, nh])


def _layout_centroid_labels_below_images(
    label_axes: list,
    struct_axes: list,
) -> None:
    """Snap cluster-tag axes just under centroid images (after tile scaling)."""
    for ax_img, ax_lbl in zip(struct_axes, label_axes, strict=True):
        ip = ax_img.get_position()
        lbl_h = ip.height * BOTTOM_LABEL_ROW_FRAC
        gap = ip.height * BOTTOM_LABEL_GAP_FRAC
        ax_lbl.set_position([ip.x0, ip.y0 - gap - lbl_h, ip.width, lbl_h])


def _draw_horizontal_centroid_strip(
    fig,
    bottom_gs,
    entries: list[dict],
    motif_json: dict,
    struct_cmap,
    shared_half_span: float,
) -> tuple[list, list]:
    n = len(entries)
    label_axes = []
    struct_axes = []
    for i, entry in enumerate(entries):
        ax_img = fig.add_subplot(bottom_gs[0, i])
        ax_lbl = fig.add_subplot(bottom_gs[1, i])
        label_axes.append(ax_lbl)
        struct_axes.append(ax_img)

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

    for i, entry in enumerate(entries):
        _draw_cluster_tag(label_axes[i], int(entry["cluster"]), struct_cmap)

    return struct_axes, label_axes


def _draw_compact_atom_scatter(ax, fit, species: str, cmap_name: str) -> None:
    k = int(fit["k"])
    Xp = np.asarray(fit["X_plot"])
    labels = np.asarray(fit["labels"], dtype=int)
    colors = discrete_palette(cmap_name, k) if k > 0 else []
    for c in range(k):
        mask = labels == c
        if not mask.any():
            continue
        ax.scatter(
            Xp[mask, 0], Xp[mask, 1],
            c=[colors[c]], s=5, alpha=0.65, edgecolors="none",
            label=ace_atom_display_label(species, c),
        )
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)


def _finalize_atom_umap_legends(fig, axes: np.ndarray, ir_fit, o_fit) -> None:
    """O / Ir UMAP legends — anchored to panel bottom-left in figure coordinates."""
    for row, fit, species, cmap_name in (
        (0, o_fit, "O", O_CMAP),
        (1, ir_fit, "Ir", IR_CMAP),
    ):
        ax = axes[row, 0]
        old = ax.get_legend()
        if old is not None:
            old.remove()
        k = int(fit["k"])
        colors = discrete_palette(cmap_name, k)
        handles = [
            Line2D(
                [0], [0], linestyle="", marker="o", markersize=UMAP_LEGEND_MARKERSIZE,
                markerfacecolor=colors[c], markeredgecolor="none",
                alpha=1.0, label=ace_atom_display_label(species, c),
            )
            for c in range(k)
        ]
        pos = ax.get_position()
        pad_x = UMAP_LEGEND_PAD_FRAC * pos.width
        pad_y = UMAP_LEGEND_PAD_FRAC * pos.height
        leg = ax.legend(
            handles=handles,
            fontsize=LEGEND_FONTSIZE,
            frameon=False,
            loc="lower left",
            bbox_to_anchor=(pos.x0 + pad_x, pos.y0 + pad_y),
            bbox_transform=fig.transFigure,
            markerscale=UMAP_LEGEND_MARKERSCALE,
            ncol=2,
            borderaxespad=0,
            handlelength=1.0,
            handletextpad=0.35,
            labelspacing=0.25,
        )
        leg.set_in_layout(False)
        leg.set_zorder(20)
        leg.set_clip_on(False)


def _draw_compact_struct_pca(ax, struct_cache: dict) -> None:
    k = int(struct_cache["k"])
    Xp = np.asarray(struct_cache["X_plot"])
    labels = np.asarray(struct_cache["labels"], dtype=int)
    ds_idx = np.asarray(struct_cache["dataset_index"], dtype=int)
    ds_names = [str(x) for x in np.asarray(struct_cache["dataset_names"]).tolist()]
    cmap = struct_cluster_palette(max(k, 1))
    markers = {"PBE": "o", "PBE0": "s"}

    for di, name in enumerate(ds_names):
        sel = ds_idx == di
        if not sel.any():
            continue
        mk = markers.get(name, "o")
        for c in range(k):
            mask = sel & (labels == c)
            if not mask.any():
                continue
            ax.scatter(
                Xp[mask, 0], Xp[mask, 1],
                c=[cmap(c % cmap.N)], s=16, alpha=0.75,
                marker=mk, edgecolors="none",
            )

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)


def _style_scatter_column(axes: np.ndarray) -> None:
    axes[0, 0].set_title(SCATTER_COL_TITLE, fontsize=TITLE_FONTSIZE)
    for row in range(3):
        ax = axes[row, 0]
        ax.set_ylabel(SCATTER_ROW_YLABELS[row], fontsize=AXIS_LABEL_FONTSIZE)
        ax.tick_params(labelbottom=False, labelleft=False)
        ax.set_xlabel("")
        if row > 0:
            ax.set_title("")


def _draw_atom_pop_panel(
    ax,
    d,
    species: str,
    k: int,
    *,
    show_xlabel: bool,
    show_ylabel: bool,
    show_legend: bool,
    title: str = "",
    ylabel: str = "",
) -> None:
    key = "per_struct_ir_cluster" if species == "Ir" else "per_struct_o_cluster"
    cmap_name = IR_CMAP if species == "Ir" else O_CMAP
    x_grid, Y = atom_population_stack_vs_oxidation(
        d, k, key, n_bins=N_BINS, smooth_sigma=SMOOTH_SIGMA,
    )
    colors = discrete_palette(cmap_name, k)
    labels = [ace_atom_display_label(species, c) for c in range(k)]
    draw_population_stack(
        ax, x_grid, Y, colors,
        labels=labels if show_legend else None,
        xlabel=r"$n_O / n_{Ir}$" if show_xlabel else "",
        ylabel=ylabel if show_ylabel else "",
        title=title,
        legend_ncol=2,
    )
    if not show_xlabel:
        ax.tick_params(labelbottom=False)
    if not show_ylabel:
        ax.set_ylabel("")
        ax.tick_params(labelleft=False)


def _style_pop_pair_axes(axes: np.ndarray, row: int) -> None:
    """PBE | PBE0 columns share axes; labels and title on the left panel only."""
    ax_pbe, ax_pbe0 = axes[row, 1], axes[row, 2]
    ax_pbe0.tick_params(labelleft=False)
    ax_pbe0.set_ylabel("")
    if row < 2:
        ax_pbe0.tick_params(labelbottom=False)
        ax_pbe0.set_xlabel("")
    else:
        # bottom row: show x ticks and label on PBE0 too
        ax_pbe0.set_xlabel(r"$n_O / n_{Ir}$", fontsize=AXIS_LABEL_FONTSIZE)
        ax_pbe0.tick_params(labelbottom=True)


def _set_pop_oxidation_xlim(axes: np.ndarray, xmax: float = POP_OX_XMAX) -> None:
    """Cap population-vs-oxidation stacks at ``xmax`` (shared-x pairs per row)."""
    for row in range(3):
        ax = axes[row, 1]
        xmin = float(ax.get_xlim()[0])
        ax.set_xlim(xmin, xmax)


def _style_top_grid_axes(axes: np.ndarray) -> None:
    _style_scatter_column(axes)

    for row in range(3):
        axes[row, 1].set_title("PBE" if row == 0 else "", fontsize=TITLE_FONTSIZE)
        axes[row, 2].set_title("PBE0" if row == 0 else "", fontsize=TITLE_FONTSIZE)

    for row in range(3):
        _style_pop_pair_axes(axes, row)

    _set_pop_oxidation_xlim(axes)


def _hide_pop_legends(axes: np.ndarray) -> None:
    for row in range(3):
        for col in (1, 2):
            leg = axes[row, col].get_legend()
            if leg is not None:
                leg.set_visible(False)


def _draw_top_grid(
    fig, scatter_gs, pop_gs, datasets, ir_fit, o_fit, struct_cache, k, cmap,
) -> np.ndarray:
    k_Ir = int(ir_fit["k"])
    k_O = int(o_fit["k"])
    axes = np.empty((3, 4), dtype=object)

    axes[0, 0] = fig.add_subplot(scatter_gs[0, 0])
    _draw_compact_atom_scatter(axes[0, 0], o_fit, "O", O_CMAP)
    axes[1, 0] = fig.add_subplot(scatter_gs[1, 0])
    _draw_compact_atom_scatter(axes[1, 0], ir_fit, "Ir", IR_CMAP)

    for col_idx, d in enumerate(datasets[:2]):
        col = col_idx + 1
        if col == 1:
            axes[0, col] = fig.add_subplot(pop_gs[0, col_idx])
        else:
            axes[0, col] = fig.add_subplot(
                pop_gs[0, col_idx], sharex=axes[0, 1], sharey=axes[0, 1],
            )
        _draw_atom_pop_panel(
            axes[0, col], d, "O", k_O,
            show_xlabel=False,
            show_ylabel=(col == 1),
            show_legend=False,
            title="",
            ylabel=POP_YLABEL if col == 1 else "",
        )
        if col == 1:
            axes[1, col] = fig.add_subplot(pop_gs[1, col_idx])
        else:
            axes[1, col] = fig.add_subplot(
                pop_gs[1, col_idx], sharex=axes[1, 1], sharey=axes[1, 1],
            )
        _draw_atom_pop_panel(
            axes[1, col], d, "Ir", k_Ir,
            show_xlabel=False,
            show_ylabel=(col == 1),
            show_legend=False,
            title="",
            ylabel=POP_YLABEL if col == 1 else "",
        )

    axes[2, 0] = fig.add_subplot(scatter_gs[2, 0])
    _draw_compact_struct_pca(axes[2, 0], struct_cache)

    for col_idx, d in enumerate(datasets[:2]):
        col = col_idx + 1
        if col == 1:
            axes[2, col] = fig.add_subplot(pop_gs[2, col_idx])
        else:
            axes[2, col] = fig.add_subplot(
                pop_gs[2, col_idx], sharex=axes[2, 1], sharey=axes[2, 1],
            )
        x_grid, Y = struct_population_stack_vs_oxidation(
            d, d["struct_labels"], k,
            n_bins=N_BINS, smooth_sigma=SMOOTH_SIGMA,
        )
        draw_population_stack(
            axes[2, col], x_grid, Y, [cmap(c % cmap.N) for c in range(k)],
            labels=None,
            xlabel=r"$n_O / n_{Ir}$" if col == 1 else "",
            ylabel=POP_YLABEL if col == 1 else "",
            title="",
            legend_ncol=2,
        )
        axes[2, col].tick_params(axis="both", labelsize=TICK_FONTSIZE)

    _style_top_grid_axes(axes)
    _hide_pop_legends(axes)
    return axes


def _layout_moment_ratio_panels(
    fig,
    moment_ratio_gs,
    moment_ratio_data: dict,
    axes: np.ndarray,
    pop_row_heights: list[float],
) -> None:
    """Draw moment-ratio column and run post-draw layout (size uses pop_row_heights, not square panels)."""
    _draw_moment_ratio_column(fig, moment_ratio_gs, moment_ratio_data, axes)
    _layout_moment_ratio_in_rows(axes, row_heights=pop_row_heights)
    _zoom_moment_ratio_panels(axes)
    _anchor_moment_ratio_right(axes)
    _stretch_moment_ratio_floor_aspect(axes)
    _stack_axes_above_moment_ratio(axes)


def _apply_panel_layout(
    fig,
    top_axes: np.ndarray,
    moment_ratio_gs,
    moment_ratio_data: dict,
    struct_axes: list,
    label_axes: list,
    ir_fit,
    o_fit,
) -> None:
    """Post-draw geometry: square top panels, moment-ratio column, centroid labels, UMAP legends."""
    fig.subplots_adjust(
        left=SUBPLOT_LEFT, right=SUBPLOT_RIGHT,
        top=SUBPLOT_TOP, bottom=SUBPLOT_BOTTOM,
    )
    fig.canvas.draw()
    _nudge_pop_axes_right(top_axes, POP_AXES_LEFT_PAD)
    pop_row_heights = [top_axes[r, 1].get_position().height for r in range(3)]
    _layout_square_clustering_panels(top_axes)
    _layout_pop_ylabel(top_axes)
    _layout_moment_ratio_panels(fig, moment_ratio_gs, moment_ratio_data, top_axes, pop_row_heights)
    fig.canvas.draw()
    _scale_tile_axes(np.array(struct_axes, dtype=object), CENTROID_TILE_SCALE)
    _layout_centroid_labels_below_images(label_axes, struct_axes)
    _finalize_atom_umap_legends(fig, top_axes, ir_fit, o_fit)
    fig.canvas.draw()


def main() -> None:
    ensure_for_figures([2])
    ensure_dirs(OUTPUT_DIR)

    ir_fit = atom_fit_from_npz(ACE_ATOM_IR_NPZ)
    o_fit = atom_fit_from_npz(ACE_ATOM_O_NPZ)
    struct_cache = load_npz(ACE_STRUCT_NPZ)
    moment_ratio_data = load_moment_ratio_inertia_data()

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

    centroid_traj = ACE_CENTROIDS["pooled"]
    motif_json_path = MOTIF_JSON["pooled"]
    if not os.path.exists(centroid_traj):
        raise SystemExit(f"[fig02] missing centroid traj: {centroid_traj}")
    if not os.path.exists(motif_json_path):
        raise SystemExit(f"[fig02] missing motif JSON: {motif_json_path}")

    entries = _load_centroids(centroid_traj)
    motif_json = load_motif_atoms_json(motif_json_path)
    k = int(struct_cache["k"])
    struct_cmap = struct_cluster_palette(k)

    combined = [(0, 0, e["atoms"]) for e in entries]
    shared_half_span = _half_span(combined) * CENTROID_HALF_SPAN_ZOOM
    print(f"[fig02] centroid half_span (I1/I2 swap, fig01 view): {shared_half_span:.4f}")

    n_tiles = len(entries)
    fig = plt.figure(figsize=(FIG_W_MM * MM, FIG_H_MM * MM), dpi=DPI)
    main_gs = fig.add_gridspec(
        2, 1,
        height_ratios=[TOP_H_MM, BOTTOM_H_MM],
        hspace=TOP_BOTTOM_GAP,
    )
    top_outer = main_gs[0].subgridspec(
        1, 3,
        width_ratios=[1.1, 2.2, 1.0],
        wspace=GROUP_COL_GAP,
    )
    scatter_gs = top_outer[0].subgridspec(3, 1, hspace=TOP_ROW_HSPACE)
    pop_gs = top_outer[1].subgridspec(3, 2, wspace=TOP_COL_WSPACE, hspace=TOP_ROW_HSPACE)
    moment_ratio_gs = top_outer[2].subgridspec(3, 1, hspace=TOP_ROW_HSPACE)
    bottom_gs = main_gs[1].subgridspec(
        2, n_tiles,
        wspace=BOTTOM_STRIP_WSPACE,
        hspace=BOTTOM_STRIP_HSPACE,
        width_ratios=[1] * n_tiles,
        height_ratios=[BOTTOM_STRIP_IMG_FRAC, 1.0 - BOTTOM_STRIP_IMG_FRAC],
    )

    top_axes = _draw_top_grid(
        fig, scatter_gs, pop_gs, datasets, ir_fit, o_fit,
        struct_cache, k, struct_cmap,
    )
    struct_axes, label_axes = _draw_horizontal_centroid_strip(
        fig, bottom_gs, entries, motif_json, struct_cmap, shared_half_span,
    )

    _apply_panel_layout(
        fig, top_axes, moment_ratio_gs, moment_ratio_data,
        struct_axes, label_axes, ir_fit, o_fit,
    )

    # Fixed canvas: never use bbox_inches='tight' (moment-ratio zoom must not resize output).
    fig.savefig(OUTPUT_PATH, format="svg", dpi=DPI, bbox_inches=None, pad_inches=0)
    plt.close(fig)
    print(f"[save] {OUTPUT_PATH}")
    print(
        f"\n[fig02] {OUTPUT_PATH}  "
        f"({FIG_W_MM}×{FIG_H_MM} mm: 3×4 clustering + moment ratios + {n_tiles} horizontal centroids)"
    )


if __name__ == "__main__":
    main()
