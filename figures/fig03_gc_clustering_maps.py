"""Figure 3 – Grand-canonical phase diagrams with ACE motif overlays.

Layout (180 mm wide):
  Row 0: PBE | PBE0 ensemble oxidation ⟨n_O/n_Ir⟩ phase maps (former Fig. 3 row 1)
  Rows 1–4: Per-Ir gradient bars at ``FIG04_LOW_PRESSURE_PA`` — ⟨n_O/n_Ir⟩,
            structure cluster, Ir motif, O motif (``supp_gc_clustering_bars_low_po2``
            rows 1–4)

Output: ``figures/outputs/fig03_gc_clustering_maps.svg``
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator

from figures.config import (
    ACE_FEATURES,
    DPI,
    FIG04_LOW_PRESSURE_PA,
    OUTPUT_DIR,
    PHASE_CMAP,
    THERMO_IR_COUNTS,
    TRAJ_PBE,
    TRAJ_PBE0,
)
from figures.utils.cache import ensure_for_figures
from figures.utils.comparison import (
    atom_motif_colors_labels,
    build_comparison_joined_table,
    load_gc_comparison_view,
    structure_cluster_colors_labels,
)
from figures.utils.gc_gradient_bars import (
    SectionKind,
    draw_rgb_gradient_bars,
    draw_scalar_gradient_bars,
    ensemble_scalar_vs_T_at_pressure,
    per_structure_cluster_rgb,
    per_structure_rgb_from_fractions,
    sync_temperature_xaxis_grid,
)
from figures.utils.io import ensure_dirs, load_ace_features
from figures.utils.plotting import (
    AXIS_LABEL_FONTSIZE,
    TICK_FONTSIZE,
    add_phase_colorbar,
    apply_gc_temperature_xticks,
    discrete_category_colorbar,
    draw_global_minimum_phase_overlay,
    margin_colorbar_cax,
    stoichiometry_overlay_labels,
    style_phase_ax,
    style_shared_gc_panel_ax,
    save_fig,
)

OUTPUT_PATH = os.path.join(OUTPUT_DIR, "fig03_gc_clustering_maps.svg")
FIG_W_MM = 180
MAP_ROW_H_MM = 42
BAR_ROW_H_MM = 44
N_BAR_ROWS = 4
N_ROWS = 1 + N_BAR_ROWS
FIG_H_MM = MAP_ROW_H_MM + N_BAR_ROWS * BAR_ROW_H_MM
LAYOUT_RIGHT = 0.91
CBAR_GAP = 0.008
CBAR_WIDTH = 0.012
MAP_HSPACE = 0.22
BAR_HSPACE = 0.48
WSPACE = 0.12

DATASETS = [
    {
        "label": "PBE",
        "load_gc": lambda: load_gc_comparison_view("PBE"),
        "ace_csv": ACE_FEATURES["PBE"],
        "traj": TRAJ_PBE,
        "join_tag": "PBE",
    },
    {
        "label": "PBE0",
        "load_gc": lambda: load_gc_comparison_view("PBE0"),
        "ace_csv": ACE_FEATURES["PBE0"],
        "traj": TRAJ_PBE0,
        "join_tag": "PBE0",
    },
]

BAR_ROW_SPECS: tuple[tuple[str, str, SectionKind, str | None], ...] = (
    ("stoich", "scalar", "global_min", None),
    ("cluster_rgb", "rgb", "majority_cluster", "cluster_labels"),
    ("ir_rgb", "rgb", "none", None),
    ("o_rgb", "rgb", "none", None),
)


def _load_column(ds: dict) -> dict | None:
    try:
        gc = ds["load_gc"]()
    except FileNotFoundError:
        return None
    traj_path = ds["traj"]
    if not os.path.exists(traj_path):
        return None
    rows = load_ace_features(ds["ace_csv"])
    jt = build_comparison_joined_table(gc, rows, ds.get("join_tag", "PBE"))

    struct_colors, _ = structure_cluster_colors_labels(jt)
    ir_colors, ir_labels = atom_motif_colors_labels(jt, "Ir")
    o_colors, o_labels = atom_motif_colors_labels(jt, "O")
    n_ir = np.asarray(jt["n_Ir"], dtype=float)
    n_o = np.asarray(jt["n_O"], dtype=float)
    stoich = np.where(n_ir > 0, n_o / n_ir, np.nan)

    return {
        "label": ds["label"],
        "gc": gc,
        "joined": jt,
        "stoich": stoich,
        "stoich_labels": stoichiometry_overlay_labels(gc["formulas"][:jt["n"]]),
        "cluster_rgb": per_structure_cluster_rgb(jt["cluster"], struct_colors),
        "ir_rgb": per_structure_rgb_from_fractions(jt["ir_frac"], ir_colors),
        "o_rgb": per_structure_rgb_from_fractions(jt["o_frac"], o_colors),
        "struct_colors": struct_colors,
        "cluster_labels": [f"C{c}" for c in range(len(struct_colors))],
        "ir_colors": ir_colors,
        "ir_labels": ir_labels,
        "o_colors": o_colors,
        "o_labels": o_labels,
    }


def _oxidation_limits(
    columns: list[dict | None],
    pressure_pa: float,
) -> tuple[float, float]:
    vals: list[float] = []
    for col in columns:
        if col is None:
            continue
        Z = np.asarray(col["gc"]["x_per_non_o"], dtype=float)
        finite = Z[np.isfinite(Z)]
        if finite.size:
            vals.extend([float(finite.min()), float(finite.max())])
        for n in THERMO_IR_COUNTS:
            curve = ensemble_scalar_vs_T_at_pressure(
                col["joined"], n, col["stoich"], pressure_pa,
            )
            if curve is None:
                continue
            bar_finite = curve[2][np.isfinite(curve[2])]
            if bar_finite.size:
                vals.extend([float(bar_finite.min()), float(bar_finite.max())])
    if not vals:
        return 0.0, 1.0
    vmin, vmax = float(min(vals)), float(max(vals))
    if vmax <= vmin:
        vmax = vmin + 1.0
    return vmin, vmax


def _draw_oxidation_map(
    ax,
    col: dict,
    *,
    vmin: float,
    vmax: float,
    pressure_pa: float,
) -> object | None:
    gc = col["gc"]
    Ts, Ps = gc["Ts"], gc["Ps"]
    Z = np.asarray(gc["x_per_non_o"], dtype=float)
    if not np.isfinite(Z).any():
        ax.text(
            0.5, 0.5, "No data", ha="center", va="center",
            fontsize=AXIS_LABEL_FONTSIZE, transform=ax.transAxes,
        )
        return None

    mesh = ax.pcolormesh(
        Ts, Ps, Z, shading="auto",
        cmap=PHASE_CMAP,
        norm=Normalize(vmin=vmin, vmax=vmax),
        rasterized=True,
    )
    draw_global_minimum_phase_overlay(
        ax, Ts, Ps, gc["phase_index"], col["stoich_labels"],
    )
    ax.axhline(
        pressure_pa,
        color="black",
        linewidth=0.8,
        linestyle="--",
        zorder=9,
    )
    style_phase_ax(ax)
    return mesh


def _style_bar_axes(axes: np.ndarray) -> None:
    for row in range(axes.shape[0]):
        for col_idx in range(axes.shape[1]):
            ax = axes[row, col_idx]
            if not ax.axison:
                continue
            ax.set_yscale("linear")
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))   # n_Ir rows, not 3.5, 4.0, …
            if col_idx == 0:
                ax.set_ylabel(r"$n_{\mathrm{Ir}}$", fontsize=AXIS_LABEL_FONTSIZE)
            else:
                ax.set_ylabel("")
                ax.tick_params(labelleft=False)
            ax.tick_params(axis="y", labelsize=TICK_FONTSIZE)


def main() -> None:
    ensure_for_figures([3])
    ensure_dirs(OUTPUT_DIR)

    pressure_pa = FIG04_LOW_PRESSURE_PA
    columns = [_load_column(ds) for ds in DATASETS]
    if not any(c is not None for c in columns):
        raise FileNotFoundError("No GC data found for figure 3.")

    ox_vmin, ox_vmax = _oxidation_limits(columns, pressure_pa)
    ref = next(c for c in columns if c is not None)
    ts_ref = np.asarray(ref["joined"]["Ts"], dtype=float)

    fig = plt.figure(figsize=(FIG_W_MM / 25.4, FIG_H_MM / 25.4), dpi=DPI)
    gs = fig.add_gridspec(
        N_ROWS, 2,
        height_ratios=[MAP_ROW_H_MM] + [BAR_ROW_H_MM] * N_BAR_ROWS,
        hspace=MAP_HSPACE,
        wspace=WSPACE,
    )
    axes = np.empty((N_ROWS, 2), dtype=object)
    for row in range(N_ROWS):
        for col_idx in range(2):
            axes[row, col_idx] = fig.add_subplot(gs[row, col_idx])
            if row == 0 and col_idx > 0:
                axes[row, col_idx].sharey(axes[row, 0])
            if row > 0:
                axes[row, col_idx].sharex(axes[1, col_idx])

    mesh_ref = None
    col_titles = ("PBE", "PBE0")
    last_row = N_ROWS - 1
    for col_idx, col in enumerate(columns):
        ax = axes[0, col_idx]
        if col is None:
            ax.text(
                0.5, 0.5, "GC data missing", ha="center", va="center",
                fontsize=AXIS_LABEL_FONTSIZE, transform=ax.transAxes,
            )
            ax.set_axis_off()
            continue
        mesh = _draw_oxidation_map(
            ax, col, vmin=ox_vmin, vmax=ox_vmax, pressure_pa=pressure_pa,
        )
        style_shared_gc_panel_ax(
            ax, 0, col_idx, last_row, title=col_titles[col_idx],
        )
        apply_gc_temperature_xticks(ax)
        ax.tick_params(labelbottom=True)
        ax.set_xlabel("")
        if mesh_ref is None and mesh is not None:
            mesh_ref = mesh

    bar_axes = axes[1:, :]
    for bar_row, (key, kind, section_kind, labels_key) in enumerate(BAR_ROW_SPECS):
        for col_idx, col in enumerate(columns):
            ax = bar_axes[bar_row, col_idx]
            if col is None:
                ax.text(
                    0.5, 0.5, "GC data missing", ha="center", va="center",
                    fontsize=AXIS_LABEL_FONTSIZE, transform=ax.transAxes,
                )
                ax.set_axis_off()
                continue
            section_labels = col[labels_key] if labels_key else None
            if kind == "scalar":
                draw_scalar_gradient_bars(
                    ax, col["joined"], pressure_pa, THERMO_IR_COUNTS, col[key],
                    cmap=PHASE_CMAP, vmin=ox_vmin, vmax=ox_vmax,
                    section_kind=section_kind,
                    section_labels=section_labels,
                )
            else:
                draw_rgb_gradient_bars(
                    ax, col["joined"], pressure_pa, THERMO_IR_COUNTS, col[key],
                    section_kind=section_kind,
                    section_labels=section_labels,
                )
            ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)

    _style_bar_axes(bar_axes)
    sync_temperature_xaxis_grid(bar_axes, ts_ref)

    fig.subplots_adjust(
        left=0.08,
        right=LAYOUT_RIGHT,
        top=0.98,
        bottom=0.05,
        hspace=BAR_HSPACE,
        wspace=WSPACE,
    )

    cbar_kw = {"gap": CBAR_GAP, "width": CBAR_WIDTH}
    if mesh_ref is not None:
        add_phase_colorbar(
            fig, mesh_ref,
            cax=margin_colorbar_cax(fig, axes[0, 0], axes[0, 1], **cbar_kw),
            label=r"$n_O/n_{Ir}$",
        )

    mesh_ox_bar = ScalarMappable(norm=Normalize(vmin=ox_vmin, vmax=ox_vmax), cmap=PHASE_CMAP)
    mesh_ox_bar.set_array([])
    add_phase_colorbar(
        fig, mesh_ox_bar,
        cax=margin_colorbar_cax(fig, bar_axes[0, 0], bar_axes[0, 1], **cbar_kw),
        label=r"$n_O/n_{Ir}$",
    )

    discrete_category_colorbar(
        ref["struct_colors"], ref["cluster_labels"],
        cax=margin_colorbar_cax(fig, bar_axes[1, 0], bar_axes[1, 1], **cbar_kw),
        cbar_label="Cluster",
    )
    discrete_category_colorbar(
        ref["ir_colors"], ref["ir_labels"],
        cax=margin_colorbar_cax(fig, bar_axes[2, 0], bar_axes[2, 1], **cbar_kw),
    )
    discrete_category_colorbar(
        ref["o_colors"], ref["o_labels"],
        cax=margin_colorbar_cax(fig, bar_axes[3, 0], bar_axes[3, 1], **cbar_kw),
    )

    save_fig(fig, OUTPUT_PATH, dpi=DPI, bbox_inches="tight", pad_inches=0)
    print(f"\n[fig03] {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
