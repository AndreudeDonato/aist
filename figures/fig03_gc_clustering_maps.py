"""Figure 3 – Grand-canonical phase diagrams with ACE motif overlays.

Layout: 5 rows × 2 columns
  Row 1: Ensemble-averaged oxidation ⟨n_O/n_Ir⟩
  Row 2: Structure-cluster population (RGB mix of cluster colours)
  Row 3: Ir atom-motif population (RGB mix of Ir motif colours)
  Row 4: O atom-motif population (RGB mix of O motif colours)
  Row 5: Inertia shape × extent (R=L, G=P, B=S; lightness = extent)

Each row: PBE | PBE0 with shared T and pO₂ axes and one colorbar per row.

Row 1 (PBE | PBE0): global-minimum boundaries + stoichiometry labels.
Row 2: majority-cluster boundaries + ``C*`` label of the dominant cluster.
Rows 3–5: no stoichiometry overlay.

Output: ``figures/outputs/fig03_gc_clustering_maps.svg``
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from ase.io import read
from matplotlib.colors import Normalize

from aist.ace.common import ace_atom_display_label
from aist.ace.pipeline import extent_blend as ACE_EXTENT_BLEND
from aist.inertia import per_structure_shape_extent_rgb
from aist.thermo import ThermoPartitionFunctions

from figures.config import (
    ACE_FEATURES,
    GC_NPZ,
    GC_NPZ_PBE0,
    GC_VARIANTS_NPZ,
    OUTPUT_DIR,
    TRAJ_PBE,
    TRAJ_PBE0,
    DPI,
)
from figures.utils.io import (
    ensure_dirs,
    load_ace_features,
    load_gc_npz,
    load_gc_pbe_novib_view,
)
from figures.utils.cache import ensure_for_figures
from figures.utils.join import (
    build_joined_table,
    cluster_population_fractions,
    ensemble_weighted_rgb_map,
    ir_motif_population_fractions,
    o_motif_population_fractions,
)
from figures.utils.plotting import (
    TITLE_FONTSIZE,
    AXIS_LABEL_FONTSIZE,
    add_phase_colorbar,
    cluster_colors,
    discrete_category_colorbar,
    draw_global_minimum_phase_overlay,
    draw_population_mixed_phase_panel,
    draw_shape_extent_rgb_phase_panel,
    ir_motif_colors,
    margin_colorbar_cax,
    o_motif_colors,
    stoichiometry_overlay_labels,
    style_phase_ax,
    style_shared_gc_panel_ax,
)

OUTPUT_PATH = os.path.join(OUTPUT_DIR, "fig03_gc_clustering_maps.svg")
N_ROWS = 5
FIG_W_MM = 180
FIG_H_MM = 210
ACE_EXTENT_NORMALIZATION = "partial"

DATASETS = [
    {
        "label": "PBE",
        "load_gc": lambda: load_gc_pbe_novib_view(GC_VARIANTS_NPZ, GC_NPZ),
        "ace_csv": ACE_FEATURES["PBE"],
        "traj": TRAJ_PBE,
        "join_tag": "PBE",
    },
    {
        "label": "PBE0",
        "load_gc": lambda: load_gc_npz(GC_NPZ_PBE0),
        "ace_csv": ACE_FEATURES["PBE0"],
        "traj": TRAJ_PBE0,
        "join_tag": "PBE0",
    },
]

PHASE_CMAP = "coolwarm"
FIG03_LAYOUT_RECT = [0.04, 0, 0.91, 1]
FIG03_CBAR_GAP = 0.008
FIG03_CBAR_WIDTH = 0.012


def _scalar_ensemble_panel(
    ax,
    gc,
    field,
    title,
    cmap,
    vmin,
    vmax,
    *,
    overlay_labels: list[str] | None = None,
):
    """2D phase map of an ensemble-averaged scalar field."""
    Ts, Ps = gc["Ts"], gc["Ps"]
    Z = np.asarray(field, dtype=float)
    if not np.isfinite(Z).any():
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                fontsize=AXIS_LABEL_FONTSIZE, transform=ax.transAxes)
        return None

    if vmin is None:
        vmin = float(np.nanmin(Z))
    if vmax is None:
        vmax = float(np.nanmax(Z))
    if vmax <= vmin:
        vmax = vmin + 1.0

    mesh = ax.pcolormesh(
        Ts, Ps, Z, shading="auto",
        cmap=cmap,
        norm=Normalize(vmin=vmin, vmax=vmax),
        rasterized=True,
    )
    if overlay_labels is not None:
        draw_global_minimum_phase_overlay(
            ax, Ts, Ps, gc["phase_index"], overlay_labels,
        )
    style_phase_ax(ax)
    if title:
        ax.set_title(title, fontsize=TITLE_FONTSIZE)
    return mesh


def _load_column(ds: dict) -> dict | None:
    try:
        gc = ds["load_gc"]()
    except FileNotFoundError:
        return None
    traj_path = ds["traj"]
    if not os.path.exists(traj_path):
        return None
    rows = load_ace_features(ds["ace_csv"])
    jt = build_joined_table(gc, rows, ds.get("join_tag", "PBE"))
    n = jt["n"]
    structures = list(read(traj_path, ":"))[:n]
    tp = ThermoPartitionFunctions()
    inertia_rgb = per_structure_shape_extent_rgb(
        structures, tp,
        extent_normalization=ACE_EXTENT_NORMALIZATION,
        extent_blend=ACE_EXTENT_BLEND,
    )
    inertia_field = ensemble_weighted_rgb_map(jt["Pgi"], inertia_rgb)
    return {
        "label": ds["label"],
        "gc": gc,
        "jt": jt,
        "cluster_pop": cluster_population_fractions(jt),
        "ir_pop": ir_motif_population_fractions(jt),
        "o_pop": o_motif_population_fractions(jt),
        "inertia_rgb": inertia_field,
        "stoich_labels": stoichiometry_overlay_labels(gc["formulas"][:jt["n"]]),
        "Ts": gc["Ts"],
        "Ps": gc["Ps"],
        "k_struct": int(jt["cluster"].max()) + 1,
        "k_Ir": jt["ir_frac"].shape[1],
        "k_O": jt["o_frac"].shape[1],
    }


def _scalar_limits(columns: list[dict | None], field_key: str) -> tuple[float, float]:
    vals: list[float] = []
    for col in columns:
        if col is None:
            continue
        if field_key == "oxidation":
            Z = np.asarray(col["gc"]["x_per_non_o"], dtype=float)
        else:
            Z = np.asarray(col[field_key], dtype=float)
        finite = Z[np.isfinite(Z)]
        if finite.size:
            vals.extend([float(finite.min()), float(finite.max())])
    if not vals:
        return 0.0, 1.0
    vmin, vmax = float(min(vals)), float(max(vals))
    if vmax <= vmin:
        vmax = vmin + 1.0
    return vmin, vmax


def _apply_shared_axis_style(axes: np.ndarray) -> None:
    last_row = axes.shape[0] - 1
    col_titles = ("PBE", "PBE0")
    for row in range(axes.shape[0]):
        for col in range(axes.shape[1]):
            ax = axes[row, col]
            if not ax.axison:
                continue
            title = col_titles[col] if row == 0 else ""
            style_shared_gc_panel_ax(ax, row, col, last_row, title=title)


def main() -> None:
    ensure_for_figures([3])
    ensure_dirs(OUTPUT_DIR)

    columns = [_load_column(ds) for ds in DATASETS]

    fig, axes = plt.subplots(
        N_ROWS, 2,
        figsize=(FIG_W_MM / 25.4, FIG_H_MM / 25.4),
        dpi=DPI,
        sharex=True,
        sharey=True,
    )

    ox_vmin, ox_vmax = _scalar_limits(columns, "oxidation")
    mesh0_ref = None

    for col, data in enumerate(columns):
        if data is None:
            for row in range(N_ROWS):
                axes[row, col].text(
                    0.5, 0.5, "GC data missing",
                    ha="center", va="center", fontsize=AXIS_LABEL_FONTSIZE,
                    transform=axes[row, col].transAxes,
                )
                axes[row, col].set_axis_off()
            continue

        gc = data["gc"]
        Ts, Ps = data["Ts"], data["Ps"]

        mesh0 = _scalar_ensemble_panel(
            axes[0, col], gc, gc["x_per_non_o"],
            "",
            PHASE_CMAP, ox_vmin, ox_vmax,
            overlay_labels=data["stoich_labels"],
        )
        if mesh0 is not None:
            mesh0_ref = mesh0

        struct_colors = cluster_colors(data["k_struct"])
        cluster_majority_labels = [f"C{c}" for c in range(data["k_struct"])]
        draw_population_mixed_phase_panel(
            axes[1, col], Ts, Ps, data["cluster_pop"], struct_colors,
            "",
            majority_labels=cluster_majority_labels,
        )

        ir_colors = ir_motif_colors(data["k_Ir"])
        draw_population_mixed_phase_panel(
            axes[2, col], Ts, Ps, data["ir_pop"], ir_colors,
            "",
        )

        o_colors = o_motif_colors(data["k_O"])
        draw_population_mixed_phase_panel(
            axes[3, col], Ts, Ps, data["o_pop"], o_colors,
            "",
        )

        draw_shape_extent_rgb_phase_panel(
            axes[4, col], Ts, Ps, data["inertia_rgb"],
            "",
        )

    _apply_shared_axis_style(axes)

    fig.tight_layout(pad=6 / 25.4, rect=FIG03_LAYOUT_RECT)

    ref = next(c for c in columns if c is not None)
    k_struct, k_Ir, k_O = ref["k_struct"], ref["k_Ir"], ref["k_O"]
    struct_colors = cluster_colors(k_struct)
    ir_colors = ir_motif_colors(k_Ir)
    o_colors = o_motif_colors(k_O)

    cbar_kw = {"gap": FIG03_CBAR_GAP, "width": FIG03_CBAR_WIDTH}

    if mesh0_ref is not None:
        add_phase_colorbar(
            fig, mesh0_ref,
            cax=margin_colorbar_cax(fig, axes[0, 0], axes[0, 1], **cbar_kw),
            label=r"$n_O/n_{Ir}$",
        )

    discrete_category_colorbar(
        struct_colors,
        [f"C{c}" for c in range(k_struct)],
        cax=margin_colorbar_cax(fig, axes[1, 0], axes[1, 1], **cbar_kw),
        cbar_label="Cluster",
    )
    discrete_category_colorbar(
        ir_colors,
        [ace_atom_display_label("Ir", c) for c in range(k_Ir)],
        cax=margin_colorbar_cax(fig, axes[2, 0], axes[2, 1], **cbar_kw),
    )
    discrete_category_colorbar(
        o_colors,
        [ace_atom_display_label("O", c) for c in range(k_O)],
        cax=margin_colorbar_cax(fig, axes[3, 0], axes[3, 1], **cbar_kw),
    )
    discrete_category_colorbar(
        ["#e85c5c", "#5ce85c", "#5c5ce8"],
        ["L", "P", "S"],
        cax=margin_colorbar_cax(fig, axes[4, 0], axes[4, 1], **cbar_kw),
        cbar_label="Shape",
    )

    fig.savefig(OUTPUT_PATH, format="svg", dpi=DPI, bbox_inches=None, pad_inches=0)
    plt.close(fig)
    print(f"[save] {OUTPUT_PATH}")
    print(f"\n[fig03] {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
