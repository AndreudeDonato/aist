"""Supplementary – Fig. 3–style clustering panels with per-size gradient bars at fixed pO₂.

Mirrors Fig. 3 (oxidation, cluster, Ir motif, O motif, inertia) plus
``⟨HOMO–LUMO⟩`` (Fig. 5 property, not in main Fig. 3). Each panel is
population-weighted gradient bars vs ``T`` — one band per ``n_Ir`` (Ir₃…Ir₉,
top → bottom) at fixed pressure.

Outputs (180 × 252 mm each):
  ``supp_gc_clustering_bars_low_po2.svg``  — ``FIG04_LOW_PRESSURE_PA``
  ``supp_gc_clustering_bars_high_po2.svg`` — ``FIG04_HIGH_PRESSURE_PA``
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from ase.io import read
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import MaxNLocator

from aist.ace.pipeline import extent_blend as ACE_EXTENT_BLEND
from aist.inertia import per_structure_shape_extent_rgb
from aist.thermo import ThermoPartitionFunctions

from figures.config import (
    ACE_FEATURES,
    DPI,
    FIG04_HIGH_PRESSURE_PA,
    FIG04_LOW_PRESSURE_PA,
    OUTPUT_DIR,
    PHASE_CMAP,
    THERMO_IR_COUNTS,
    TRAJ_PBE,
    TRAJ_PBE0,
)
from figures.utils.cache import ensure_for_supplementary
from figures.utils.comparison import (
    per_frame_rows,
    atom_motif_colors_labels,
    build_comparison_joined_table,
    homo_lumo_gaps_for_comparison,
    load_gc_comparison_view,
    structure_cluster_colors_labels,
)
from figures.utils.homo_lumo import HOMO_LUMO_CMAP
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
    TITLE_FONTSIZE,
    add_phase_colorbar,
    discrete_category_colorbar,
    margin_colorbar_cax,
    save_fig,
)

FIG_W_MM = 180
FIG_H_MM = 252
N_ROWS = 6
ACE_EXTENT_NORMALIZATION = "partial"
FIG03_LAYOUT_RECT = [0.04, 0, 0.91, 1]
FIG03_CBAR_GAP = 0.008
FIG03_CBAR_WIDTH = 0.012
ROW_H_MM = 42
SUBPLOT_HSPACE = 0.28
SUBPLOT_WSPACE = 0.12

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

OUTPUTS = (
    (FIG04_LOW_PRESSURE_PA, "supp_gc_clustering_bars_low_po2.svg"),
    (FIG04_HIGH_PRESSURE_PA, "supp_gc_clustering_bars_high_po2.svg"),
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
    tp = ThermoPartitionFunctions()
    structures = list(read(traj_path, ":"))
    inertia_rgb = per_structure_shape_extent_rgb(
        structures, tp,
        extent_normalization=ACE_EXTENT_NORMALIZATION,
        extent_blend=ACE_EXTENT_BLEND,
    )
    n_traj = len(structures)
    inertia_rgb = per_frame_rows(inertia_rgb, jt["frame_index"])   # GC row order, gas rows NaN

    struct_colors, _ = structure_cluster_colors_labels(jt)
    ir_colors, ir_labels = atom_motif_colors_labels(jt, "Ir")
    o_colors, o_labels = atom_motif_colors_labels(jt, "O")
    n_ir = np.asarray(jt["n_Ir"], dtype=float)
    n_o = np.asarray(jt["n_O"], dtype=float)
    stoich = np.where(n_ir > 0, n_o / n_ir, np.nan)

    return {
        "label": ds["label"],
        "joined": jt,
        "gaps": homo_lumo_gaps_for_comparison(ds["label"], traj_path, jt["n"], frame_index=jt["frame_index"]),
        "stoich": stoich,
        "cluster_rgb": per_structure_cluster_rgb(jt["cluster"], struct_colors),
        "ir_rgb": per_structure_rgb_from_fractions(jt["ir_frac"], ir_colors),
        "o_rgb": per_structure_rgb_from_fractions(jt["o_frac"], o_colors),
        "inertia_rgb": np.asarray(inertia_rgb, dtype=float),
        "struct_colors": struct_colors,
        "cluster_labels": [f"C{c}" for c in range(len(struct_colors))],
        "ir_colors": ir_colors,
        "ir_labels": ir_labels,
        "o_colors": o_colors,
        "o_labels": o_labels,
    }


def _scalar_field_limits(
    columns: list[dict | None],
    pressure_pa: float,
    field_key: str,
) -> tuple[float, float]:
    vals: list[float] = []
    for col in columns:
        if col is None:
            continue
        per_structure = col[field_key]
        for n in THERMO_IR_COUNTS:
            curve = ensemble_scalar_vs_T_at_pressure(
                col["joined"], n, per_structure, pressure_pa,
            )
            if curve is None:
                continue
            finite = curve[2][np.isfinite(curve[2])]
            if finite.size:
                vals.extend([float(finite.min()), float(finite.max())])
    if not vals:
        return 0.0, 1.0
    vmin, vmax = float(min(vals)), float(max(vals))
    if vmax <= vmin:
        vmax = vmin + 1.0
    return vmin, vmax


def _draw_bar_row(
    ax,
    col: dict,
    *,
    pressure_pa: float,
    per_structure,
    cmap: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    section_kind: SectionKind = "none",
    section_labels: list[str] | None = None,
) -> None:
    if cmap is None:
        draw_rgb_gradient_bars(
            ax, col["joined"], pressure_pa, THERMO_IR_COUNTS, per_structure,
            section_kind=section_kind,
            section_labels=section_labels,
        )
    else:
        draw_scalar_gradient_bars(
            ax, col["joined"], pressure_pa, THERMO_IR_COUNTS, per_structure,
            cmap=cmap, vmin=float(vmin), vmax=float(vmax),
            section_kind=section_kind,
            section_labels=section_labels,
        )
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)


def _apply_shared_axis_style(axes: np.ndarray) -> None:
    col_titles = ("PBE", "PBE0")
    for row in range(axes.shape[0]):
        for col_idx in range(axes.shape[1]):
            ax = axes[row, col_idx]
            if not ax.axison:
                continue
            if row == 0:
                ax.set_title(col_titles[col_idx], fontsize=TITLE_FONTSIZE, pad=3)
            ax.set_yscale("linear")
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))   # n_Ir rows, not 3.5, 4.0, …
            if col_idx == 0:
                ax.set_ylabel(r"$n_{\mathrm{Ir}}$", fontsize=AXIS_LABEL_FONTSIZE)
            else:
                ax.set_ylabel("")
                ax.tick_params(labelleft=False)
            ax.tick_params(axis="y", labelsize=TICK_FONTSIZE)


def _render_figure(pressure_pa: float, filename: str) -> None:
    columns = [_load_column(ds) for ds in DATASETS]
    if not any(c is not None for c in columns):
        raise FileNotFoundError("No GC data for clustering pressure-bar supplementary.")

    ox_vmin, ox_vmax = _scalar_field_limits(columns, pressure_pa, "stoich")
    hl_vmin, hl_vmax = _scalar_field_limits(columns, pressure_pa, "gaps")

    fig, axes = plt.subplots(
        N_ROWS, 2,
        figsize=(FIG_W_MM / 25.4, FIG_H_MM / 25.4),
        dpi=DPI,
        sharex="col",
        gridspec_kw={"height_ratios": [ROW_H_MM] * N_ROWS},
    )

    row_specs = (
        ("⟨n_O/n_Ir⟩", "stoich", PHASE_CMAP, ox_vmin, ox_vmax, "global_min", None),
        ("structure cluster", "cluster_rgb", None, None, None, "majority_cluster", "cluster_labels"),
        ("Ir atom motif", "ir_rgb", None, None, None, "none", None),
        ("O atom motif", "o_rgb", None, None, None, "none", None),
        ("inertia shape × extent", "inertia_rgb", None, None, None, "none", None),
        ("⟨HOMO–LUMO⟩", "gaps", HOMO_LUMO_CMAP, hl_vmin, hl_vmax, "none", None),
    )

    for row, spec in enumerate(row_specs):
        _row_label, key, cmap, vmin, vmax, section_kind, labels_key = spec
        for col_idx, col in enumerate(columns):
            ax = axes[row, col_idx]
            if col is None:
                ax.text(
                    0.5, 0.5, "GC data missing", ha="center", va="center",
                    fontsize=AXIS_LABEL_FONTSIZE, transform=ax.transAxes,
                )
                ax.set_axis_off()
                continue
            section_labels = col[labels_key] if labels_key else None
            _draw_bar_row(
                ax, col,
                pressure_pa=pressure_pa,
                per_structure=col[key],
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                section_kind=section_kind,
                section_labels=section_labels,
            )

    ref = next(c for c in columns if c is not None)
    ts_ref = np.asarray(ref["joined"]["Ts"], dtype=float)
    _apply_shared_axis_style(axes)
    fig.subplots_adjust(
        left=0.08,
        right=FIG03_LAYOUT_RECT[2],
        top=0.98,
        bottom=0.05,
        hspace=SUBPLOT_HSPACE,
        wspace=SUBPLOT_WSPACE,
    )
    sync_temperature_xaxis_grid(axes, ts_ref)
    for ax in axes.ravel():
        if ax.axison:
            ax.tick_params(axis="x", labelbottom=True)

    cbar_kw = {"gap": FIG03_CBAR_GAP, "width": FIG03_CBAR_WIDTH}

    mesh0 = ScalarMappable(norm=Normalize(vmin=ox_vmin, vmax=ox_vmax), cmap=PHASE_CMAP)
    mesh0.set_array([])
    add_phase_colorbar(
        fig, mesh0,
        cax=margin_colorbar_cax(fig, axes[0, 0], axes[0, 1], **cbar_kw),
        label=r"$n_O/n_{Ir}$",
    )
    discrete_category_colorbar(
        ref["struct_colors"], ref["cluster_labels"],
        cax=margin_colorbar_cax(fig, axes[1, 0], axes[1, 1], **cbar_kw),
        cbar_label="Cluster",
    )
    discrete_category_colorbar(
        ref["ir_colors"], ref["ir_labels"],
        cax=margin_colorbar_cax(fig, axes[2, 0], axes[2, 1], **cbar_kw),
    )
    discrete_category_colorbar(
        ref["o_colors"], ref["o_labels"],
        cax=margin_colorbar_cax(fig, axes[3, 0], axes[3, 1], **cbar_kw),
    )
    discrete_category_colorbar(
        ["#e85c5c", "#5ce85c", "#5c5ce8"],
        ["L", "P", "S"],
        cax=margin_colorbar_cax(fig, axes[4, 0], axes[4, 1], **cbar_kw),
        cbar_label="Shape",
    )
    mesh_hl = ScalarMappable(norm=Normalize(vmin=hl_vmin, vmax=hl_vmax), cmap=HOMO_LUMO_CMAP)
    mesh_hl.set_array([])
    add_phase_colorbar(
        fig, mesh_hl,
        cax=margin_colorbar_cax(fig, axes[5, 0], axes[5, 1], **cbar_kw),
        label="HOMO–LUMO gap (eV)",
    )

    out_path = os.path.join(OUTPUT_DIR, filename)
    save_fig(fig, out_path, dpi=DPI, bbox_inches="tight", pad_inches=0)
    print(f"[supp] {out_path}")


def main() -> None:
    ensure_for_supplementary(["supp_gc_clustering_pressure_bars"])
    ensure_dirs(OUTPUT_DIR)
    for pressure_pa, filename in OUTPUTS:
        _render_figure(pressure_pa, filename)


if __name__ == "__main__":
    main()
