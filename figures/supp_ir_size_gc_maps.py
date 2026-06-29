"""Supplementary – GC phase maps per Ir-atom count (PBE | PBE0).

Seven outputs (7 rows × 2 columns each, Ir₃…Ir₉ × PBE | PBE0), matching the
property rows of Figure 3 but restricted to one ``n_Ir`` bucket per row:

  ``supp_ir_size_oxidation_maps.svg``   — ⟨n_O/n_Ir⟩
  ``supp_ir_size_cluster_maps.svg``     — structure-cluster RGB population
  ``supp_ir_size_ir_motif_maps.svg``    — Ir atom-motif RGB population
  ``supp_ir_size_o_motif_maps.svg``      — O atom-motif RGB population
  ``supp_ir_size_inertia_maps.svg``     — inertia shape × extent RGB
  ``supp_ir_size_homo_lumo_maps.svg`` — ⟨HOMO–LUMO gap⟩
  ``supp_ir_size_global_min_pop_maps.svg`` — global-minimum population (%)
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from ase.io import read
from matplotlib.colors import Normalize

from aist.ace.common import ace_atom_display_label
from aist.ace.pipeline import extent_blend as ACE_EXTENT_BLEND
from aist.core.grandcanonical import global_minimum_population_map
from aist.inertia import per_structure_shape_extent_rgb
from aist.thermo import ThermoPartitionFunctions

from figures.config import (
    ACE_FEATURES,
    DPI,
    GC_NPZ,
    GC_NPZ_PBE0,
    GC_VARIANTS_NPZ,
    OUTPUT_DIR,
    PHASE_CMAP,
    TRAJ_PBE,
    TRAJ_PBE0,
)
from figures.utils.cache import ensure_for_supplementary
from figures.utils.gc_subset import subset_joined_table_by_n_ir
from figures.utils.homo_lumo import HOMO_LUMO_CMAP, homo_lumo_gap_map, homo_lumo_gaps
from figures.utils.io import (
    ensure_dirs,
    load_ace_features,
    load_gc_npz,
    load_gc_pbe_novib_view,
)
from figures.utils.join import (
    build_joined_table,
    cluster_population_fractions,
    ensemble_weighted_rgb_map,
    ir_motif_population_fractions,
    o_motif_population_fractions,
)
from figures.utils.plotting import (
    AXIS_LABEL_FONTSIZE,
    add_phase_colorbar,
    cluster_colors,
    discrete_category_colorbar,
    draw_global_minimum_phase_overlay,
    draw_global_minimum_population_panel,
    draw_population_mixed_phase_panel,
    draw_shape_extent_rgb_phase_panel,
    ir_motif_colors,
    margin_colorbar_cax,
    o_motif_colors,
    save_fig,
    stoichiometry_overlay_labels,
    style_ir_size_panel_ax,
)

FIG_W_MM = 180
ROW_H_MM = 32
FIG_H_MM = 7 * ROW_H_MM
IR_COUNTS = list(range(3, 10))
ACE_EXTENT_NORMALIZATION = "partial"
LAYOUT_RECT = [0, 0, 0.84, 1]
CBAR_GAP = 0.01
CBAR_WIDTH = 0.012

MapKind = Literal[
    "global_min_pop", "oxidation", "cluster", "ir_motif", "o_motif", "inertia", "homo_lumo",
]

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


@dataclass(frozen=True)
class MapSpec:
    kind: MapKind
    filename: str
    cbar_label: str | None = None


MAP_SPECS: tuple[MapSpec, ...] = (
    MapSpec("oxidation", "supp_ir_size_oxidation_maps.svg", r"$n_O/n_{Ir}$"),
    MapSpec("cluster", "supp_ir_size_cluster_maps.svg", "Cluster"),
    MapSpec("ir_motif", "supp_ir_size_ir_motif_maps.svg"),
    MapSpec("o_motif", "supp_ir_size_o_motif_maps.svg"),
    MapSpec("inertia", "supp_ir_size_inertia_maps.svg", "Shape"),
    MapSpec("homo_lumo", "supp_ir_size_homo_lumo_maps.svg", "HOMO–LUMO gap (eV)"),
    MapSpec("global_min_pop", "supp_ir_size_global_min_pop_maps.svg", "Population (%)"),
)


@dataclass
class DatasetBundle:
    label: str
    jt: dict
    inertia_rgb: np.ndarray
    gaps: np.ndarray
    k_struct: int
    k_Ir: int
    k_O: int


def _load_dataset(ds: dict) -> DatasetBundle | None:
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
    return DatasetBundle(
        label=ds["label"],
        jt=jt,
        inertia_rgb=inertia_rgb,
        gaps=homo_lumo_gaps(structures),
        k_struct=int(jt["cluster"].max()) + 1,
        k_Ir=jt["ir_frac"].shape[1],
        k_O=jt["o_frac"].shape[1],
    )


def _scalar_limits(panels: list[dict], field_key: str) -> tuple[float, float]:
    vals: list[float] = []
    for panel in panels:
        Z = np.asarray(panel[field_key], dtype=float)
        finite = Z[np.isfinite(Z)]
        if finite.size:
            vals.extend([float(finite.min()), float(finite.max())])
    if not vals:
        return 0.0, 1.0
    vmin, vmax = float(min(vals)), float(max(vals))
    if vmax <= vmin:
        vmax = vmin + 1.0
    return vmin, vmax


def _panel_payload(
    bundle: DatasetBundle,
    n_ir: int,
    kind: MapKind,
) -> dict | None:
    jt_sub = subset_joined_table_by_n_ir(bundle.jt, n_ir)
    if jt_sub is None:
        return None

    mask_idx = np.where(bundle.jt["n_Ir"] == n_ir)[0]
    inertia_sub = bundle.inertia_rgb[mask_idx]

    base = {
        "Ts": jt_sub["Ts"],
        "Ps": jt_sub["Ps"],
        "phase_index": jt_sub["phase_index"],
        "x_per_non_o": jt_sub["x_per_non_o"],
        "stoich_labels": stoichiometry_overlay_labels(jt_sub["formulas_gc"]),
    }

    if kind == "global_min_pop":
        base["Pgi"] = jt_sub["Pgi"]
        base["min_pop_field"] = global_minimum_population_map({
            "Pgi": jt_sub["Pgi"],
            "phase_index": jt_sub["phase_index"],
        })
        return base

    if kind == "oxidation":
        return base

    if kind == "cluster":
        base["pop"] = cluster_population_fractions(jt_sub)
        return base

    if kind == "ir_motif":
        base["pop"] = ir_motif_population_fractions(jt_sub)
        return base

    if kind == "o_motif":
        base["pop"] = o_motif_population_fractions(jt_sub)
        return base

    if kind == "inertia":
        base["rgb"] = ensemble_weighted_rgb_map(jt_sub["Pgi"], inertia_sub)
        return base

    if kind == "homo_lumo":
        gaps_sub = bundle.gaps[mask_idx]
        base["gap_field"] = homo_lumo_gap_map(jt_sub["Pgi"], gaps_sub)
        return base

    raise ValueError(f"Unknown map kind: {kind}")


def _draw_scalar_panel(
    ax,
    panel: dict,
    field_key: str,
    cmap: str,
    vmin: float,
    vmax: float,
    *,
    overlay: bool = False,
):
    Ts, Ps = panel["Ts"], panel["Ps"]
    Z = np.asarray(panel[field_key], dtype=float)
    mesh = ax.pcolormesh(
        Ts, Ps, Z, shading="auto",
        cmap=cmap,
        norm=Normalize(vmin=vmin, vmax=vmax),
        rasterized=True,
    )
    if overlay:
        draw_global_minimum_phase_overlay(
            ax, Ts, Ps, panel["phase_index"], panel["stoich_labels"],
        )
    return mesh


def _add_discrete_cbar(
    fig,
    axes: np.ndarray,
    colors: list,
    labels: list[str],
    cbar_label: str | None,
) -> None:
    cax = margin_colorbar_cax(
        fig, axes[0, 0], axes[-1, 1],
        gap=CBAR_GAP, width=CBAR_WIDTH,
    )
    discrete_category_colorbar(colors, labels, cax=cax, cbar_label=cbar_label)


def _build_map_grid(
    bundles: dict[str, DatasetBundle],
    kind: MapKind,
) -> dict[tuple[int, str], dict]:
    grid: dict[tuple[int, str], dict] = {}
    for label, bundle in bundles.items():
        for n_ir in IR_COUNTS:
            panel = _panel_payload(bundle, n_ir, kind)
            if panel is not None:
                grid[(n_ir, label)] = panel
    return grid


def _render_map(spec: MapSpec, bundles: dict[str, DatasetBundle]) -> None:
    grid = _build_map_grid(bundles, spec.kind)
    if not grid:
        raise RuntimeError(f"[supp] No Ir-size panels for {spec.kind}")

    fig, axes = plt.subplots(
        len(IR_COUNTS), 2,
        figsize=(FIG_W_MM / 25.4, FIG_H_MM / 25.4),
        dpi=DPI,
        sharex=True,
        sharey=True,
    )

    ref_bundle = next(iter(bundles.values()))
    last_row = len(IR_COUNTS) - 1

    if spec.kind in ("global_min_pop", "oxidation", "homo_lumo"):
        field_key = {
            "global_min_pop": "min_pop_field",
            "oxidation": "x_per_non_o",
            "homo_lumo": "gap_field",
        }[spec.kind]
        if spec.kind == "global_min_pop":
            cmap, vmin, vmax = "PuBu", 0.0, 100.0
        elif spec.kind == "oxidation":
            cmap = PHASE_CMAP
            vmin, vmax = _scalar_limits(list(grid.values()), field_key)
        else:
            cmap = HOMO_LUMO_CMAP
            vmin, vmax = _scalar_limits(list(grid.values()), field_key)
        mesh_ref = None
        for row, n_ir in enumerate(IR_COUNTS):
            for col, label in enumerate(("PBE", "PBE0")):
                ax = axes[row, col]
                panel = grid.get((n_ir, label))
                if panel is None:
                    ax.text(0.5, 0.5, "No data", ha="center", va="center",
                            fontsize=AXIS_LABEL_FONTSIZE, transform=ax.transAxes)
                    ax.set_axis_off()
                    continue
                title = label if row == 0 else ""
                if spec.kind == "global_min_pop":
                    mesh = draw_global_minimum_population_panel(
                        ax,
                        {
                            "Ts": panel["Ts"],
                            "Ps": panel["Ps"],
                            "Pgi": panel["Pgi"],
                            "phase_index": panel["phase_index"],
                        },
                        panel["stoich_labels"],
                    )
                else:
                    mesh = _draw_scalar_panel(
                        ax, panel, field_key, cmap, vmin, vmax,
                        overlay=(spec.kind == "oxidation"),
                    )
                style_ir_size_panel_ax(ax, n_ir, col, row, last_row, title=title)
                if mesh_ref is None and mesh is not None:
                    mesh_ref = mesh

        fig.tight_layout(pad=6 / 25.4, rect=LAYOUT_RECT)
        if mesh_ref is not None and spec.cbar_label:
            add_phase_colorbar(
                fig, mesh_ref,
                cax=margin_colorbar_cax(
                    fig, axes[0, 0], axes[-1, 1],
                    gap=CBAR_GAP, width=CBAR_WIDTH,
                ),
                label=spec.cbar_label,
            )
    else:
        struct_colors = cluster_colors(ref_bundle.k_struct)
        ir_colors = ir_motif_colors(ref_bundle.k_Ir)
        o_colors = o_motif_colors(ref_bundle.k_O)
        color_lookup = {
            "cluster": struct_colors,
            "ir_motif": ir_colors,
            "o_motif": o_colors,
        }
        cluster_majority_labels = [f"C{c}" for c in range(ref_bundle.k_struct)]

        for row, n_ir in enumerate(IR_COUNTS):
            for col, label in enumerate(("PBE", "PBE0")):
                ax = axes[row, col]
                panel = grid.get((n_ir, label))
                if panel is None:
                    ax.text(0.5, 0.5, "No data", ha="center", va="center",
                            fontsize=AXIS_LABEL_FONTSIZE, transform=ax.transAxes)
                    ax.set_axis_off()
                    continue

                title = label if row == 0 else ""

                if spec.kind == "inertia":
                    draw_shape_extent_rgb_phase_panel(
                        ax, panel["Ts"], panel["Ps"], panel["rgb"], "",
                    )
                else:
                    majority_labels = cluster_majority_labels if spec.kind == "cluster" else None
                    draw_population_mixed_phase_panel(
                        ax, panel["Ts"], panel["Ps"], panel["pop"],
                        color_lookup[spec.kind], "",
                        majority_labels=majority_labels,
                    )

                style_ir_size_panel_ax(ax, n_ir, col, row, last_row, title=title)

        fig.tight_layout(pad=6 / 25.4, rect=LAYOUT_RECT)

        if spec.kind == "cluster":
            _add_discrete_cbar(
                fig, axes, struct_colors,
                [f"C{c}" for c in range(ref_bundle.k_struct)],
                spec.cbar_label,
            )
        elif spec.kind == "ir_motif":
            _add_discrete_cbar(
                fig, axes, ir_colors,
                [ace_atom_display_label("Ir", c) for c in range(ref_bundle.k_Ir)],
                spec.cbar_label,
            )
        elif spec.kind == "o_motif":
            _add_discrete_cbar(
                fig, axes, o_colors,
                [ace_atom_display_label("O", c) for c in range(ref_bundle.k_O)],
                spec.cbar_label,
            )
        elif spec.kind == "inertia":
            _add_discrete_cbar(
                fig, axes,
                ["#e85c5c", "#5ce85c", "#5c5ce8"],
                ["L", "P", "S"],
                spec.cbar_label,
            )

    out_path = os.path.join(OUTPUT_DIR, spec.filename)
    save_fig(fig, out_path, dpi=DPI)
    print(f"[supp] {out_path}")


def main() -> None:
    ensure_for_supplementary(["supp_ir_size_gc_maps"])
    ensure_dirs(OUTPUT_DIR)

    bundles: dict[str, DatasetBundle] = {}
    for ds in DATASETS:
        bundle = _load_dataset(ds)
        if bundle is not None:
            bundles[bundle.label] = bundle

    if not bundles:
        raise RuntimeError("[supp] No GC datasets available for Ir-size maps.")

    for spec in MAP_SPECS:
        _render_map(spec, bundles)


if __name__ == "__main__":
    main()
