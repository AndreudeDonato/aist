"""Supplementary – vib vs no-vib oxidation maps per Ir size, PBE | PBE0.

Rows Ir₄…Ir₆ × columns PBE (+ vib) | PBE (no vib) | PBE0 (+ vib) | PBE0 (no vib): ensemble
``⟨n_O/n_Ir⟩`` maps restricted to one ``n_Ir`` bucket per row, one colour scale.

Output: ``figures/outputs/supp_ir_size_vib_vs_novib_maps.svg``
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

from figures.config import (
    DPI,
    OUTPUT_DIR,
    PHASE_CMAP,
    THERMO_IR_COUNTS,
)
from figures.utils.comparison import build_comparison_joined_table, load_gc_comparison_view
from figures.utils.cache import ensure_for_supplementary
from figures.utils.datasets import DATASETS, dataset_inputs
from figures.utils.gc_subset import oxidation_map_by_n_ir
from figures.utils.io import ensure_dirs, load_ace_features, load_npz
from figures.utils.plotting import (
    AXIS_LABEL_FONTSIZE,
    add_phase_colorbar,
    draw_global_minimum_phase_overlay,
    margin_colorbar_cax,
    save_fig,
    stoichiometry_overlay_labels,
    style_ir_size_panel_ax,
)

OUTPUT_PATH = os.path.join(OUTPUT_DIR, "supp_ir_size_vib_vs_novib_maps.svg")
FIG_W_MM = 180
ROW_H_MM = 38
FIG_H_MM = len(THERMO_IR_COUNTS) * ROW_H_MM
IR_COUNTS = THERMO_IR_COUNTS
LAYOUT_RECT = [0, 0, 0.92, 1]
CBAR_GAP = 0.01
CBAR_WIDTH = 0.012

COLUMN_SPECS = (
    ("full_Pgi", "+ vib"),
    ("novib_Pgi", "no vib"),
)


def _scalar_limits(panels: list[dict]) -> tuple[float, float]:
    vals: list[float] = []
    for panel in panels:
        Z = np.asarray(panel["x_per_non_o"], dtype=float)
        finite = Z[np.isfinite(Z)]
        if finite.size:
            vals.extend([float(finite.min()), float(finite.max())])
    if not vals:
        return 0.0, 1.0
    vmin, vmax = float(min(vals)), float(max(vals))
    if vmax <= vmin:
        vmax = vmin + 1.0
    return vmin, vmax


def _dataset_grid(label: str) -> dict[tuple[int, str], dict]:
    inp = dataset_inputs(label)
    vd = load_npz(
        inp.variants_npz,
        keys=("Ts", "Ps", "stoich", "full_Pgi", "novib_Pgi"),
    )
    Ts, Ps = vd["Ts"], vd["Ps"]
    stoich = np.asarray(vd["stoich"], dtype=float)

    gc_ref = load_gc_comparison_view(label, light=True)
    rows = load_ace_features(inp.ace_features)
    n_gc = min(
        len(gc_ref["formulas"]),
        vd["full_Pgi"].shape[0],
        vd["novib_Pgi"].shape[0],
    )
    jt = build_comparison_joined_table(
        {**gc_ref, "Pgi": gc_ref["Pgi"][:n_gc]},
        rows,
        label,
    )
    n_ir = np.asarray(jt["n_Ir"], dtype=int)
    formulas = np.asarray(jt["formulas_gc"], dtype=object)

    grid: dict[tuple[int, str], dict] = {}
    for n_ir_target in IR_COUNTS:
        for pgi_key, col_label in COLUMN_SPECS:
            panel = oxidation_map_by_n_ir(
                Ts, Ps, vd[pgi_key][:n_gc], stoich[:n_gc], n_ir, n_ir_target,
                formulas=formulas,
            )
            if panel is not None:
                panel["stoich_labels"] = stoichiometry_overlay_labels(panel["formulas"])
                grid[(n_ir_target, f"{label} ({col_label})")] = panel
    return grid


def main() -> None:
    ensure_for_supplementary(["supp_ir_size_vib_vs_novib_maps"])
    ensure_dirs(OUTPUT_DIR)

    grid: dict[tuple[int, str], dict] = {}
    for label in DATASETS:
        grid.update(_dataset_grid(label))
    if not grid:
        raise RuntimeError("[supp] No Ir-size panels for vib vs novib maps.")

    vmin, vmax = _scalar_limits(list(grid.values()))
    col_labels = [f"{label} ({col_label})" for label in DATASETS for _, col_label in COLUMN_SPECS]

    fig, axes = plt.subplots(
        len(IR_COUNTS), len(col_labels),
        figsize=(FIG_W_MM / 25.4, FIG_H_MM / 25.4),
        dpi=DPI,
        sharex=True,
        sharey=True,
    )

    mesh_ref = None
    last_row = len(IR_COUNTS) - 1

    for row, n_ir_target in enumerate(IR_COUNTS):
        for col, col_label in enumerate(col_labels):
            ax = axes[row, col]
            panel = grid.get((n_ir_target, col_label))
            if panel is None:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        fontsize=AXIS_LABEL_FONTSIZE, transform=ax.transAxes)
                ax.set_axis_off()
                continue

            Ts_p, Ps_p = panel["Ts"], panel["Ps"]
            Z = np.asarray(panel["x_per_non_o"], dtype=float)
            mesh = ax.pcolormesh(
                Ts_p, Ps_p, Z, shading="auto",
                cmap=PHASE_CMAP,
                norm=Normalize(vmin=vmin, vmax=vmax),
                rasterized=True,
            )
            draw_global_minimum_phase_overlay(
                ax, Ts_p, Ps_p, panel["phase_index"], panel["stoich_labels"],
            )
            if mesh_ref is None:
                mesh_ref = mesh

            title = col_label if row == 0 else ""
            style_ir_size_panel_ax(ax, n_ir_target, col, row, last_row, title=title)

    fig.tight_layout(pad=6 / 25.4, rect=LAYOUT_RECT)
    if mesh_ref is not None:
        add_phase_colorbar(
            fig, mesh_ref,
            cax=margin_colorbar_cax(
                fig, axes[0, -1], axes[-1, -1],
                gap=CBAR_GAP, width=CBAR_WIDTH,
            ),
            label=r"$n_O/n_{Ir}$",
        )

    save_fig(fig, OUTPUT_PATH, dpi=DPI)
    print(f"[supp] {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
