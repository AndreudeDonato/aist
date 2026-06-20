"""Supplementary – PBE vib vs no-vib oxidation maps per Ir size.

7 rows × 2 columns (Ir₃…Ir₉ × PBE with vib | PBE without vib), matching fig 7's
ensemble ``⟨n_O/n_Ir⟩`` maps but restricted to one ``n_Ir`` bucket per row.

Output: ``figures/outputs/supp_ir_size_vib_vs_novib_maps.svg``
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

from figures.config import (
    ACE_FEATURES,
    DPI,
    GC_NPZ,
    GC_VARIANTS_NPZ,
    OUTPUT_DIR,
    PHASE_CMAP,
)
from figures.utils.cache import ensure_for_supplementary
from figures.utils.gc_subset import oxidation_map_by_n_ir
from figures.utils.io import ensure_dirs, load_ace_features, load_gc_npz, load_npz
from figures.utils.plotting import (
    add_phase_colorbar,
    draw_global_minimum_phase_overlay,
    margin_colorbar_cax,
    save_fig,
    stoichiometry_overlay_labels,
    style_ir_size_panel_ax,
)

OUTPUT_PATH = os.path.join(OUTPUT_DIR, "supp_ir_size_vib_vs_novib_maps.svg")
FIG_W_MM = 180
ROW_H_MM = 32
FIG_H_MM = 7 * ROW_H_MM
IR_COUNTS = list(range(3, 10))
LAYOUT_RECT = [0, 0, 0.84, 1]
CBAR_GAP = 0.01
CBAR_WIDTH = 0.012

COLUMN_SPECS = (
    ("full_Pgi", "PBE (+ vib)"),
    ("novib_Pgi", "PBE (no vib)"),
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


def main() -> None:
    ensure_for_supplementary(["supp_ir_size_vib_vs_novib_maps"])
    ensure_dirs(OUTPUT_DIR)

    vd = load_npz(
        GC_VARIANTS_NPZ,
        keys=("Ts", "Ps", "stoich", "full_Pgi", "novib_Pgi"),
    )
    Ts, Ps = vd["Ts"], vd["Ps"]
    stoich = np.asarray(vd["stoich"], dtype=float)

    gc = load_gc_npz(GC_NPZ, light=True)
    formulas = np.asarray(gc["formulas"], dtype=object)
    rows = load_ace_features(ACE_FEATURES["PBE"])
    n = min(formulas.size, vd["full_Pgi"].shape[0], len(rows))
    formulas = formulas[:n]
    n_ir = np.array([r["n_Ir"] for r in rows[:n]], dtype=int)

    grid: dict[tuple[int, str], dict] = {}
    for n_ir_target in IR_COUNTS:
        for pgi_key, col_label in COLUMN_SPECS:
            panel = oxidation_map_by_n_ir(
                Ts, Ps, vd[pgi_key][:n], stoich[:n], n_ir, n_ir_target,
                formulas=formulas,
            )
            if panel is not None:
                panel["stoich_labels"] = stoichiometry_overlay_labels(panel["formulas"])
                grid[(n_ir_target, col_label)] = panel

    if not grid:
        raise RuntimeError("[supp] No Ir-size panels for vib vs novib maps.")

    vmin, vmax = _scalar_limits(list(grid.values()))

    fig, axes = plt.subplots(
        len(IR_COUNTS), 2,
        figsize=(FIG_W_MM / 25.4, FIG_H_MM / 25.4),
        dpi=DPI,
        sharex=True,
        sharey=True,
    )

    mesh_ref = None
    last_row = len(IR_COUNTS) - 1
    col_labels = [label for _, label in COLUMN_SPECS]

    for row, n_ir_target in enumerate(IR_COUNTS):
        for col, col_label in enumerate(col_labels):
            ax = axes[row, col]
            panel = grid.get((n_ir_target, col_label))
            if panel is None:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        fontsize=7, transform=ax.transAxes)
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
                fig, axes[0, 0], axes[-1, 1],
                gap=CBAR_GAP, width=CBAR_WIDTH,
            ),
            label=r"$n_O/n_{Ir}$",
        )

    save_fig(fig, OUTPUT_PATH, dpi=DPI)
    print(f"[supp] {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
