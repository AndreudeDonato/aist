"""Supplementary – Same as Fig. 9 for all Ir sizes (PBE).

7 rows × 2 columns (Ir₃…Ir₉), PiYG divergent Δ(n_O/n_Ir) maps:

  Left:  full (vib + ZPE + ensemble) − no vib/ZPE, single minimum
  Right: full (vib + ZPE + ensemble) − no vib/ZPE, with ensemble

Layout matches ``supp_ir_size_vib_vs_novib_maps``; content matches Fig. 9.

Output: ``figures/outputs/supp_ir_size_oxidation_delta_maps.svg``
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from figures.config import (
    ACE_FEATURES,
    DPI,
    GC_VARIANTS_NPZ,
    OUTPUT_DIR,
    SUPP_IR_COUNTS,
)
from figures.utils.cache import ensure_for_supplementary
from figures.utils.comparison import build_comparison_joined_table, load_gc_comparison_view
from figures.utils.gc_subset import oxidation_map_by_n_ir, stoich_winner_map_by_n_ir
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

OUTPUT_PATH = os.path.join(OUTPUT_DIR, "supp_ir_size_oxidation_delta_maps.svg")
FIG_W_MM = 180
ROW_H_MM = 32
IR_COUNTS = list(SUPP_IR_COUNTS)
FIG_H_MM = len(IR_COUNTS) * ROW_H_MM
LAYOUT_RECT = [0, 0, 0.90, 1]
CBAR_GAP = 0.01
CBAR_WIDTH = 0.012
DELTA_CMAP = "PiYG"

COLUMN_TITLES = (
    r"Full − no vib/ZPE, single min",
    r"Full − no vib/ZPE, ensemble",
)


def _symmetric_limit(deltas: list[np.ndarray]) -> float:
    peak = 0.0
    for z in deltas:
        finite = np.asarray(z, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size:
            peak = max(peak, float(np.max(np.abs(finite))))
    if peak <= 0.0:
        return 1.0
    return peak


def main() -> None:
    ensure_for_supplementary(["supp_ir_size_oxidation_delta_maps"])
    ensure_dirs(OUTPUT_DIR)

    vd = load_npz(
        GC_VARIANTS_NPZ,
        keys=("Ts", "Ps", "stoich", "full_Pgi", "novib_Pgi"),
    )
    Ts, Ps = vd["Ts"], vd["Ps"]
    stoich = np.asarray(vd["stoich"], dtype=float)

    gc_ref = load_gc_comparison_view("PBE", light=True)
    rows = load_ace_features(ACE_FEATURES["PBE"])
    n_gc = min(
        len(gc_ref["formulas"]),
        vd["full_Pgi"].shape[0],
        vd["novib_Pgi"].shape[0],
    )
    jt = build_comparison_joined_table(
        {**gc_ref, "Pgi": gc_ref["Pgi"][:n_gc]},
        rows,
        "PBE",
    )
    n_ir = np.asarray(jt["n_Ir"], dtype=int)
    formulas = np.asarray(jt["formulas_gc"], dtype=object)
    full_Pgi = vd["full_Pgi"][:n_gc]
    novib_Pgi = vd["novib_Pgi"][:n_gc]
    stoich_n = stoich[:n_gc]

    panels: dict[tuple[int, int], dict] = {}
    for n_ir_target in IR_COUNTS:
        full = oxidation_map_by_n_ir(
            Ts, Ps, full_Pgi, stoich_n, n_ir, n_ir_target, formulas=formulas,
        )
        novib_ens = oxidation_map_by_n_ir(
            Ts, Ps, novib_Pgi, stoich_n, n_ir, n_ir_target, formulas=formulas,
        )
        novib_min = stoich_winner_map_by_n_ir(
            Ts, Ps, novib_Pgi, stoich_n, n_ir, n_ir_target, formulas=formulas,
        )
        if full is None or novib_ens is None or novib_min is None:
            continue

        labels = stoichiometry_overlay_labels(full["formulas"])
        z_full = np.asarray(full["x_per_non_o"], dtype=float)
        panels[(n_ir_target, 0)] = {
            "Ts": full["Ts"],
            "Ps": full["Ps"],
            "delta": z_full - np.asarray(novib_min["stoich_map"], dtype=float),
            "phase_index": full["phase_index"],
            "stoich_labels": labels,
        }
        panels[(n_ir_target, 1)] = {
            "Ts": full["Ts"],
            "Ps": full["Ps"],
            "delta": z_full - np.asarray(novib_ens["x_per_non_o"], dtype=float),
            "phase_index": full["phase_index"],
            "stoich_labels": labels,
        }

    if not panels:
        raise RuntimeError("[supp] No Ir-size Δ oxidation panels.")

    vlim = _symmetric_limit([p["delta"] for p in panels.values()])
    norm = TwoSlopeNorm(vmin=-vlim, vcenter=0.0, vmax=vlim)

    fig, axes = plt.subplots(
        len(IR_COUNTS), 2,
        figsize=(FIG_W_MM / 25.4, FIG_H_MM / 25.4),
        dpi=DPI,
        sharex=True,
        sharey=True,
    )

    mesh_ref = None
    last_row = len(IR_COUNTS) - 1
    for row, n_ir_target in enumerate(IR_COUNTS):
        for col in range(2):
            ax = axes[row, col]
            panel = panels.get((n_ir_target, col))
            if panel is None:
                ax.text(
                    0.5, 0.5, "No data", ha="center", va="center",
                    fontsize=AXIS_LABEL_FONTSIZE, transform=ax.transAxes,
                )
                ax.set_axis_off()
                continue

            mesh = ax.pcolormesh(
                panel["Ts"], panel["Ps"], panel["delta"],
                shading="auto", cmap=DELTA_CMAP, norm=norm, rasterized=True,
            )
            draw_global_minimum_phase_overlay(
                ax, panel["Ts"], panel["Ps"],
                panel["phase_index"], panel["stoich_labels"],
            )
            if mesh_ref is None:
                mesh_ref = mesh

            title = COLUMN_TITLES[col] if row == 0 else ""
            style_ir_size_panel_ax(
                ax, n_ir_target, col, row, last_row, title=title,
            )

    fig.tight_layout(pad=6 / 25.4, rect=LAYOUT_RECT)
    if mesh_ref is not None:
        add_phase_colorbar(
            fig, mesh_ref,
            cax=margin_colorbar_cax(
                fig, axes[0, 0], axes[-1, 1],
                gap=CBAR_GAP, width=CBAR_WIDTH,
            ),
            label=r"$\Delta(n_O/n_{Ir})$",
        )

    save_fig(fig, OUTPUT_PATH, dpi=DPI)
    print(f"\n[supp] {OUTPUT_PATH}")
    print(f"  Δ colour scale: ±{vlim:.3g}")


if __name__ == "__main__":
    main()
