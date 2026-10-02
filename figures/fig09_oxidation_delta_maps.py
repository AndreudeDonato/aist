"""Figure 9 – Per-Ir Δ(n_O/n_Ir) phase maps (thermo variants), PBE | PBE0.

180 mm wide, 3×4 (Ir₄|Ir₅|Ir₆ × two differences per functional), PiYG divergent, one colour scale:

  per functional — full (vib + ZPE + ensemble) − no vib/ZPE, single minimum
                 | full (vib + ZPE + ensemble) − no vib/ZPE, with ensemble

Layout follows ``supp_vib_vs_novib_quadrants`` / per-Ir vib maps.

Output: ``figures/outputs/fig09_oxidation_delta_maps.svg``
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from figures.config import (
    DPI,
    FIG04_IR_COUNTS,
    OUTPUT_DIR,
)
from figures.utils.cache import ensure_for_figures
from figures.utils.comparison import build_comparison_joined_table, load_gc_comparison_view
from figures.utils.datasets import DATASETS, dataset_inputs
from figures.utils.gc_subset import oxidation_map_by_n_ir, stoich_winner_map_by_n_ir
from figures.utils.io import ensure_dirs, load_ace_features, load_npz, load_variants_like
from figures.utils.plotting import (
    AXIS_LABEL_FONTSIZE,
    add_phase_colorbar,
    draw_global_minimum_phase_overlay,
    margin_colorbar_cax,
    save_fig,
    stoichiometry_overlay_labels,
    style_ir_size_panel_ax,
)

OUTPUT_PATH = os.path.join(OUTPUT_DIR, "fig09_oxidation_delta_maps.svg")
FIG_W_MM = 180
ROW_H_MM = 48
IR_COUNTS = FIG04_IR_COUNTS
FIG_H_MM = len(IR_COUNTS) * ROW_H_MM
LAYOUT_RECT = [0, 0, 0.93, 1]
CBAR_GAP = 0.012
CBAR_WIDTH = 0.014
DELTA_CMAP = "PiYG"

COLUMN_TITLES = (
    r"full − no vib, single min",
    r"full − no vib, ensemble",
)   # shown under the functional's name (two-line column titles)


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


def _dataset_panels(label: str) -> dict[tuple[int, int], dict]:
    """{(n_Ir, difference index): panel} of one functional."""
    inp = dataset_inputs(label)
    gc_ref = load_gc_comparison_view(label, light=True)
    vd = load_variants_like(inp.variants_npz, gc_ref, ("Ts", "Ps", "stoich", "full_Pgi", "novib_Pgi"))
    Ts, Ps = vd["Ts"], vd["Ps"]
    stoich = np.asarray(vd["stoich"], dtype=float)

    rows = load_ace_features(inp.ace_features)
    jt = build_comparison_joined_table(
        gc_ref,
        rows,
        label,
    )
    n_ir = np.asarray(jt["n_Ir"], dtype=int)
    formulas = np.asarray(jt["formulas_gc"], dtype=object)
    full_Pgi = vd["full_Pgi"]
    novib_Pgi = vd["novib_Pgi"]
    stoich_n = stoich

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
        for j, ref in enumerate((novib_min["stoich_map"], novib_ens["x_per_non_o"])):
            panels[(n_ir_target, j)] = {
                "Ts": full["Ts"],
                "Ps": full["Ps"],
                "delta": z_full - np.asarray(ref, dtype=float),
                "phase_index": full["phase_index"],
                "stoich_labels": labels,
            }
    return panels


def build(output_path: str = OUTPUT_PATH, *, row_h_mm: float = ROW_H_MM, tag: str = "fig09") -> None:
    """Draw the PBE | PBE0 Δ maps into ``output_path`` (also used by supp_ir_size_oxidation_delta_maps)."""
    ensure_dirs(OUTPUT_DIR)
    panels = {label: _dataset_panels(label) for label in DATASETS}
    if not any(panels.values()):
        raise RuntimeError(f"[{tag}] No Ir-size Δ oxidation panels.")

    vlim = _symmetric_limit([p["delta"] for d in panels.values() for p in d.values()])
    norm = TwoSlopeNorm(vmin=-vlim, vcenter=0.0, vmax=vlim)
    columns = [(label, j) for label in DATASETS for j in range(len(COLUMN_TITLES))]

    fig, axes = plt.subplots(
        len(IR_COUNTS), len(columns),
        figsize=(FIG_W_MM / 25.4, len(IR_COUNTS) * row_h_mm / 25.4),
        dpi=DPI,
        sharex=True,
        sharey=True,
    )

    mesh_ref = None
    last_row = len(IR_COUNTS) - 1
    for row, n_ir_target in enumerate(IR_COUNTS):
        for col, (label, j) in enumerate(columns):
            ax = axes[row, col]
            panel = panels[label].get((n_ir_target, j))
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

            title = f"{label}\n{COLUMN_TITLES[j]}" if row == 0 else ""
            style_ir_size_panel_ax(
                ax, n_ir_target, col, row, last_row, title=title,
            )

    fig.tight_layout(pad=6 / 25.4, rect=LAYOUT_RECT)
    if mesh_ref is not None:
        add_phase_colorbar(
            fig, mesh_ref,
            cax=margin_colorbar_cax(
                fig, axes[0, -1], axes[-1, -1],
                gap=CBAR_GAP, width=CBAR_WIDTH,
            ),
            label=r"$\Delta(n_O/n_{Ir})$",
        )

    save_fig(fig, output_path, dpi=DPI)
    print(f"\n[{tag}] {output_path}")
    print(f"  Δ colour scale: ±{vlim:.3g}")


def main() -> None:
    ensure_for_figures([9])
    build()


if __name__ == "__main__":
    main()
