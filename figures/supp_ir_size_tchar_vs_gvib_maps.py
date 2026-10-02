"""Supplementary – per-Ir Θ_eff and ⟨G_vib⟩ maps, PBE | PBE0.

Rows Ir₄…Ir₆ × columns PBE ⟨Θ_eff⟩ | PBE ⟨G_vib⟩ | PBE0 ⟨Θ_eff⟩ | PBE0 ⟨G_vib⟩ (one colour scale per
quantity), matching other per-Ir supplementary grids. Θ_eff columns follow Figure 8's characteristic-
temperature panel; G_vib columns follow Figure 8's vibrational Gibbs contribution map.

Output: ``figures/outputs/supp_ir_size_tchar_vs_gvib_maps.svg``
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

from figures.config import (
    DPI,
    OUTPUT_DIR,
    THERMO_IR_COUNTS,
)
from figures.utils.cache import ensure_for_supplementary
from figures.utils.datasets import DATASETS, dataset_inputs, gc_structures
from figures.utils.io import ensure_dirs, load_gc_npz
from figures.utils.plotting import (
    AXIS_LABEL_FONTSIZE,
    CBAR_LABEL_FONTSIZE,
    TICK_FONTSIZE,
    add_phase_colorbar,
    draw_global_minimum_phase_overlay,
    draw_vib_reference_contours,
    save_fig,
    stoichiometry_overlay_labels,
    style_ir_size_panel_ax,
)
from figures.utils.validation import check_vib_availability
from figures.utils.vib_contrib_by_ir import (
    _n_ir_n_o,
    compute_per_structure_e_vib_tensor,
    e_vib_avg_map_by_n_ir,
    tchar_map_by_n_ir,
)

OUTPUT_PATH = os.path.join(OUTPUT_DIR, "supp_ir_size_tchar_vs_gvib_maps.svg")
FIG_W_MM = 180
ROW_H_MM = 38
FIG_H_MM = len(THERMO_IR_COUNTS) * ROW_H_MM
IR_COUNTS = THERMO_IR_COUNTS
LAYOUT_RECT = [0, 0, 0.91, 1]
CBAR_GAP = 0.01
CBAR_WIDTH = 0.012
CBAR_STACK_GAP = 0.025
CBAR_STACK_HEIGHT_FRAC = 0.50

COLUMN_SPECS = (
    ("tchar", r"$\langle\Theta_{\mathrm{eff}}(T)\rangle$"),
    ("gvib", r"$\langle G_{vib}\rangle$ (eV)"),
)


def _stacked_margin_colorbar_caxes(fig, ax_top, ax_bottom_right) -> list:
    """Two colorbars stacked vertically in the right margin (Θ_eff above G_vib)."""
    pos_top = ax_top.get_position()
    pos_bot = ax_bottom_right.get_position()
    x = pos_bot.x1 + CBAR_GAP
    y0 = pos_bot.y0
    total_h = pos_top.y1 - pos_bot.y0
    stack_h = total_h * CBAR_STACK_HEIGHT_FRAC
    cbar_h = (stack_h - CBAR_STACK_GAP) / 2
    stack_y0 = y0 + (total_h - stack_h) / 2
    return [
        fig.add_axes([x, stack_y0 + cbar_h + CBAR_STACK_GAP, CBAR_WIDTH, cbar_h]),
        fig.add_axes([x, stack_y0, CBAR_WIDTH, cbar_h]),
    ]


def _tchar_limits(panels: list[dict]) -> tuple[float, float]:
    vals: list[float] = []
    for panel in panels:
        Z = np.asarray(panel["tchar_map"], dtype=float)
        finite = Z[np.isfinite(Z)]
        if finite.size:
            vals.extend([float(finite.min()), float(finite.max())])
    if not vals:
        return 0.0, 1.0
    return float(min(vals)), float(max(vals))


def _gvib_vmax(panels: list[dict]) -> float:
    vals: list[float] = []
    for panel in panels:
        Z = np.asarray(panel["e_vib_avg"], dtype=float)
        finite = Z[np.isfinite(Z)]
        if finite.size:
            vals.extend(np.abs(finite).tolist())
    if not vals:
        return 1.0
    return float(np.nanpercentile(vals, 97))


def _dataset_grid(label: str) -> dict[tuple[int, str], dict]:
    """{(n_Ir, "tchar" | "gvib"): panel} of one functional."""
    inp = dataset_inputs(label)
    gc = load_gc_npz(inp.gc_npz)
    structures = gc_structures(label)
    if len(structures) != gc["Pgi"].shape[0]:
        raise ValueError(f"{label}: {len(structures)} structures but {gc['Pgi'].shape[0]} GC rows")
    Pgi = np.asarray(gc["Pgi"], dtype=float)
    Ts, Ps = gc["Ts"], gc["Ps"]
    formulas = np.asarray(gc["formulas"], dtype=object)
    n_ir = np.array([_n_ir_n_o(s)[0] for s in structures], dtype=int)

    check_vib_availability(structures, label, required=True)

    print(f"[supp] {label}: computing per-structure E_vib tensor …")
    e_vib_tensor = compute_per_structure_e_vib_tensor(
        structures, Ts, Ps, config=inp.grid_config(include_decomposition=True), o2_path=inp.o2_ref,
    )

    grid: dict[tuple[int, str], dict] = {}
    for n_ir_target in IR_COUNTS:
        tchar_panel = tchar_map_by_n_ir(
            structures, Pgi, Ts, Ps, n_ir, n_ir_target, formulas=formulas, degeneracy=gc["degeneracy"],
        )
        if tchar_panel is not None:
            grid[(n_ir_target, "tchar")] = tchar_panel

        gvib_panel = e_vib_avg_map_by_n_ir(
            e_vib_tensor, Pgi, Ts, Ps, n_ir, n_ir_target, formulas=formulas, degeneracy=gc["degeneracy"],
        )
        if gvib_panel is not None:
            gvib_panel["stoich_labels"] = stoichiometry_overlay_labels(gvib_panel["formulas"])
            grid[(n_ir_target, "gvib")] = gvib_panel
    return grid


def main() -> None:
    ensure_for_supplementary(["supp_ir_size_tchar_vs_gvib_maps"])
    ensure_dirs(OUTPUT_DIR)

    grids = {label: _dataset_grid(label) for label in DATASETS}
    if not any(grids.values()):
        raise RuntimeError("[supp] No Ir-size panels for tchar vs G_vib maps.")

    tchar_panels = [p for g in grids.values() for (n, k), p in g.items() if k == "tchar"]
    gvib_panels = [p for g in grids.values() for (n, k), p in g.items() if k == "gvib"]
    tchar_vmin, tchar_vmax = _tchar_limits(tchar_panels)
    gvib_vmax = _gvib_vmax(gvib_panels)
    gvib_norm = Normalize(vmin=-gvib_vmax, vmax=gvib_vmax)

    columns = [(label, key, f"{label}: {title}") for label in DATASETS for key, title in COLUMN_SPECS]
    fig, axes = plt.subplots(
        len(IR_COUNTS), len(columns),
        figsize=(FIG_W_MM / 25.4, FIG_H_MM / 25.4),
        dpi=DPI,
        sharex=True,
        sharey=True,
    )

    tchar_mesh_ref = None
    gvib_mesh_ref = None
    last_row = len(IR_COUNTS) - 1

    for row, n_ir_target in enumerate(IR_COUNTS):
        for col, (label, col_key, col_label) in enumerate(columns):
            ax = axes[row, col]
            grid = grids[label]
            panel = grid.get((n_ir_target, col_key))
            if panel is None:
                ax.text(
                    0.5, 0.5, "No data", ha="center", va="center",
                    fontsize=AXIS_LABEL_FONTSIZE, transform=ax.transAxes,
                )
                ax.set_axis_off()
                continue

            Ts_p, Ps_p = panel["Ts"], panel["Ps"]
            paired = grid.get((n_ir_target, "gvib" if col_key == "tchar" else "tchar"))
            if col_key == "tchar":
                Z = np.asarray(panel["tchar_map"], dtype=float)
                mesh = ax.pcolormesh(
                    Ts_p, Ps_p, Z, shading="auto",
                    cmap="plasma",
                    norm=Normalize(vmin=tchar_vmin, vmax=tchar_vmax),
                    rasterized=True,
                )
                gvib_map = paired["e_vib_avg"] if paired is not None else None
                draw_vib_reference_contours(
                    ax, Ts_p, Ps_p, tchar_map=Z, gvib_map=gvib_map,
                )
                if tchar_mesh_ref is None:
                    tchar_mesh_ref = mesh
            else:
                Z = np.asarray(panel["e_vib_avg"], dtype=float)
                mesh = ax.pcolormesh(
                    Ts_p, Ps_p, Z, shading="auto",
                    cmap="BrBG", norm=gvib_norm, rasterized=True,
                )
                draw_global_minimum_phase_overlay(
                    ax, Ts_p, Ps_p, panel["phase_index"], panel["stoich_labels"],
                )
                tchar_map = paired["tchar_map"] if paired is not None else None
                draw_vib_reference_contours(
                    ax, Ts_p, Ps_p, tchar_map=tchar_map, gvib_map=Z,
                )
                if gvib_mesh_ref is None:
                    gvib_mesh_ref = mesh

            title = col_label if row == 0 else ""
            style_ir_size_panel_ax(ax, n_ir_target, col, row, last_row, title=title)

    fig.tight_layout(pad=6 / 25.4, rect=LAYOUT_RECT)

    caxes = _stacked_margin_colorbar_caxes(fig, axes[0, 0], axes[-1, -1])
    if tchar_mesh_ref is not None:
        add_phase_colorbar(
            fig, tchar_mesh_ref,
            cax=caxes[0],
            label=r"$\langle\Theta_{\mathrm{eff}}\rangle$ (K)",
            label_fontsize=CBAR_LABEL_FONTSIZE,
            tick_fontsize=TICK_FONTSIZE,
        )
    if gvib_mesh_ref is not None:
        add_phase_colorbar(
            fig, gvib_mesh_ref,
            cax=caxes[1],
            label="eV",
            label_fontsize=CBAR_LABEL_FONTSIZE,
            tick_fontsize=TICK_FONTSIZE,
        )

    save_fig(fig, OUTPUT_PATH, dpi=DPI)
    print(f"[supp] {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
