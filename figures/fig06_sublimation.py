"""Figure 6 – Sublimation (PBE0).

4×1 layout (88 mm wide):
  Panel 1 — IrO₂ rutile + IrO₂(g) + IrO₃(g) GC phase map (PBE0)
  Panels 2–4 — PBE0 ensemble ⟨n_O/n_Ir⟩ gradient bars at atmospheric pO₂
                with gas reservoir at three support levels.

Output: ``figures/outputs/fig06_sublimation.svg``
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

from figures.config import (
    ACE_FEATURES,
    DPI,
    FIG06_PRESSURE_PA,
    FIG06_SUPPORT_VARIANTS,
    FIG06_T_MAX,
    FIG06_T_MIN,
    OUTPUT_DIR,
    PHASE_CMAP,
    RUTILE_GC_CALIBRATION_T_K,
    RUTILE_GC_NPZ,
    THERMO_IR_COUNTS,
)
from figures.utils.comparison import (
    build_comparison_joined_table,
    gas_population_fraction,
    gas_structure_mask,
    scalar_field_rgb_with_gas,
)
from figures.utils.cache import ensure_for_figures
from figures.utils.fig06_gc_cache import load_fig06_gc_view
from figures.utils.gc_gradient_bars import (
    draw_scalar_gradient_bars,
    ensemble_scalar_vs_T_at_pressure,
)
from figures.utils.io import ensure_dirs, load_ace_features, load_gc_npz
from figures.utils.plotting import (
    AXIS_LABEL_FONTSIZE,
    TICK_FONTSIZE,
    draw_global_minimum_phase_overlay,
    save_fig,
    stoichiometry_overlay_labels,
    style_phase_ax,
)

OUTPUT_PATH = os.path.join(OUTPUT_DIR, "fig06_sublimation.svg")
FIG_W_MM = 88
RUTILE_H_MM = 70
BAR_H_MM = 40
FIG_H_MM = RUTILE_H_MM + len(FIG06_SUPPORT_VARIANTS) * BAR_H_MM
N_PANELS = 1 + len(FIG06_SUPPORT_VARIANTS)
T_XLIM = (FIG06_T_MIN, FIG06_T_MAX)
GAS_PANEL_LABEL = r"IrO$_3$(g)"

ROW_TITLES = (
    "No support",
    "Weak support",
    "Strong support",
)


def _annotate_gas_label(ax) -> None:
    ax.text(
        0.98, 0.5, GAS_PANEL_LABEL,
        transform=ax.transAxes,
        ha="right", va="center",
        fontsize=AXIS_LABEL_FONTSIZE,
        color="black",
        zorder=10,
    )


def _style_fig06_rutile_ax(ax) -> None:
    style_phase_ax(ax, apply_gc_t_ticks=False)
    ax.tick_params(axis="y", labelleft=True, labelsize=TICK_FONTSIZE)
    ax.tick_params(axis="x", labelbottom=True, labelsize=TICK_FONTSIZE)


def _style_fig06_bar_ax(ax, *, title: str) -> None:
    ax.set_ylabel(title, fontsize=AXIS_LABEL_FONTSIZE)
    ax.tick_params(axis="y", labelleft=True, labelsize=TICK_FONTSIZE)
    ax.tick_params(axis="x", labelbottom=True, labelsize=TICK_FONTSIZE)
    ax.set_xlabel("T (K)", fontsize=AXIS_LABEL_FONTSIZE)


def _rutile_panel(
    ax,
    gc: dict,
    overlay_labels: list[str],
    vmin: float,
    vmax: float,
) -> None:
    Ts, Ps = gc["Ts"], gc["Ps"]
    Z = np.asarray(gc["x_per_non_o"], dtype=float)
    gas_mask = gas_structure_mask(gc["formulas"])
    f_gas = gas_population_fraction(gc["Pgi"], gas_mask)
    rgb = scalar_field_rgb_with_gas(
        Z, f_gas, cmap=PHASE_CMAP, vmin=vmin, vmax=vmax,
    )
    ax.pcolormesh(Ts, Ps, rgb, shading="auto", rasterized=True)
    draw_global_minimum_phase_overlay(
        ax, Ts, Ps, gc["phase_index"], overlay_labels,
    )
    ax.axhline(
        FIG06_PRESSURE_PA,
        color="black",
        linewidth=0.8,
        linestyle="--",
        zorder=9,
    )
    ax.plot(
        RUTILE_GC_CALIBRATION_T_K,
        FIG06_PRESSURE_PA,
        marker="x",
        color="black",
        markersize=4,
        markeredgewidth=0.8,
        linestyle="none",
        zorder=10,
    )
    ax.set_xlim(T_XLIM)


def _load_support_column(variant: dict) -> dict:
    gc = load_fig06_gc_view(
        ev_per_ir=float(variant["ev_per_ir"]),
        ev_per_o=float(variant["ev_per_o"]),
    )
    rows = load_ace_features(ACE_FEATURES["PBE0"])
    jt = build_comparison_joined_table(
        gc, rows, "PBE0", include_gas_reservoir=True,
    )
    n_ir = np.asarray(jt["n_Ir"], dtype=float)
    n_o = np.asarray(jt["n_O"], dtype=float)
    stoich = np.where(n_ir > 0, n_o / n_ir, np.nan)
    return {
        "label": variant["label"],
        "joined": jt,
        "stoich": stoich,
    }


def _oxidation_scale(rutile_gc: dict, columns: list[dict], pressure_pa: float) -> tuple[float, float]:
    """Shared oxidation colour scale: 0 … global max ``n_O/n_Ir`` across all panels."""
    vals: list[float] = [0.0]
    rutile_z = np.asarray(rutile_gc["x_per_non_o"], dtype=float)
    finite = rutile_z[np.isfinite(rutile_z)]
    if finite.size:
        vals.append(float(finite.max()))

    for col in columns:
        stoich = np.asarray(col["stoich"], dtype=float)
        finite = stoich[np.isfinite(stoich)]
        if finite.size:
            vals.append(float(finite.max()))
        for n in THERMO_IR_COUNTS:
            curve = ensemble_scalar_vs_T_at_pressure(
                col["joined"], n, col["stoich"], pressure_pa,
                include_gas_reservoir=True,
            )
            if curve is None:
                continue
            finite = curve[2][np.isfinite(curve[2])]
            if finite.size:
                vals.append(float(finite.max()))

    vmax = float(max(vals))
    if vmax <= 0.0:
        vmax = 3.0
    return 0.0, vmax


def main() -> None:
    ensure_for_figures([6])
    ensure_dirs(OUTPUT_DIR)

    rutile_gc = load_gc_npz(RUTILE_GC_NPZ["PBE0"], light=True)
    rutile_labels = stoichiometry_overlay_labels(rutile_gc["formulas"])
    support_cols = [_load_support_column(v) for v in FIG06_SUPPORT_VARIANTS]
    ox_vmin, ox_vmax = _oxidation_scale(rutile_gc, support_cols, FIG06_PRESSURE_PA)

    fig, axes = plt.subplots(
        N_PANELS, 1,
        figsize=(FIG_W_MM / 25.4, FIG_H_MM / 25.4),
        dpi=DPI,
        sharex=False,
        sharey=False,
    )
    if N_PANELS == 1:
        axes = np.array([axes])

    _rutile_panel(
        axes[0], rutile_gc, rutile_labels, ox_vmin, ox_vmax,
    )
    _style_fig06_rutile_ax(axes[0])

    for row, col in enumerate(support_cols, start=1):
        ax = axes[row]
        draw_scalar_gradient_bars(
            ax,
            col["joined"],
            FIG06_PRESSURE_PA,
            THERMO_IR_COUNTS,
            col["stoich"],
            cmap=PHASE_CMAP,
            vmin=ox_vmin,
            vmax=ox_vmax,
            section_kind="none",
            include_gas_reservoir=True,
            gas_white_mix=True,
        )
        _style_fig06_bar_ax(ax, title=ROW_TITLES[row - 1])
        _annotate_gas_label(ax)

    fig.tight_layout(pad=6 / 25.4)

    save_fig(fig, OUTPUT_PATH, dpi=DPI)
    print(f"\n[fig06] {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
