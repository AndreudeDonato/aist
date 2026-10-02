"""Supplementary – Global-minimum phase map + fixed-point population lines, PBE | PBE0.

Former Figure 9. Novib and full (vib+ZPE) populations come from each functional's variants NPZ
(pooled Ir₄–Ir₆ ensembles).

3×2 layout (180 mm width; columns PBE | PBE0):
  (0): Global-minimum structure population (%) on T × pO₂
  (1): At T=273 K, ambient pO₂ — s1, s2, … (97% novib population)
  (2): Same at T=500 K — s'1, s'2, …

Output: ``figures/outputs/supp_min_pop_fixedpoints.svg``
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from figures.config import (
    DPI,
    FIXED_POINTS,
    OUTPUT_DIR,
)
from figures.utils.cache import ensure_for_supplementary
from figures.utils.datasets import DATASETS, dataset_inputs
from figures.utils.io import ensure_dirs, load_gc_npz
from figures.utils.join import (
    ranked_cumulative_population_lines,
    structure_populations_at_fixed_point,
)
from figures.utils.plotting import (
    TITLE_FONTSIZE,
    AXIS_LABEL_FONTSIZE,
    TICK_FONTSIZE,
    LEGEND_FONTSIZE,
    CBAR_LABEL_FONTSIZE,
    draw_global_minimum_population_panel,
    save_fig,
    stoichiometry_overlay_labels,
)

OUTPUT_PATH = os.path.join(OUTPUT_DIR, "supp_min_pop_fixedpoints.svg")
FIG_W_MM = 180
FIG_H_MM = 165
CUMULATIVE_FRAC = 0.97


def _global_min_pop_panel(ax, gc, overlay_labels: list[str], *, title: str = "Global-minimum population") -> None:
    return draw_global_minimum_population_panel(
        ax, gc, overlay_labels,
        title=title,
        title_fontsize=TITLE_FONTSIZE,
    )


def _edge_to_edge_x(n: int) -> np.ndarray:
    """x in [0, 1] so the first/last points sit on the left/right axes edges."""
    if n <= 1:
        return np.zeros(max(n, 1), dtype=float)
    return np.linspace(0.0, 1.0, n, dtype=float)


def _pin_xlim_edge_to_edge(ax) -> None:
    ax.set_xlim(0.0, 1.0)
    ax.set_xmargin(0)
    ax.margins(x=0)


def _structure_tick_labels(n_show: int, *, primed: bool = False) -> list[str]:
    if primed:
        return [f"s'{k + 1}" for k in range(n_show)]
    return [f"s{k + 1}" for k in range(n_show)]


def _cumulative_population_panel(
    ax,
    pops_novib: np.ndarray,
    pops_vib: np.ndarray,
    *,
    T: float,
    P: float,
    cumulative: float = CUMULATIVE_FRAC,
    primed_ticks: bool = False,
) -> None:
    """Line plot: structures to ``cumulative`` total pop., ordered by novib (high→low)."""
    lines = ranked_cumulative_population_lines(
        pops_novib, pops_vib, cumulative=cumulative,
    )
    n_show = int(lines["indices"].size)
    if n_show == 0:
        ax.text(0.5, 0.5, "No population data", ha="center", va="center",
                fontsize=AXIS_LABEL_FONTSIZE, transform=ax.transAxes)
        return

    x = _edge_to_edge_x(n_show)
    ax.plot(x, lines["y_order"], color="tomato", linewidth=1.0,
            label="No vib", alpha=0.9, solid_capstyle="butt")
    ax.plot(x, lines["y_overlay"], color="steelblue", linewidth=1.0,
            label="With vib", alpha=0.9, solid_capstyle="butt")

    ax.set_xlabel("Lowest energy structures", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Population (%)", fontsize=AXIS_LABEL_FONTSIZE)
    cov = 100.0 * float(lines["cumulative_fraction"])
    ax.set_title(
        rf"$T={T:.0f}$ K, $p_{{\mathrm{{O_2}}}}={P:.3g}$ Pa"
        "\n"
        rf"{n_show} structures ({cov:.0f}% of novib pop.)",
        fontsize=TITLE_FONTSIZE,
    )
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    ax.legend(fontsize=LEGEND_FONTSIZE, frameon=False, loc="upper right")
    ymax = max(float(lines["y_order"].max()), float(lines["y_overlay"].max()), 0.5)
    ax.set_ylim(0, ymax * 1.1)
    _pin_xlim_edge_to_edge(ax)

    ax.set_xticks(x)
    ax.set_xticklabels(
        _structure_tick_labels(n_show, primed=primed_ticks),
        fontsize=TICK_FONTSIZE,
        rotation=45 if n_show > 12 else 0,
        ha="right" if n_show > 12 else "center",
    )


def main() -> None:
    ensure_for_supplementary(["supp_min_pop_fixedpoints"])
    ensure_dirs(OUTPUT_DIR)

    fig = plt.figure(figsize=(FIG_W_MM / 25.4, FIG_H_MM / 25.4), dpi=DPI)
    gs_main = GridSpec(
        3, len(DATASETS), figure=fig,
        height_ratios=[1.45, 1.0, 1.0],
        hspace=0.62,
        wspace=0.40,
    )
    for col, label in enumerate(DATASETS):
        inp = dataset_inputs(label)
        gc_full = load_gc_npz(inp.gc_npz)
        vd = np.load(inp.variants_npz, allow_pickle=True)
        Ts_v, Ps_v = vd["Ts"], vd["Ps"]
        novib_Pgi = vd["novib_Pgi"]
        full_Pgi = vd["full_Pgi"]
        overlay_labels = stoichiometry_overlay_labels(gc_full["formulas"])

        gs_top = gs_main[0, col].subgridspec(1, 2, width_ratios=[1.0, 0.035], wspace=0.04)
        ax_phase = fig.add_subplot(gs_top[0, 0])
        cax = fig.add_subplot(gs_top[0, 1])
        ax_lines = [
            fig.add_subplot(gs_main[1, col]),
            fig.add_subplot(gs_main[2, col]),
        ]

        mesh0 = _global_min_pop_panel(ax_phase, gc_full, overlay_labels,
                                      title=f"{label}: global-minimum population")
        if mesh0 is not None:
            cbar = fig.colorbar(mesh0, cax=cax)
            cbar.set_label("Population (%)", fontsize=CBAR_LABEL_FONTSIZE)
            cbar.ax.tick_params(labelsize=TICK_FONTSIZE)

        for i, fp in enumerate(FIXED_POINTS[:2]):
            T, P = fp["T"], fp["P"]
            pops_novib = structure_populations_at_fixed_point(novib_Pgi, Ts_v, Ps_v, T, P)
            pops_vib = structure_populations_at_fixed_point(full_Pgi, Ts_v, Ps_v, T, P)
            _cumulative_population_panel(
                ax_lines[i], pops_novib, pops_vib, T=T, P=P,
                primed_ticks=(i == 1),
            )
        del gc_full, vd

    fig.subplots_adjust(left=0.07, right=0.97, top=0.97, bottom=0.08)
    save_fig(fig, OUTPUT_PATH, dpi=DPI)
    print(f"[supp] {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
