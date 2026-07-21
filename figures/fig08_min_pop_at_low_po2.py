"""Figure 8 – Novib vs vib ranked populations for Ir₃, Ir₆, Ir₉ (PBE).

3×2 layout (180 mm wide, short rectangular panels) at ambient ``pO₂``:

  Rows — Ir₃ | Ir₆ | Ir₉
  Columns — ~273 K | ~500 K

Each panel: population (%) of the five lowest-energy structures (s0…s4;
order fixed by novib ``G``). Blue = no vib; orange = with vib.

Shared ``x`` and ``y`` across all panels. Global y-label is Population (%);
each left-column panel is labelled Irₙ.

Output: ``figures/outputs/fig08_min_pop_at_low_po2.svg``
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

from figures.config import (
    ACE_FEATURES,
    DPI,
    FIG04_IR_COUNTS,
    FIG08_PRESSURE_PA,
    GC_NPZ,
    GC_VARIANTS_NPZ,
    OUTPUT_DIR,
)
from figures.utils.cache import ensure_for_figures
from figures.utils.gc_subset import ranked_structure_pops_by_ir_at_fixed_point
from figures.utils.io import ensure_dirs, load_ace_features, load_gc_npz, load_npz
from figures.utils.join import build_joined_table
from figures.utils.plotting import (
    AXIS_LABEL_FONTSIZE,
    LEGEND_FONTSIZE,
    TICK_FONTSIZE,
    TITLE_FONTSIZE,
    save_fig,
)

OUTPUT_PATH = os.path.join(OUTPUT_DIR, "fig08_min_pop_at_low_po2.svg")
FIG_W_MM = 180
ROW_H_MM = 28
FIG_H_MM = len(FIG04_IR_COUNTS) * ROW_H_MM
IR_COUNTS = FIG04_IR_COUNTS
TEMPERATURES = (273.0, 500.0)
N_STRUCTURES = 5
NOVIB_COLOR = "#1f77b4"  # blue
VIB_COLOR = "#ff7f0e"    # orange


def _edge_to_edge_x(n: int) -> np.ndarray:
    if n <= 1:
        return np.zeros(max(n, 1), dtype=float)
    return np.linspace(0.0, float(n - 1), n, dtype=float)


def _global_ymax(
    panels_by_t: dict[float, dict[int, tuple[np.ndarray, np.ndarray]]],
) -> float:
    ymax = 1.0
    for lines_by_ir in panels_by_t.values():
        for n_ir in IR_COUNTS:
            pair = lines_by_ir.get(n_ir)
            if pair is None:
                continue
            for pops in pair:
                if pops.size:
                    ymax = max(ymax, float(np.nanmax(pops[:N_STRUCTURES])))
    return ymax * 1.08


def _structure_tick_labels(*, primed: bool) -> list[str]:
    if primed:
        return [rf"$s'_{{{k}}}$" for k in range(N_STRUCTURES)]
    return [f"s{k}" for k in range(N_STRUCTURES)]


def _plot_ir_panel(
    ax,
    pops_novib: np.ndarray | None,
    pops_vib: np.ndarray | None,
    *,
    n_ir: int,
    ymax: float,
    show_xlabel: bool,
    show_ir_ylabel: bool,
    show_legend: bool,
    primed_ticks: bool = False,
    col_title: str = "",
) -> None:
    x = _edge_to_edge_x(N_STRUCTURES)
    if pops_novib is None or pops_vib is None:
        ax.text(
            0.5, 0.5, "No data", ha="center", va="center",
            fontsize=AXIS_LABEL_FONTSIZE, transform=ax.transAxes,
        )
    else:
        y_n = pops_novib[:N_STRUCTURES]
        y_v = pops_vib[:N_STRUCTURES]
        ax.plot(
            x[: y_n.size], y_n,
            color=NOVIB_COLOR, linewidth=1.2, label="No vib",
            zorder=2, solid_capstyle="butt",
        )
        ax.plot(
            x[: y_v.size], y_v,
            color=VIB_COLOR, linewidth=1.2, label="Vib",
            zorder=2, solid_capstyle="butt",
        )

    ax.set_xlim(0.0, float(N_STRUCTURES - 1))
    ax.set_ylim(0.0, ymax)
    ax.set_xticks(x)
    ax.set_xticklabels(
        _structure_tick_labels(primed=primed_ticks),
        fontsize=TICK_FONTSIZE,
    )
    ax.set_xlabel(
        "Lowest energy structures" if show_xlabel else "",
        fontsize=AXIS_LABEL_FONTSIZE,
    )
    if show_ir_ylabel:
        ax.set_ylabel(rf"Ir$_{{{n_ir}}}$", fontsize=AXIS_LABEL_FONTSIZE)
    else:
        ax.set_ylabel("")
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE, labelleft=True)
    if col_title:
        ax.set_title(col_title, fontsize=TITLE_FONTSIZE, pad=3)
    ax.margins(x=0)
    if show_legend:
        ax.legend(fontsize=LEGEND_FONTSIZE, frameon=False, loc="best")


def main() -> None:
    ensure_for_figures([8])
    ensure_dirs(OUTPUT_DIR)

    vd = load_npz(
        GC_VARIANTS_NPZ,
        keys=("Ts", "Ps", "novib_Pgi", "full_Pgi", "novib_G_tensor"),
    )
    Ts, Ps = vd["Ts"], vd["Ps"]
    novib_Pgi = vd["novib_Pgi"]
    full_Pgi = vd["full_Pgi"]
    novib_G = vd["novib_G_tensor"]

    gc = load_gc_npz(GC_NPZ)
    rows = load_ace_features(ACE_FEATURES["PBE"])
    jt = build_joined_table(gc, rows, "PBE")
    n_ir = np.asarray(jt["n_Ir"], dtype=int)
    formulas = np.asarray(jt["formulas_gc"], dtype=object)

    pressure_pa = FIG08_PRESSURE_PA
    panels_by_t: dict[float, dict[int, tuple[np.ndarray, np.ndarray]]] = {}
    meta: dict[float, tuple[float, float]] = {}
    for T in TEMPERATURES:
        t_used, p_used, lines_by_ir = ranked_structure_pops_by_ir_at_fixed_point(
            novib_Pgi, full_Pgi, novib_G, n_ir, Ts, Ps,
            T, pressure_pa,
            ir_counts=IR_COUNTS,
            formulas=formulas,
            max_structures=N_STRUCTURES,
        )
        panels_by_t[T] = lines_by_ir
        meta[T] = (t_used, p_used)

    ymax = _global_ymax(panels_by_t)
    p_used = meta[TEMPERATURES[0]][1]

    fig, axes = plt.subplots(
        len(IR_COUNTS), 2,
        figsize=(FIG_W_MM / 25.4, FIG_H_MM / 25.4),
        dpi=DPI,
        sharex=True,
        sharey=True,
    )

    last_row = len(IR_COUNTS) - 1
    for col, T in enumerate(TEMPERATURES):
        t_used, _ = meta[T]
        lines_by_ir = panels_by_t[T]
        for row, n in enumerate(IR_COUNTS):
            pair = lines_by_ir.get(n)
            pops_n = pair[0] if pair is not None else None
            pops_v = pair[1] if pair is not None else None
            _plot_ir_panel(
                axes[row, col], pops_n, pops_v,
                n_ir=n,
                ymax=ymax,
                show_xlabel=(row == last_row),
                show_ir_ylabel=(col == 0),
                show_legend=(row == 0 and col == 1),
                primed_ticks=(col == 1),
                col_title=(
                    rf"$T={t_used:.0f}$ K"
                    if row == 0 else ""
                ),
            )

    for ax in axes.ravel():
        ax.tick_params(axis="x", labelbottom=True)
        ax.tick_params(axis="y", labelleft=True)

    fig.supylabel("Population (%)", fontsize=AXIS_LABEL_FONTSIZE)
    fig.suptitle(
        rf"$p_{{\mathrm{{O_2}}}}={p_used:.3g}$ Pa",
        fontsize=TITLE_FONTSIZE,
        y=1.0,
    )
    fig.subplots_adjust(
        left=0.10, right=0.98, top=0.88, bottom=0.14,
        hspace=0.40, wspace=0.18,
    )
    save_fig(fig, OUTPUT_PATH, dpi=DPI, bbox_inches="tight", pad_inches=0.02)
    print(f"\n[fig08] {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
