"""Figure 8 – Novib vs vib ranked populations for Ir₄, Ir₅, Ir₆, PBE | PBE0.

3×4 layout (180 mm wide, short rectangular panels) at ambient ``pO₂``:

  Rows — Ir₄ | Ir₅ | Ir₆
  Columns — PBE ~273 K | PBE ~500 K | PBE0 ~273 K | PBE0 ~500 K

Each panel: population (%) of the five lowest-energy structures (s0…s4; order fixed by novib ``G``,
within each functional). Blue = no vib; orange = with vib. Shared ``x`` and ``y`` across all panels.

Output: ``figures/outputs/fig08_min_pop_at_low_po2.svg``
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

from figures.config import (
    DPI,
    FIG04_IR_COUNTS,
    FIG08_PRESSURE_PA,
    OUTPUT_DIR,
)
from figures.utils.cache import ensure_for_figures
from figures.utils.datasets import DATASETS, dataset_inputs
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
ROW_H_MM = 32
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
    panels_by_t: dict,
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
        # Plain-text prime: the mathtext prime is a cmsy10 glyph that SVG→PNG (text kept as text) draws as "0".
        return [f"s\u2032{k}" for k in range(N_STRUCTURES)]
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


def _dataset_panels(label: str, pressure_pa: float) -> tuple[dict, dict]:
    """Ranked novib / vib populations by Ir size at each temperature, for one functional."""
    inp = dataset_inputs(label)
    vd = load_npz(
        inp.variants_npz,
        keys=("Ts", "Ps", "novib_Pgi", "full_Pgi", "novib_G_tensor"),
    )
    gc = load_gc_npz(inp.gc_npz)
    jt = build_joined_table(gc, load_ace_features(inp.ace_features), label)
    n_ir = np.asarray(jt["n_Ir"], dtype=int)
    formulas = np.asarray(jt["formulas_gc"], dtype=object)
    panels, meta = {}, {}
    for T in TEMPERATURES:
        t_used, p_used, lines_by_ir = ranked_structure_pops_by_ir_at_fixed_point(
            vd["novib_Pgi"], vd["full_Pgi"], vd["novib_G_tensor"], n_ir, vd["Ts"], vd["Ps"],
            T, pressure_pa,
            ir_counts=IR_COUNTS,
            formulas=formulas,
            max_structures=N_STRUCTURES,
        )
        panels[(label, T)] = lines_by_ir
        meta[(label, T)] = (t_used, p_used)
    return panels, meta


def main() -> None:
    ensure_for_figures([8])
    ensure_dirs(OUTPUT_DIR)

    pressure_pa = FIG08_PRESSURE_PA
    panels: dict = {}
    meta: dict = {}
    for label in DATASETS:
        p, m = _dataset_panels(label, pressure_pa)
        panels.update(p)
        meta.update(m)
    columns = [(label, T) for label in DATASETS for T in TEMPERATURES]

    ymax = _global_ymax(panels)
    p_used = meta[columns[0]][1]

    fig, axes = plt.subplots(
        len(IR_COUNTS), len(columns),
        figsize=(FIG_W_MM / 25.4, FIG_H_MM / 25.4),
        dpi=DPI,
        sharex=False,   # shared x would make the last column's s′k tick labels overwrite the sk ones
        sharey=True,
    )

    last_row = len(IR_COUNTS) - 1
    for col, key in enumerate(columns):
        label, T = key
        t_used, _ = meta[key]
        lines_by_ir = panels[key]
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
                show_legend=(row == 0 and col == len(columns) - 1),
                primed_ticks=(T != TEMPERATURES[0]),
                col_title=(
                    rf"{label}, $T={t_used:.0f}$ K"
                    if row == 0 else ""
                ),
            )

    for ax in axes.ravel():
        ax.tick_params(axis="x", labelbottom=True)
        ax.tick_params(axis="y", labelleft=True)

    fig.supylabel("Population (%)", fontsize=AXIS_LABEL_FONTSIZE, x=0.005)
    fig.suptitle(
        rf"$p_{{\mathrm{{O_2}}}}={p_used:.3g}$ Pa",
        fontsize=TITLE_FONTSIZE,
        y=1.0,
    )
    fig.subplots_adjust(
        left=0.085, right=0.99, top=0.88, bottom=0.14,
        hspace=0.40, wspace=0.22,
    )
    save_fig(fig, OUTPUT_PATH, dpi=DPI, bbox_inches="tight", pad_inches=0.02)
    print(f"\n[fig08] {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
