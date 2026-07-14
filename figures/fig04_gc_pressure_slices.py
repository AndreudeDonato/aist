"""Figure 4 – Per-Ir atom motif populations at low pO₂.

Layout (180 mm wide, 3×2):
  Row 0: Ir₃ — PBE | PBE0 atom-motif pop. (%) vs T
  Row 1: Ir₆ — PBE | PBE0
  Row 2: Ir₉ — PBE | PBE0

O motifs stacked above Ir motifs (black divider); optional GAS band at low pO₂.

Output: ``figures/outputs/fig04_gc_pressure_slices.svg``
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

from figures.config import (
    ACE_FEATURES,
    DPI,
    FIG04_IR_COUNTS,
    FIG04_LOW_PRESSURE_PA,
    FIG04_SMOOTH_SIGMA,
    OUTPUT_DIR,
)
from figures.utils.cache import ensure_for_figures
from figures.utils.comparison import (
    atom_motif_population_stack_by_ir_size,
    load_comparison_joined,
)
from figures.utils.io import ensure_dirs
from figures.utils.plotting import (
    AXIS_LABEL_FONTSIZE,
    TICK_FONTSIZE,
    TITLE_FONTSIZE,
    apply_gc_temperature_xticks,
    draw_population_stack,
    save_fig,
)

OUTPUT_PATH = os.path.join(OUTPUT_DIR, "fig04_gc_pressure_slices.svg")
FIG_W_MM = 180
ROW_H_MM = 53
FIG_H_MM = len(FIG04_IR_COUNTS) * ROW_H_MM

DATASETS = [
    {"label": "PBE", "ace_csv": ACE_FEATURES["PBE"]},
    {"label": "PBE0", "ace_csv": ACE_FEATURES["PBE0"]},
]
COL_LABELS = ("PBE", "PBE0")


def _load_stack(
    joined: dict,
    n_ir: int,
    pressure_pa: float,
) -> tuple[np.ndarray, np.ndarray, list, list, int | None] | None:
    stack = atom_motif_population_stack_by_ir_size(
        joined, n_ir, pressure_pa, smooth_sigma=FIG04_SMOOTH_SIGMA,
    )
    if stack is None:
        return None
    Ts, _, Y, colors, motif_labels, boundary_after = stack
    return Ts, Y, colors, motif_labels, boundary_after


def _stack_ymax(Y: np.ndarray) -> float:
    return float(np.nanmax(np.sum(Y, axis=0)))


def _draw_atom_panel(
    ax,
    joined: dict,
    *,
    n_ir: int,
    pressure_pa: float,
) -> None:
    stack = _load_stack(joined, n_ir, pressure_pa)
    if stack is None:
        ax.text(
            0.5, 0.5, "No data", ha="center", va="center",
            fontsize=AXIS_LABEL_FONTSIZE, transform=ax.transAxes,
        )
        ax.set_axis_off()
        return

    Ts, Y, colors, motif_labels, boundary_after = stack
    draw_population_stack(
        ax, Ts, Y, colors,
        labels=motif_labels,
        xlabel="",
        ylabel="",
        title="",
        legend_ncol=2,
        boundary_after=boundary_after,
    )
    ax.margins(x=0)


def _style_grid(axes: np.ndarray) -> None:
    last_row = axes.shape[0] - 1
    for row, n_ir in enumerate(FIG04_IR_COUNTS):
        for col in range(axes.shape[1]):
            ax = axes[row, col]
            if not ax.axison:
                continue
            if row == 0:
                ax.set_title(COL_LABELS[col], fontsize=TITLE_FONTSIZE, pad=3)
            if col == 0:
                ax.set_ylabel(
                    rf"Ir$_{{{n_ir}}}$",
                    fontsize=AXIS_LABEL_FONTSIZE,
                )
            else:
                ax.set_ylabel("")
                ax.tick_params(labelleft=False)
            if row < last_row:
                ax.tick_params(labelbottom=False)
                ax.set_xlabel("")
            else:
                ax.set_xlabel("T (K)", fontsize=AXIS_LABEL_FONTSIZE)
                apply_gc_temperature_xticks(ax)
            ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)


def main() -> None:
    ensure_for_figures([4])
    ensure_dirs(OUTPUT_DIR)

    joined_by_label: dict[str, dict | None] = {}
    for ds in DATASETS:
        try:
            joined_by_label[ds["label"]] = load_comparison_joined(
                ds["label"], ds["ace_csv"],
            )
        except FileNotFoundError:
            joined_by_label[ds["label"]] = None

    if not any(v is not None for v in joined_by_label.values()):
        raise FileNotFoundError("No GC data found for figure 4.")

    pressure_pa = FIG04_LOW_PRESSURE_PA

    ymax = 100.0
    for label in COL_LABELS:
        jt = joined_by_label.get(label)
        if jt is None:
            continue
        for n_ir in FIG04_IR_COUNTS:
            stack = _load_stack(jt, n_ir, pressure_pa)
            if stack is None:
                continue
            _, Y, _, _, _ = stack
            ymax = max(ymax, _stack_ymax(Y))

    fig, axes = plt.subplots(
        len(FIG04_IR_COUNTS), 2,
        figsize=(FIG_W_MM / 25.4, FIG_H_MM / 25.4),
        dpi=DPI,
        sharex="col",
        sharey=True,
    )

    for row, n_ir in enumerate(FIG04_IR_COUNTS):
        for col, label in enumerate(COL_LABELS):
            jt = joined_by_label.get(label)
            if jt is None:
                axes[row, col].text(
                    0.5, 0.5, "GC data missing", ha="center", va="center",
                    fontsize=AXIS_LABEL_FONTSIZE, transform=axes[row, col].transAxes,
                )
                axes[row, col].set_axis_off()
                continue
            _draw_atom_panel(
                axes[row, col], jt,
                n_ir=n_ir, pressure_pa=pressure_pa,
            )

    for ax in axes.ravel():
        if ax.axison:
            ax.set_ylim(0.0, ymax)

    _style_grid(axes)
    fig.supylabel("Atom motif pop. (%)", fontsize=AXIS_LABEL_FONTSIZE, x=0.02)
    fig.subplots_adjust(left=0.12, right=0.97, top=0.97, bottom=0.08)

    save_fig(fig, OUTPUT_PATH, dpi=DPI)
    print(f"\n[fig04] {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
