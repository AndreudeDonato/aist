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
from matplotlib.gridspec import GridSpec

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


def _panel_title(label: str, n_ir: int, pressure_pa: float) -> str:
    return (
        f"{label} — Ir$_{{{n_ir}}}$ atom motifs — "
        rf"$p_{{\mathrm{{O_2}}}}={pressure_pa:.3g}$ Pa"
    )


def _draw_atom_panel(
    ax,
    joined: dict,
    *,
    label: str,
    n_ir: int,
    pressure_pa: float,
    show_ylabel: bool,
) -> None:
    stack = atom_motif_population_stack_by_ir_size(
        joined, n_ir, pressure_pa, smooth_sigma=FIG04_SMOOTH_SIGMA,
    )
    if stack is None:
        ax.text(
            0.5, 0.5, "No data", ha="center", va="center",
            fontsize=AXIS_LABEL_FONTSIZE, transform=ax.transAxes,
        )
        ax.set_axis_off()
        return

    Ts, p_used, Y, colors, motif_labels, boundary_after = stack
    draw_population_stack(
        ax, Ts, Y, colors,
        labels=motif_labels,
        xlabel="",
        ylabel="Atom motif pop. (%)" if show_ylabel else "",
        title=_panel_title(label, n_ir, p_used),
        legend_ncol=2,
        boundary_after=boundary_after,
    )
    ax.margins(x=0)


def _style_grid(axes: np.ndarray) -> None:
    last_row = axes.shape[0] - 1
    for row in range(axes.shape[0]):
        for col in range(axes.shape[1]):
            ax = axes[row, col]
            if not ax.axison:
                continue
            if col > 0:
                ax.set_ylabel("")
            if row < last_row:
                ax.tick_params(labelbottom=False)
                ax.set_xlabel("")
            else:
                ax.set_xlabel("T (K)", fontsize=AXIS_LABEL_FONTSIZE)
                apply_gc_temperature_xticks(ax)


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

    fig = plt.figure(figsize=(FIG_W_MM / 25.4, FIG_H_MM / 25.4), dpi=DPI)
    gs = GridSpec(
        len(FIG04_IR_COUNTS), 2, figure=fig,
        height_ratios=[1.0] * len(FIG04_IR_COUNTS),
        hspace=0.32,
        wspace=0.28,
    )
    axes = np.empty((len(FIG04_IR_COUNTS), 2), dtype=object)
    for row, n_ir in enumerate(FIG04_IR_COUNTS):
        for col, label in enumerate(("PBE", "PBE0")):
            if row == 0 and col == 0:
                axes[row, col] = fig.add_subplot(gs[row, col])
            elif row == 0:
                axes[row, col] = fig.add_subplot(gs[row, col], sharex=axes[0, 0])
            else:
                axes[row, col] = fig.add_subplot(gs[row, col], sharex=axes[0, col])

            jt = joined_by_label.get(label)
            if jt is None:
                ax = axes[row, col]
                ax.text(
                    0.5, 0.5, "GC data missing", ha="center", va="center",
                    fontsize=AXIS_LABEL_FONTSIZE, transform=ax.transAxes,
                )
                ax.set_axis_off()
                continue
            _draw_atom_panel(
                axes[row, col], jt,
                label=label, n_ir=n_ir, pressure_pa=pressure_pa,
                show_ylabel=(col == 0),
            )

    _style_grid(axes)
    fig.subplots_adjust(left=0.10, right=0.97, top=0.97, bottom=0.08)

    save_fig(fig, OUTPUT_PATH, dpi=DPI)
    print(f"\n[fig04] {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
