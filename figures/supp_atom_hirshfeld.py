"""Supplementary – Hirshfeld q vs |spin| by atom motif (PBE | PBE0).

1×2 layout:
  Hirshfeld charge vs |spin| scatter, coloured by atom motif (PBE | PBE0).

Output: ``figures/outputs/supp_atom_hirshfeld.svg``
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt

from figures.config import DPI, MOTIF_JSON, OUTPUT_DIR, TRAJ_PBE, TRAJ_PBE0
from figures.utils.cache import ensure_for_supplementary
from figures.utils.hirshfeld import (
    charge_spin_panel,
    global_hirshfeld_limits,
    hirshfeld_by_motif,
    pin_limits,
)
from figures.utils.io import ensure_dirs
from figures.utils.plotting import (
    save_fig,
    AXIS_LABEL_FONTSIZE,
)

OUTPUT_PATH = os.path.join(OUTPUT_DIR, "supp_atom_hirshfeld.svg")
FIG_W_MM = 180
FIG_H_MM = 60

DATASETS = [
    {"label": "PBE", "traj": TRAJ_PBE, "motif_json": MOTIF_JSON["PBE"]},
    {"label": "PBE0", "traj": TRAJ_PBE0, "motif_json": MOTIF_JSON["PBE0"]},
]


def main() -> None:
    ensure_for_supplementary(["supp_atom_hirshfeld"])
    ensure_dirs(OUTPUT_DIR)

    limits = global_hirshfeld_limits(DATASETS)
    if limits is not None:
        xlim, ylim = limits
    else:
        xlim, ylim = None, None

    fig, axes = plt.subplots(
        1, 2, figsize=(FIG_W_MM / 25.4, FIG_H_MM / 25.4), dpi=DPI,
        sharex=True, sharey=True,
    )

    for ax, ds in zip(axes, DATASETS):
        if not os.path.exists(ds["traj"]):
            ax.text(0.5, 0.5, "Trajectory missing", ha="center", va="center",
                    fontsize=AXIS_LABEL_FONTSIZE, transform=ax.transAxes)
            ax.set_axis_off()
            continue
        if not os.path.exists(ds["motif_json"]):
            ax.text(0.5, 0.5, "Motif JSON missing", ha="center", va="center",
                    fontsize=AXIS_LABEL_FONTSIZE, transform=ax.transAxes)
            ax.set_axis_off()
            continue

        points, motif = hirshfeld_by_motif(ds["traj"], ds["motif_json"])
        charge_spin_panel(
            ax, points, motif,
            f"{ds['label']} — Hirshfeld q vs |spin|",
            xlim=xlim,
            ylim=ylim,
        )

    fig.tight_layout(pad=6 / 25.4)

    if limits is not None:
        for ax in axes:
            if ax.get_visible():
                pin_limits(ax, xlim=xlim, ylim=ylim)

    save_fig(fig, OUTPUT_PATH, dpi=DPI)
    print(f"\n[supp] {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
