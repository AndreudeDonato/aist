"""Supplementary – |Hirshfeld spin| KDE by atom motif (O | Ir).

2×2 layout (180 mm wide, PBE | PBE0 columns):
  Row 1: O motif |spin| KDE (PBE | PBE0)
  Row 2: Ir motif |spin| KDE (PBE | PBE0)

Output: ``figures/outputs/supp_motif_spin_kde.svg``
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from aist.ace.common import ace_atom_label

from figures.config import (
    DPI,
    FIG04_O_BRIDGING_IDX,
    MOTIF_JSON,
    OUTPUT_DIR,
    TRAJ_PBE,
    TRAJ_PBE0,
)
from figures.utils.cache import ensure_for_supplementary
from figures.utils.hirshfeld import (
    hirshfeld_by_motif,
    motif_order,
    ir_motif_tags,
    kde_curve,
    motif_spin_kde_panel,
    o_motif_tags,
    pin_limits,
)
from figures.utils.io import ensure_dirs
from figures.utils.plotting import (
    AXIS_LABEL_FONTSIZE,
    save_fig,
)

OUTPUT_PATH = os.path.join(OUTPUT_DIR, "supp_motif_spin_kde.svg")
FIG_W_MM = 180
FIG_H_MM = 100

DATASETS = [
    {"label": "PBE", "traj": TRAJ_PBE, "motif_json": MOTIF_JSON["PBE"]},
    {"label": "PBE0", "traj": TRAJ_PBE0, "motif_json": MOTIF_JSON["PBE0"]},
]


def _spin_range(element: str) -> tuple[float, float]:
    """(min, max) |Hirshfeld spin| of one element's atoms over both functionals."""
    lo, hi = np.inf, -np.inf
    for ds in DATASETS:
        if not os.path.exists(ds["traj"]) or not os.path.exists(ds["motif_json"]):
            continue
        points, motif = hirshfeld_by_motif(ds["traj"], ds["motif_json"])
        for tag in motif_order(motif):
            if tag.startswith(element + "-"):
                spins = np.asarray(points.get(tag, ((), ()))[1], dtype=float)
                if spins.size:
                    lo, hi = min(lo, float(spins.min())), max(hi, float(spins.max()))
    return (0.0, 1.0) if not np.isfinite(lo) else (lo, hi)


def _format_spin_kde_xaxis(ax, xmin: float, xmax: float) -> None:
    step = 0.2 if xmax <= 1.2 else 0.5
    ax.set_xlim(xmin, xmax)
    ax.set_xticks([t for t in np.arange(0.0, xmax + 1e-9, step) if t >= xmin])
    ax.margins(x=0)


def _pbe0_bridging_kde_peak(eval_xlim: tuple[float, float]) -> float | None:
    ds = next(d for d in DATASETS if d["label"] == "PBE0")
    if not os.path.exists(ds["traj"]) or not os.path.exists(ds["motif_json"]):
        return None
    points, _ = hirshfeld_by_motif(ds["traj"], ds["motif_json"])
    bridging_tag = ace_atom_label("O", FIG04_O_BRIDGING_IDX)
    _, spins = points.get(bridging_tag, ((), ()))
    if len(spins) < 2:
        return None
    curve = kde_curve(spins, xlim=eval_xlim)
    if curve is None:
        return None
    return float(curve[1].max())


def _row_kde_peak(
    datasets: list[dict],
    tags_fn,
    eval_xlim: tuple[float, float],
) -> float:
    peak = 0.0
    for ds in datasets:
        if not os.path.exists(ds["traj"]) or not os.path.exists(ds["motif_json"]):
            continue
        points, motif = hirshfeld_by_motif(ds["traj"], ds["motif_json"])
        for tag in tags_fn(motif):
            _, spins = points.get(tag, ((), ()))
            curve = kde_curve(spins, xlim=eval_xlim)
            if curve is not None:
                peak = max(peak, float(curve[1].max()))
    return peak


def _draw_panel(
    ax,
    ds: dict,
    tags_fn,
    title: str,
    *,
    eval_xlim: tuple[float, float],
    display_xlim: tuple[float, float],
    ylim: tuple[float, float] | None,
    show_xlabel: bool,
) -> None:
    if not os.path.exists(ds["traj"]) or not os.path.exists(ds["motif_json"]):
        ax.text(
            0.5, 0.5, "Data missing", ha="center", va="center",
            fontsize=AXIS_LABEL_FONTSIZE, transform=ax.transAxes,
        )
        ax.set_axis_off()
        return
    points, motif = hirshfeld_by_motif(ds["traj"], ds["motif_json"])
    motif_spin_kde_panel(
        ax, points, motif, tags_fn(motif), title,
        eval_xlim=eval_xlim,
        display_xlim=display_xlim,
        show_xlabel=show_xlabel,
        show_ylabel=False,
    )
    if ylim is not None and ax.get_visible():
        pin_limits(ax, xlim=display_xlim, ylim=ylim)
    _format_spin_kde_xaxis(ax, display_xlim[0], display_xlim[1])   # rows have their own x range
    if not show_xlabel:
        ax.set_xlabel("")


def _draw_row(
    axes: tuple,
    tags_fn,
    row_label: str,
    *,
    eval_xlim: tuple[float, float],
    display_xlim: tuple[float, float],
    ylim: tuple[float, float] | None,
    show_xlabel: bool,
) -> None:
    for ax, ds in zip(axes, DATASETS):
        title = f"{ds['label']} — {row_label}"
        _draw_panel(
            ax, ds, tags_fn, title,
            eval_xlim=eval_xlim,
            display_xlim=display_xlim,
            ylim=ylim,
            show_xlabel=show_xlabel,
        )


def _add_density_label(fig, ax) -> None:
    bbox = ax.get_position()
    fig.text(
        bbox.x0 - 0.045, bbox.y0 + bbox.height / 2, "Density",
        rotation=90, va="center", ha="center", fontsize=AXIS_LABEL_FONTSIZE,
    )


def main() -> None:
    ensure_for_supplementary(["supp_motif_spin_kde"])
    ensure_dirs(OUTPUT_DIR)

    o_xlim = _spin_range("O")
    ir_xlim = _spin_range("Ir")

    o_peak = _pbe0_bridging_kde_peak(o_xlim)
    ir_peak = _row_kde_peak(DATASETS, ir_motif_tags, ir_xlim)
    o_ylim = (0.0, o_peak) if o_peak is not None and o_peak > 0 else None
    ir_ylim = (0.0, ir_peak * 1.05) if ir_peak > 0 else None

    fig = plt.figure(figsize=(FIG_W_MM / 25.4, FIG_H_MM / 25.4), dpi=DPI)
    gs = GridSpec(2, 2, figure=fig, hspace=0.32, wspace=0.28)

    ax_o_pbe = fig.add_subplot(gs[0, 0])
    ax_o_pbe0 = fig.add_subplot(gs[0, 1], sharex=ax_o_pbe, sharey=ax_o_pbe)
    ax_ir_pbe = fig.add_subplot(gs[1, 0])
    ax_ir_pbe0 = fig.add_subplot(gs[1, 1], sharex=ax_ir_pbe, sharey=ax_ir_pbe)

    _draw_row(
        (ax_o_pbe, ax_o_pbe0), o_motif_tags,
        "O motifs — |Hirshfeld spin|",
        eval_xlim=o_xlim, display_xlim=o_xlim, ylim=o_ylim,
        show_xlabel=False,
    )
    _draw_row(
        (ax_ir_pbe, ax_ir_pbe0), ir_motif_tags,
        "Ir motifs — |Hirshfeld spin|",
        eval_xlim=ir_xlim, display_xlim=ir_xlim, ylim=ir_ylim,
        show_xlabel=True,
    )

    fig.subplots_adjust(left=0.10, right=0.97, top=0.94, bottom=0.10)
    _add_density_label(fig, ax_o_pbe)
    _add_density_label(fig, ax_ir_pbe)

    save_fig(fig, OUTPUT_PATH, dpi=DPI)
    print(f"\n[supp] {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
