"""Figure 6 – Ambient-pO₂ atom motifs, O interface, and O |spin| KDE.

Layout (180 mm wide):
  Row 0: PBE atom motifs | PBE0 atom motifs
  Rows 1–2 left: Bridging|terminal O interface
  Rows 1–2 right: PBE / PBE0 O |spin| KDE (stacked, shared axes)

Output: ``figures/outputs/fig06_gc_pressure_slices.svg``
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from aist.ace.common import ace_atom_display_label, ace_atom_label

from figures.config import (
    ACE_FEATURES,
    DATASET_COLORS,
    DPI,
    FIG06_O_BRIDGING_IDX,
    FIG06_POP_ZOOM_MIN,
    FIG06_PRESSURE_PA,
    FIG06_SMOOTH_SIGMA,
    FIG06_SPIN_XMAX,
    GC_NPZ,
    GC_NPZ_PBE0,
    MOTIF_JSON,
    OUTPUT_DIR,
    TRAJ_PBE,
    TRAJ_PBE0,
)
from figures.utils.cache import ensure_for_figures
from figures.utils.hirshfeld import (
    global_hirshfeld_limits,
    hirshfeld_by_motif,
    kde_curve,
    o_spin_kde_panel,
    pin_limits,
)
from figures.utils.io import ensure_dirs, load_ace_features, load_gc_npz
from figures.utils.join import (
    atom_motif_population_vs_T_at_pressure,
    build_joined_table,
    o_bridging_terminal_interface_vs_T_at_pressure,
)
from figures.utils.plotting import (
    add_fig06_interface_pbe_pbe0_diff_axis,
    draw_population_stack,
    ir_motif_colors,
    o_motif_colors,
    save_fig,
)

OUTPUT_PATH = os.path.join(OUTPUT_DIR, "fig06_gc_pressure_slices.svg")
FIG_W_MM = 180
FIG_H_MM = 140

DATASETS = [
    {"label": "PBE", "npz": GC_NPZ, "ace_csv": ACE_FEATURES["PBE"]},
    {"label": "PBE0", "npz": GC_NPZ_PBE0, "ace_csv": ACE_FEATURES["PBE0"]},
]

HIRSHFELD_DATASETS = [
    {"label": "PBE", "traj": TRAJ_PBE, "motif_json": MOTIF_JSON["PBE"]},
    {"label": "PBE0", "traj": TRAJ_PBE0, "motif_json": MOTIF_JSON["PBE0"]},
]


def _panel_title(label: str, panel_kind: str, pressure_pa: float) -> str:
    return (
        f"{label} — {panel_kind} — "
        rf"$p_{{\mathrm{{O_2}}}}={pressure_pa:.3g}$ Pa"
    )


def _draw_atom_panel(
    ax,
    joined: dict,
    label: str,
    pressure_pa: float,
) -> None:
    Ts, p_used, Y = atom_motif_population_vs_T_at_pressure(
        joined, pressure_pa, smooth_sigma=FIG06_SMOOTH_SIGMA,
    )
    k_ir = joined["ir_frac"].shape[1]
    k_o = joined["o_frac"].shape[1]
    colors = ir_motif_colors(k_ir) + o_motif_colors(k_o)
    motif_labels = [ace_atom_display_label("Ir", c) for c in range(k_ir)]
    motif_labels += [ace_atom_display_label("O", c) for c in range(k_o)]
    draw_population_stack(
        ax, Ts, Y, colors,
        labels=motif_labels,
        xlabel="T (K)",
        ylabel="Atom motif pop. (%)",
        title=_panel_title(label, "atom motifs", p_used),
        legend_ncol=2,
        boundary_after=k_ir,
    )
    ax.margins(x=0)


def _draw_o_interface_panel(
    ax,
    joined_by_label: dict[str, dict],
    pressure_pa: float,
) -> None:
    series: dict[str, np.ndarray] = {}
    Ts_ref: np.ndarray | None = None

    for label in ("PBE", "PBE0"):
        jt = joined_by_label.get(label)
        if jt is None:
            continue
        Ts, _, interface = o_bridging_terminal_interface_vs_T_at_pressure(
            jt,
            pressure_pa,
            FIG06_O_BRIDGING_IDX,
            smooth_sigma=FIG06_SMOOTH_SIGMA,
        )
        Ts_ref = Ts
        series[label] = interface

    if not series:
        ax.text(0.5, 0.5, "GC data missing", ha="center", va="center",
                fontsize=7, transform=ax.transAxes)
        ax.set_axis_off()
        return

    if Ts_ref is not None and "PBE" in series and "PBE0" in series:
        add_fig06_interface_pbe_pbe0_diff_axis(
            ax, Ts_ref, series["PBE"], series["PBE0"],
        )

    for label in ("PBE", "PBE0"):
        if label not in series:
            continue
        color = DATASET_COLORS.get(label, "#333333")
        ax.plot(
            Ts_ref, series[label],
            color=color, linewidth=1.5, label=label, zorder=3,
        )

    xlim = (float(Ts_ref[0]), float(Ts_ref[-1]))
    ax.set_xlim(xlim)
    ax.margins(x=0)
    ax.set_xlabel("T (K)", fontsize=7)
    ax.set_ylabel("Cumulative pop. (%)", fontsize=7)
    ax.set_ylim(FIG06_POP_ZOOM_MIN, 100.0)
    ax.set_title("Bridging Terminal oxygen interphase", fontsize=7)
    ax.tick_params(axis="both", labelsize=6)
    leg = ax.legend(fontsize=5, frameon=False, loc="best")
    for handle in leg.legend_handles:
        handle.set_alpha(1.0)


def _spin_eval_xlim(limits: tuple | None) -> tuple[float, float]:
    if limits is not None:
        return (float(limits[1][0]), float(FIG06_SPIN_XMAX))
    return (0.0, float(FIG06_SPIN_XMAX))


def _pbe0_bridging_kde_peak(eval_xlim: tuple[float, float]) -> float | None:
    ds = next(d for d in HIRSHFELD_DATASETS if d["label"] == "PBE0")
    if not os.path.exists(ds["traj"]) or not os.path.exists(ds["motif_json"]):
        return None
    points, _ = hirshfeld_by_motif(ds["traj"], ds["motif_json"])
    bridging_tag = ace_atom_label("O", FIG06_O_BRIDGING_IDX)
    _, spins = points.get(bridging_tag, (np.array([]), np.array([])))
    if spins.size < 2:
        return None
    curve = kde_curve(spins, xlim=eval_xlim)
    if curve is None:
        return None
    return float(curve[1].max())


def _draw_dataset_o_spin_kde(
    ax,
    ds: dict,
    *,
    eval_xlim: tuple[float, float],
    display_xlim: tuple[float, float],
    ylim: tuple[float, float] | None,
    show_xlabel: bool,
) -> None:
    if not os.path.exists(ds["traj"]) or not os.path.exists(ds["motif_json"]):
        ax.text(0.5, 0.5, "Data missing", ha="center", va="center",
                fontsize=7, transform=ax.transAxes)
        ax.set_axis_off()
        return
    points, motif = hirshfeld_by_motif(ds["traj"], ds["motif_json"])
    o_spin_kde_panel(
        ax, points, motif,
        "",
        eval_xlim=eval_xlim,
        display_xlim=display_xlim,
        show_xlabel=show_xlabel,
        show_ylabel=False,
        corner_label=ds["label"],
    )
    if ylim is not None and ax.get_visible():
        pin_limits(ax, xlim=display_xlim, ylim=ylim)
    if not show_xlabel:
        ax.tick_params(labelbottom=False)
    else:
        _format_spin_kde_xaxis(ax, display_xlim[0], display_xlim[1])


def _format_spin_kde_xaxis(ax, xmin: float, xmax: float) -> None:
    tick_candidates = [0.0, 0.2, 0.4, 0.6, 0.8, xmax]
    ticks = [t for t in tick_candidates if xmin <= t <= xmax]
    if xmax not in ticks:
        ticks.append(xmax)
    ax.set_xlim(xmin, xmax)
    ax.set_xticks(ticks)
    ax.margins(x=0)


def _add_stacked_kde_density_label(fig, ax_top, ax_bottom) -> None:
    """``Density`` y-label centred on the combined KDE stack."""
    bbox_top = ax_top.get_position()
    bbox_bot = ax_bottom.get_position()
    y_center = 0.5 * (bbox_top.y0 + bbox_bot.y1)
    x_main = bbox_top.x0 - 0.045
    fig.text(
        x_main, y_center, "Density",
        rotation=90, va="center", ha="center", fontsize=8,
    )


def main() -> None:
    ensure_for_figures([6])
    ensure_dirs(OUTPUT_DIR)

    joined_by_label: dict[str, dict | None] = {}
    for ds in DATASETS:
        if not os.path.exists(ds["npz"]):
            joined_by_label[ds["label"]] = None
            continue
        gc = load_gc_npz(ds["npz"])
        rows = load_ace_features(ds["ace_csv"])
        joined_by_label[ds["label"]] = build_joined_table(gc, rows, ds["label"])

    joined_ok = {k: v for k, v in joined_by_label.items() if v is not None}

    limits = global_hirshfeld_limits(HIRSHFELD_DATASETS)
    spin_eval_xlim = _spin_eval_xlim(limits)
    kde_display_xlim = (spin_eval_xlim[0], float(FIG06_SPIN_XMAX))

    kde_ylim: tuple[float, float] | None = None
    peak = _pbe0_bridging_kde_peak(spin_eval_xlim)
    if peak is not None and peak > 0:
        kde_ylim = (0.0, peak)

    fig = plt.figure(figsize=(FIG_W_MM / 25.4, FIG_H_MM / 25.4), dpi=DPI)
    gs = GridSpec(
        3, 2, figure=fig,
        height_ratios=[1.15, 0.5, 0.5],
        hspace=0.28,
        wspace=0.28,
    )

    ax_pbe = fig.add_subplot(gs[0, 0])
    ax_pbe0 = fig.add_subplot(gs[0, 1])
    ax_interface = fig.add_subplot(gs[1:, 0])

    gs_kde = gs[1:, 1].subgridspec(2, 1, hspace=0.04)
    ax_kde_pbe = fig.add_subplot(gs_kde[0, 0])
    ax_kde_pbe0 = fig.add_subplot(gs_kde[1, 0], sharex=ax_kde_pbe, sharey=ax_kde_pbe)

    for ax, label in zip((ax_pbe, ax_pbe0), ("PBE", "PBE0")):
        jt = joined_ok.get(label)
        if jt is None:
            ax.text(0.5, 0.5, "GC data missing", ha="center", va="center",
                    fontsize=7, transform=ax.transAxes)
            ax.set_axis_off()
            continue
        _draw_atom_panel(ax, jt, label, FIG06_PRESSURE_PA)

    _draw_o_interface_panel(ax_interface, joined_ok, FIG06_PRESSURE_PA)

    _draw_dataset_o_spin_kde(
        ax_kde_pbe, HIRSHFELD_DATASETS[0],
        eval_xlim=spin_eval_xlim,
        display_xlim=kde_display_xlim,
        ylim=kde_ylim,
        show_xlabel=False,
    )
    _draw_dataset_o_spin_kde(
        ax_kde_pbe0, HIRSHFELD_DATASETS[1],
        eval_xlim=spin_eval_xlim,
        display_xlim=kde_display_xlim,
        ylim=kde_ylim,
        show_xlabel=True,
    )

    fig.subplots_adjust(left=0.10, right=0.98, top=0.97, bottom=0.07)
    _add_stacked_kde_density_label(fig, ax_kde_pbe, ax_kde_pbe0)

    save_fig(fig, OUTPUT_PATH, dpi=DPI)
    print(f"\n[fig06] {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
