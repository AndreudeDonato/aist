"""Supplementary – Fig. 4–style motif panels with per-size gradient bars at fixed pO₂.

Recreates the Fig. 4 layout (terminal O, bridging O, atom motifs) plus
``⟨HOMO–LUMO⟩`` gradient bars (not in main Fig. 4). O-motif maps are replaced
with population-weighted gradient bars vs ``T``: one horizontal band per
``n_Ir`` (Ir₃…Ir₉), coloured by the ensemble ``⟨n_O^{motif} / n_Ir⟩`` at
fixed pressure.

Outputs (180 × 213 mm each):
  ``supp_gc_motif_bars_low_po2.svg``  — ``FIG04_LOW_PRESSURE_PA``
  ``supp_gc_motif_bars_high_po2.svg`` — ``FIG04_HIGH_PRESSURE_PA``
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.gridspec import GridSpec

from figures.config import (
    ACE_FEATURES,
    DPI,
    FIG04_HIGH_PRESSURE_PA,
    FIG04_LOW_PRESSURE_PA,
    FIG04_O_BRIDGING_IDX,
    FIG04_O_TERMINAL_IDX,
    FIG04_SMOOTH_SIGMA,
    OUTPUT_DIR,
    PHASE_CMAP,
    SUPP_IR_COUNTS,
    TRAJ_PBE,
    TRAJ_PBE0,
)
from figures.utils.cache import ensure_for_supplementary
from figures.utils.comparison import (
    atom_motif_population_stack_unified_gas,
    homo_lumo_gaps_for_comparison,
    load_comparison_joined,
)
from figures.utils.gc_gradient_bars import (
    draw_o_motif_gradient_bars,
    draw_scalar_gradient_bars,
    ensemble_o_motif_vs_T_at_pressure,
    ensemble_scalar_vs_T_at_pressure,
    sync_temperature_xaxis,
)
from figures.utils.homo_lumo import HOMO_LUMO_CMAP
from figures.utils.io import ensure_dirs
from figures.utils.plotting import (
    AXIS_LABEL_FONTSIZE,
    TICK_FONTSIZE,
    TITLE_FONTSIZE,
    add_phase_colorbar,
    draw_population_stack,
    margin_colorbar_cax,
    save_fig,
)

FIG_W_MM = 180
FIG_H_MM = 230
GRID_HSPACE = 0.38

DATASETS = [
    {"label": "PBE", "ace_csv": ACE_FEATURES["PBE"], "traj": TRAJ_PBE},
    {"label": "PBE0", "ace_csv": ACE_FEATURES["PBE0"], "traj": TRAJ_PBE0},
]

BAR_ROWS = (
    (FIG04_O_TERMINAL_IDX, r"terminal O / $n_{\mathrm{Ir}}$"),
    (FIG04_O_BRIDGING_IDX, r"bridging O / $n_{\mathrm{Ir}}$"),
)

OUTPUTS = (
    (FIG04_LOW_PRESSURE_PA, "supp_gc_motif_bars_low_po2.svg"),
    (FIG04_HIGH_PRESSURE_PA, "supp_gc_motif_bars_high_po2.svg"),
)


@dataclass
class JoinedBundle:
    label: str
    joined: dict
    gaps: np.ndarray


def _load_bundles() -> dict[str, JoinedBundle]:
    out: dict[str, JoinedBundle] = {}
    for ds in DATASETS:
        try:
            jt = load_comparison_joined(ds["label"], ds["ace_csv"])
        except FileNotFoundError:
            continue
        if not os.path.exists(ds["traj"]):
            continue
        gaps = homo_lumo_gaps_for_comparison(ds["label"], ds["traj"], jt["n"])
        out[ds["label"]] = JoinedBundle(label=ds["label"], joined=jt, gaps=gaps)
    return out


def _field_limits(
    bundles: dict[str, JoinedBundle],
    pressure_pa: float,
    o_idx: int,
) -> tuple[float, float]:
    vals: list[float] = []
    for bundle in bundles.values():
        for n in SUPP_IR_COUNTS:
            curve = ensemble_o_motif_vs_T_at_pressure(
                bundle.joined, n, o_idx, pressure_pa,
            )
            if curve is None:
                continue
            finite = curve[2][np.isfinite(curve[2])]
            if finite.size:
                vals.extend([float(finite.min()), float(finite.max())])
    if not vals:
        return 0.0, 1.0
    vmin, vmax = float(min(vals)), float(max(vals))
    if vmax <= vmin:
        vmax = vmin + 1.0
    return vmin, vmax


def _homo_lumo_limits(
    bundles: dict[str, JoinedBundle],
    pressure_pa: float,
) -> tuple[float, float]:
    vals: list[float] = []
    for bundle in bundles.values():
        for n in SUPP_IR_COUNTS:
            curve = ensemble_scalar_vs_T_at_pressure(
                bundle.joined, n, bundle.gaps, pressure_pa,
            )
            if curve is None:
                continue
            finite = curve[2][np.isfinite(curve[2])]
            if finite.size:
                vals.extend([float(finite.min()), float(finite.max())])
    if not vals:
        return 0.0, 1.0
    vmin, vmax = float(min(vals)), float(max(vals))
    if vmax <= vmin:
        vmax = vmin + 1.0
    return vmin, vmax


def _panel_title(label: str, panel_kind: str, pressure_pa: float) -> str:
    return (
        f"{label} — {panel_kind} — "
        rf"$p_{{\mathrm{{O_2}}}}={pressure_pa:.3g}$ Pa"
    )


def _draw_bar_panel(
    ax,
    bundle: JoinedBundle,
    *,
    o_idx: int,
    row_label: str,
    pressure_pa: float,
    vmin: float,
    vmax: float,
    section_kind: str = "global_min",
) -> None:
    draw_o_motif_gradient_bars(
        ax,
        bundle.joined,
        pressure_pa,
        o_idx,
        SUPP_IR_COUNTS,
        cmap=PHASE_CMAP,
        vmin=vmin,
        vmax=vmax,
        section_kind=section_kind,
    )
    ax.set_ylabel(r"$n_{\mathrm{Ir}}$", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title(
        _panel_title(bundle.label, row_label, pressure_pa),
        fontsize=TITLE_FONTSIZE,
    )
    ax.tick_params(axis="both", labelsize=AXIS_LABEL_FONTSIZE)


def _draw_homo_lumo_panel(
    ax,
    bundle: JoinedBundle,
    *,
    pressure_pa: float,
    vmin: float,
    vmax: float,
) -> None:
    draw_scalar_gradient_bars(
        ax,
        bundle.joined,
        pressure_pa,
        SUPP_IR_COUNTS,
        bundle.gaps,
        cmap=HOMO_LUMO_CMAP,
        vmin=vmin,
        vmax=vmax,
        section_kind="none",
    )
    ax.set_ylabel(r"$n_{\mathrm{Ir}}$", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title(
        _panel_title(bundle.label, r"$\langle\mathrm{HOMO-LUMO}\rangle$", pressure_pa),
        fontsize=TITLE_FONTSIZE,
    )
    ax.tick_params(axis="both", labelsize=AXIS_LABEL_FONTSIZE)


def _draw_motif_panel(
    ax,
    bundle: JoinedBundle,
    pressure_pa: float,
) -> None:
    Ts, p_used, Y, colors, motif_labels, boundary_after = (
        atom_motif_population_stack_unified_gas(
            bundle.joined, pressure_pa, smooth_sigma=FIG04_SMOOTH_SIGMA,
        )
    )
    draw_population_stack(
        ax, Ts, Y, colors,
        labels=motif_labels,
        xlabel="",
        ylabel="Atom motif pop. (%)",
        title=_panel_title(bundle.label, "atom motifs", p_used),
        legend_ncol=2,
        boundary_after=boundary_after,
    )
    ax.margins(x=0)


def _style_bar_row(axes: list, ts: np.ndarray) -> None:
    for ax in axes:
        if not ax.axison:
            continue
        col = ax.get_subplotspec().colspan.start
        sync_temperature_xaxis(ax, ts, xlabel=None)
        if col > 0:
            ax.tick_params(labelleft=False)
            ax.set_ylabel("")


def _style_motif_row(axes: list, ts: np.ndarray) -> None:
    for ax in axes:
        if not ax.axison:
            continue
        sync_temperature_xaxis(ax, ts, xlabel="T (K)")


def _render_figure(pressure_pa: float, filename: str, bundles: dict[str, JoinedBundle]) -> None:
    if not bundles:
        raise FileNotFoundError("No GC comparison data for motif pressure-bar supplementary.")

    limits = {
        o_idx: _field_limits(bundles, pressure_pa, o_idx)
        for o_idx, _ in BAR_ROWS
    }
    zmin = min(lo for lo, _ in limits.values())
    zmax = max(hi for _, hi in limits.values())
    if zmax <= zmin:
        zmax = zmin + 1.0
    hl_vmin, hl_vmax = _homo_lumo_limits(bundles, pressure_pa)

    fig = plt.figure(figsize=(FIG_W_MM / 25.4, FIG_H_MM / 25.4), dpi=DPI)
    gs = GridSpec(4, 2, figure=fig, height_ratios=[1.0, 1.0, 1.0, 1.0], hspace=GRID_HSPACE, wspace=0.28)

    ax_term_pbe = fig.add_subplot(gs[0, 0])
    ax_term_pbe0 = fig.add_subplot(gs[0, 1], sharex=ax_term_pbe)
    ax_bridge_pbe = fig.add_subplot(gs[1, 0], sharex=ax_term_pbe, sharey=ax_term_pbe)
    ax_bridge_pbe0 = fig.add_subplot(gs[1, 1], sharex=ax_term_pbe, sharey=ax_term_pbe)
    ax_hl_pbe = fig.add_subplot(gs[2, 0], sharex=ax_term_pbe, sharey=ax_term_pbe)
    ax_hl_pbe0 = fig.add_subplot(gs[2, 1], sharex=ax_term_pbe, sharey=ax_term_pbe)
    ax_motif_pbe = fig.add_subplot(gs[3, 0], sharex=ax_term_pbe)
    ax_motif_pbe0 = fig.add_subplot(gs[3, 1], sharex=ax_term_pbe)

    bar_axes = {
        FIG04_O_TERMINAL_IDX: (ax_term_pbe, ax_term_pbe0),
        FIG04_O_BRIDGING_IDX: (ax_bridge_pbe, ax_bridge_pbe0),
    }
    for o_idx, row_label in BAR_ROWS:
        vmin, vmax = limits[o_idx]
        for label, ax in zip(("PBE", "PBE0"), bar_axes[o_idx]):
            bundle = bundles.get(label)
            if bundle is None:
                ax.text(
                    0.5, 0.5, "GC data missing", ha="center", va="center",
                    fontsize=AXIS_LABEL_FONTSIZE, transform=ax.transAxes,
                )
                ax.set_axis_off()
                continue
            _draw_bar_panel(
                ax, bundle,
                o_idx=o_idx,
                row_label=row_label,
                pressure_pa=pressure_pa,
                vmin=vmin,
                vmax=vmax,
                section_kind="global_min",
            )

    for label, ax in zip(("PBE", "PBE0"), (ax_hl_pbe, ax_hl_pbe0)):
        bundle = bundles.get(label)
        if bundle is None:
            ax.text(
                0.5, 0.5, "GC data missing", ha="center", va="center",
                fontsize=AXIS_LABEL_FONTSIZE, transform=ax.transAxes,
            )
            ax.set_axis_off()
            continue
        _draw_homo_lumo_panel(
            ax, bundle,
            pressure_pa=pressure_pa,
            vmin=hl_vmin,
            vmax=hl_vmax,
        )

    for label, ax in zip(("PBE", "PBE0"), (ax_motif_pbe, ax_motif_pbe0)):
        bundle = bundles.get(label)
        if bundle is None:
            ax.text(
                0.5, 0.5, "GC data missing", ha="center", va="center",
                fontsize=AXIS_LABEL_FONTSIZE, transform=ax.transAxes,
            )
            ax.set_axis_off()
            continue
        _draw_motif_panel(ax, bundle, pressure_pa)

    fig.subplots_adjust(left=0.10, right=0.92, top=0.95, bottom=0.09, hspace=GRID_HSPACE)

    ref = next(iter(bundles.values()))
    ts_ref = np.asarray(ref.joined["Ts"], dtype=float)
    _style_bar_row([ax_term_pbe, ax_term_pbe0, ax_bridge_pbe, ax_bridge_pbe0, ax_hl_pbe, ax_hl_pbe0], ts_ref)
    _style_motif_row([ax_motif_pbe, ax_motif_pbe0], ts_ref)

    mesh_ref = ScalarMappable(norm=Normalize(vmin=zmin, vmax=zmax), cmap=PHASE_CMAP)
    mesh_ref.set_array([])
    add_phase_colorbar(
        fig,
        mesh_ref,
        cax=margin_colorbar_cax(fig, ax_term_pbe, ax_bridge_pbe0, gap=0.01, width=0.012),
        label=r"$n_{\mathrm{O}}^{\mathrm{motif}} / n_{\mathrm{Ir}}$",
    )
    mesh_hl = ScalarMappable(norm=Normalize(vmin=hl_vmin, vmax=hl_vmax), cmap=HOMO_LUMO_CMAP)
    mesh_hl.set_array([])
    add_phase_colorbar(
        fig,
        mesh_hl,
        cax=margin_colorbar_cax(fig, ax_hl_pbe, ax_hl_pbe0, gap=0.01, width=0.012),
        label="HOMO–LUMO gap (eV)",
    )

    out_path = os.path.join(OUTPUT_DIR, filename)
    save_fig(fig, out_path, dpi=DPI)
    print(f"[supp] {out_path}")


def main() -> None:
    ensure_for_supplementary(["supp_gc_motif_pressure_bars"])
    ensure_dirs(OUTPUT_DIR)
    bundles = _load_bundles()
    for pressure_pa, filename in OUTPUTS:
        _render_figure(pressure_pa, filename, bundles)


if __name__ == "__main__":
    main()
