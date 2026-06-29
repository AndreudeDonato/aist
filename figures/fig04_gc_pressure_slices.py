"""Figure 4 – Low-pO₂ atom motifs and terminal/bridging O maps.

Layout (180 mm wide, vertical):
  Row 1: PBE ⟨terminal O / n_Ir⟩ map | PBE0 ⟨terminal O / n_Ir⟩ map
  Row 2: PBE ⟨bridging O / n_Ir⟩ map | PBE0 ⟨bridging O / n_Ir⟩ map
  Row 3: PBE atom motifs | PBE0 atom motifs  (O motifs below Ir motifs)

Atom motif stacks at fixed low pO₂; rows share the temperature x-axis (T label on motifs).

Output: ``figures/outputs/fig04_gc_pressure_slices.svg``
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec

from aist.ace.common import ace_atom_display_label

from figures.config import (
    ACE_FEATURES,
    DPI,
    FIG04_LOW_PRESSURE_PA,
    FIG04_O_BRIDGING_IDX,
    FIG04_O_TERMINAL_IDX,
    FIG04_SMOOTH_SIGMA,
    GC_NPZ,
    GC_NPZ_PBE0,
    OUTPUT_DIR,
    PHASE_CMAP,
)
from figures.utils.cache import ensure_for_figures
from figures.utils.io import ensure_dirs, load_ace_features, load_gc_npz
from figures.utils.join import (
    atom_motif_population_vs_T_at_pressure,
    build_joined_table,
    ensemble_o_motif_per_ir_map,
)
from figures.utils.plotting import (
    AXIS_LABEL_FONTSIZE,
    TITLE_FONTSIZE,
    add_phase_colorbar,
    draw_population_stack,
    ir_motif_colors,
    margin_colorbar_cax,
    o_motif_colors,
    save_fig,
    style_phase_ax,
)

OUTPUT_PATH = os.path.join(OUTPUT_DIR, "fig04_gc_pressure_slices.svg")
FIG_W_MM = 180
FIG_H_MM = 160

DATASETS = [
    {"label": "PBE", "npz": GC_NPZ, "ace_csv": ACE_FEATURES["PBE"]},
    {"label": "PBE0", "npz": GC_NPZ_PBE0, "ace_csv": ACE_FEATURES["PBE0"]},
]

PHASE_MAP_ROWS = [
    (FIG04_O_TERMINAL_IDX, r"terminal O / $n_{\mathrm{Ir}}$"),
    (FIG04_O_BRIDGING_IDX, r"bridging O / $n_{\mathrm{Ir}}$"),
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
        joined, pressure_pa, smooth_sigma=FIG04_SMOOTH_SIGMA,
    )
    k_ir = joined["ir_frac"].shape[1]
    k_o = joined["o_frac"].shape[1]
    Y = np.vstack([Y[k_ir:], Y[:k_ir]])
    colors = o_motif_colors(k_o) + ir_motif_colors(k_ir)
    motif_labels = [ace_atom_display_label("O", c) for c in range(k_o)]
    motif_labels += [ace_atom_display_label("Ir", c) for c in range(k_ir)]
    draw_population_stack(
        ax, Ts, Y, colors,
        labels=motif_labels,
        xlabel="",
        ylabel="Atom motif pop. (%)",
        title=_panel_title(label, "atom motifs", p_used),
        legend_ncol=2,
        boundary_after=k_o,
    )
    ax.margins(x=0)


def _field_limits(joined_by_label: dict[str, dict], o_idx: int) -> tuple[float, float]:
    vals: list[float] = []
    for jt in joined_by_label.values():
        Z = ensemble_o_motif_per_ir_map(jt, o_idx)
        finite = Z[np.isfinite(Z)]
        if finite.size:
            vals.extend([float(finite.min()), float(finite.max())])
    if not vals:
        return 0.0, 1.0
    vmin, vmax = float(min(vals)), float(max(vals))
    if vmax <= vmin:
        vmax = vmin + 1.0
    return vmin, vmax


def _shared_map_limits(joined_by_label: dict[str, dict]) -> tuple[float, float]:
    vals: list[float] = []
    for o_idx, _ in PHASE_MAP_ROWS:
        vmin, vmax = _field_limits(joined_by_label, o_idx)
        vals.extend([vmin, vmax])
    if not vals:
        return 0.0, 1.0
    zmin, zmax = float(min(vals)), float(max(vals))
    if zmax <= zmin:
        zmax = zmin + 1.0
    return zmin, zmax


def _phase_panel(
    ax,
    joined: dict,
    o_idx: int,
    title: str,
    vmin: float,
    vmax: float,
) -> object | None:
    Ts, Ps = joined["Ts"], joined["Ps"]
    Z = ensemble_o_motif_per_ir_map(joined, o_idx)
    if not np.isfinite(Z).any():
        ax.text(
            0.5, 0.5, "No data", ha="center", va="center",
            fontsize=AXIS_LABEL_FONTSIZE, transform=ax.transAxes,
        )
        style_phase_ax(ax)
        ax.set_title(title, fontsize=TITLE_FONTSIZE)
        return None

    mesh = ax.pcolormesh(
        Ts, Ps, Z, shading="auto",
        cmap=PHASE_CMAP,
        norm=Normalize(vmin=vmin, vmax=vmax),
        rasterized=True,
    )
    style_phase_ax(ax)
    ax.set_title(title, fontsize=TITLE_FONTSIZE)
    return mesh


def _style_phase_map_grid(phase_axes: list) -> None:
    """y-label on left column only; x labels on motif row below."""
    for ax in phase_axes:
        col = ax.get_subplotspec().colspan.start
        ax.tick_params(labelbottom=False)
        ax.set_xlabel("")
        if col > 0:
            ax.tick_params(labelleft=False)
            ax.set_ylabel("")


def _style_motif_row(motif_axes: list) -> None:
    for ax in motif_axes:
        ax.set_xlabel("T (K)", fontsize=AXIS_LABEL_FONTSIZE)


def main() -> None:
    ensure_for_figures([4])
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
    if not joined_ok:
        raise FileNotFoundError("No GC data found for figure 4.")

    zmin, zmax = _shared_map_limits(joined_ok)

    fig = plt.figure(figsize=(FIG_W_MM / 25.4, FIG_H_MM / 25.4), dpi=DPI)
    gs = GridSpec(
        3, 2, figure=fig,
        height_ratios=[1.0, 1.0, 1.0],
        hspace=0.28,
        wspace=0.28,
    )

    ax_term_pbe = fig.add_subplot(gs[0, 0])
    ax_term_pbe0 = fig.add_subplot(gs[0, 1], sharex=ax_term_pbe)

    ax_bridge_pbe = fig.add_subplot(gs[1, 0], sharex=ax_term_pbe, sharey=ax_term_pbe)
    ax_bridge_pbe0 = fig.add_subplot(gs[1, 1], sharex=ax_term_pbe, sharey=ax_term_pbe)

    ax_pbe = fig.add_subplot(gs[2, 0], sharex=ax_term_pbe)
    ax_pbe0 = fig.add_subplot(gs[2, 1], sharex=ax_term_pbe)

    mesh_ref = None

    for o_idx, row_label in PHASE_MAP_ROWS:
        axes_by_label = (
            (ax_term_pbe, ax_term_pbe0)
            if o_idx == FIG04_O_TERMINAL_IDX
            else (ax_bridge_pbe, ax_bridge_pbe0)
        )
        for label, ax in zip(("PBE", "PBE0"), axes_by_label):
            jt = joined_ok.get(label)
            if jt is None:
                ax.text(
                    0.5, 0.5, "GC data missing", ha="center", va="center",
                    fontsize=AXIS_LABEL_FONTSIZE, transform=ax.transAxes,
                )
                ax.set_axis_off()
                continue
            title = f"{label} — ⟨{row_label}⟩"
            mesh = _phase_panel(ax, jt, o_idx, title, zmin, zmax)
            if mesh_ref is None and mesh is not None:
                mesh_ref = mesh

    for ax, label in zip((ax_pbe, ax_pbe0), ("PBE", "PBE0")):
        jt = joined_ok.get(label)
        if jt is None:
            ax.text(
                0.5, 0.5, "GC data missing", ha="center", va="center",
                fontsize=AXIS_LABEL_FONTSIZE, transform=ax.transAxes,
            )
            ax.set_axis_off()
            continue
        _draw_atom_panel(ax, jt, label, FIG04_LOW_PRESSURE_PA)

    _style_phase_map_grid([ax_term_pbe, ax_term_pbe0, ax_bridge_pbe, ax_bridge_pbe0])
    _style_motif_row([ax_pbe, ax_pbe0])

    fig.subplots_adjust(left=0.10, right=0.92, top=0.97, bottom=0.07)

    if mesh_ref is not None:
        add_phase_colorbar(
            fig, mesh_ref,
            cax=margin_colorbar_cax(fig, ax_term_pbe, ax_bridge_pbe0, gap=0.01, width=0.012),
            label=r"$n_{\mathrm{O}}^{\mathrm{motif}} / n_{\mathrm{Ir}}$",
        )

    save_fig(fig, OUTPUT_PATH, dpi=DPI)
    print(f"\n[fig04] {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
