"""Figure 6 alt – Terminal / bridging O per Ir phase maps (PBE | PBE0).

2×2 layout with shared axes and colorbar:
  Row 1: ⟨n_O^{terminal} / n_Ir⟩   — PBE | PBE0
  Row 2: ⟨n_O^{bridging} / n_Ir⟩ — PBE | PBE0

Ensemble average on the grand-canonical ``(T, p_O₂)`` grid:
``Σ_i Pgi_i · n_O,i · f_{motif,i} / Σ_i Pgi_i · n_Ir,i``.

Output: ``figures/outputs/fig06_gc_pressure_slices_alt.svg``
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

from figures.config import (
    ACE_FEATURES,
    DPI,
    FIG06_O_BRIDGING_IDX,
    FIG06_O_TERMINAL_IDX,
    GC_NPZ,
    GC_NPZ_PBE0,
    OUTPUT_DIR,
    PHASE_CMAP,
)
from figures.utils.cache import ensure_for_figures
from figures.utils.io import ensure_dirs, load_ace_features, load_gc_npz
from figures.utils.join import build_joined_table, ensemble_o_motif_per_ir_map
from figures.utils.plotting import add_phase_colorbar, margin_colorbar_cax, save_fig, style_phase_ax

OUTPUT_PATH = os.path.join(OUTPUT_DIR, "fig06_gc_pressure_slices_alt.svg")
FIG_W_MM = 180
FIG_H_MM = 120

DATASETS = [
    {"label": "PBE", "npz": GC_NPZ, "ace_csv": ACE_FEATURES["PBE"]},
    {"label": "PBE0", "npz": GC_NPZ_PBE0, "ace_csv": ACE_FEATURES["PBE0"]},
]

ROWS = [
    (FIG06_O_TERMINAL_IDX, r"terminal O / $n_{\mathrm{Ir}}$"),
    (FIG06_O_BRIDGING_IDX, r"bridging O / $n_{\mathrm{Ir}}$"),
]


def _phase_panel(ax, joined: dict, o_idx: int, title: str, vmin: float, vmax: float):
    Ts, Ps = joined["Ts"], joined["Ps"]
    Z = ensemble_o_motif_per_ir_map(joined, o_idx)
    if not np.isfinite(Z).any():
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                fontsize=7, transform=ax.transAxes)
        style_phase_ax(ax)
        ax.set_title(title, fontsize=7)
        return None

    mesh = ax.pcolormesh(
        Ts, Ps, Z, shading="auto",
        cmap=PHASE_CMAP,
        norm=Normalize(vmin=vmin, vmax=vmax),
        rasterized=True,
    )
    style_phase_ax(ax)
    ax.set_title(title, fontsize=7)
    return mesh


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


def main() -> None:
    ensure_for_figures([6])
    ensure_dirs(OUTPUT_DIR)

    joined_by_label: dict[str, dict] = {}
    for ds in DATASETS:
        if not os.path.exists(ds["npz"]):
            continue
        gc = load_gc_npz(ds["npz"])
        rows = load_ace_features(ds["ace_csv"])
        joined_by_label[ds["label"]] = build_joined_table(gc, rows, ds["label"])

    if not joined_by_label:
        raise FileNotFoundError("No GC data found for figure 6 alt.")

    all_vals: list[float] = []
    for o_idx, _ in ROWS:
        vmin, vmax = _field_limits(joined_by_label, o_idx)
        all_vals.extend([vmin, vmax])
    zmin, zmax = float(min(all_vals)), float(max(all_vals))
    if zmax <= zmin:
        zmax = zmin + 1.0

    fig, axes = plt.subplots(
        2, 2,
        figsize=(FIG_W_MM / 25.4, FIG_H_MM / 25.4),
        dpi=DPI,
        sharex=True,
        sharey=True,
    )

    mesh_ref = None
    for row, (o_idx, row_label) in enumerate(ROWS):
        for col, label in enumerate(("PBE", "PBE0")):
            ax = axes[row, col]
            jt = joined_by_label.get(label)
            if jt is None:
                ax.text(0.5, 0.5, "GC data missing", ha="center", va="center",
                        fontsize=7, transform=ax.transAxes)
                ax.set_axis_off()
                continue
            title = f"{label} — ⟨{row_label}⟩"
            mesh = _phase_panel(ax, jt, o_idx, title, zmin, zmax)
            if mesh_ref is None and mesh is not None:
                mesh_ref = mesh

    for ax in axes[0, :]:
        ax.tick_params(labelbottom=False)
    for ax in axes[:, 1]:
        ax.tick_params(labelleft=False)
    for ax in axes[1, :]:
        ax.set_xlabel("T (K)", fontsize=8)

    fig.tight_layout(pad=6 / 25.4, rect=[0, 0, 0.92, 1])
    if mesh_ref is not None:
        add_phase_colorbar(
            fig, mesh_ref,
            cax=margin_colorbar_cax(fig, axes[0, 0], axes[1, 1]),
            label=r"$n_{\mathrm{O}}^{\mathrm{motif}} / n_{\mathrm{Ir}}$",
        )

    save_fig(fig, OUTPUT_PATH, dpi=DPI)
    print(f"\n[fig06_alt] {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
