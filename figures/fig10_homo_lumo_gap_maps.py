"""Figure 10 – Ensemble ⟨HOMO–LUMO gap⟩ phase maps (PBE | PBE0) + oxidation envelope.

3×1 layout:
  (0): PBE (no vib/ZPE) — ⟨HOMO–LUMO⟩ on T × pO₂, stoichiometry overlay
  (1): PBE0 — same
  (2): HOMO–LUMO min/max vs oxygen fraction (n_O/n_Ir), shaded band per dataset

PBE uses electronic-only GC from ``gc_variants.npz`` (fair comparison with PBE0).
Gap per structure from ``atoms.info['HOMO-LUMO']`` (eV); map field is ``Pgi``-weighted.

Output: ``figures/outputs/fig10_homo_lumo_gap_maps.svg``
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec

from figures.config import (
    DATASET_COLORS,
    DPI,
    GC_NPZ,
    GC_NPZ_PBE0,
    GC_VARIANTS_NPZ,
    OUTPUT_DIR,
    TRAJ_PBE,
    TRAJ_PBE0,
)
from figures.utils.cache import ensure_for_figures
from figures.utils.homo_lumo import (
    HOMO_LUMO_CMAP,
    HOMO_LUMO_KEY,
    homo_lumo_gap_map,
    homo_lumo_gap_minmax_vs_oxidation,
    homo_lumo_gaps,
)
from figures.utils.io import (
    ensure_dirs,
    load_gc_npz,
    load_gc_pbe_novib_view,
    load_traj,
)
from figures.utils.plotting import (
    TITLE_FONTSIZE,
    AXIS_LABEL_FONTSIZE,
    TICK_FONTSIZE,
    LEGEND_FONTSIZE,
    add_phase_colorbar,
    draw_global_minimum_phase_overlay,
    margin_colorbar_cax,
    save_fig,
    stoichiometry_overlay_labels,
    style_phase_ax,
)

OUTPUT_PATH = os.path.join(OUTPUT_DIR, "fig10_homo_lumo_gap_maps.svg")
FIG_W_MM = 88
FIG_H_MM = 195

DATASETS = [
    {
        "label": "PBE",
        "load_gc": lambda: load_gc_pbe_novib_view(GC_VARIANTS_NPZ, GC_NPZ),
        "traj": TRAJ_PBE,
    },
    {
        "label": "PBE0",
        "load_gc": lambda: load_gc_npz(GC_NPZ_PBE0),
        "traj": TRAJ_PBE0,
    },
]


def _gap_panel(ax, gc, gaps, overlay_labels, title, vmin, vmax):
    Ts, Ps = gc["Ts"], gc["Ps"]
    Z = homo_lumo_gap_map(gc["Pgi"], gaps)
    if not np.isfinite(Z).any():
        ax.text(0.5, 0.5, "No HOMO–LUMO data", ha="center", va="center",
                fontsize=AXIS_LABEL_FONTSIZE, transform=ax.transAxes)
        style_phase_ax(ax)
        ax.set_title(title, fontsize=TITLE_FONTSIZE)
        return None

    mesh = ax.pcolormesh(
        Ts, Ps, Z, shading="auto",
        cmap=HOMO_LUMO_CMAP, norm=Normalize(vmin=vmin, vmax=vmax),
        rasterized=True,
    )
    draw_global_minimum_phase_overlay(
        ax, Ts, Ps, gc["phase_index"], overlay_labels,
    )
    style_phase_ax(ax)
    ax.set_title(title, fontsize=TITLE_FONTSIZE)
    return mesh


def _envelope_panel(ax, envelope_data: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]) -> None:
    for label, xs, ymin, ymax in envelope_data:
        color = DATASET_COLORS.get(label, "#333333")
        ax.fill_between(xs, ymin, ymax, color=color, alpha=0.25, linewidth=0)
        ax.plot(xs, ymin, color=color, linewidth=1.0, linestyle="-")
        ax.plot(xs, ymax, color=color, linewidth=1.0, linestyle="-", label=label)

    ax.set_xlabel("O fraction (n_O / n_Ir)", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("HOMO–LUMO gap (eV)", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title("HOMO–LUMO range vs O fraction", fontsize=TITLE_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    leg = ax.legend(fontsize=LEGEND_FONTSIZE, frameon=False, loc="best", ncol=2)
    for handle in leg.legend_handles:
        handle.set_alpha(1.0)


def main() -> None:
    ensure_for_figures([10])
    ensure_dirs(OUTPUT_DIR)

    map_panels: list[tuple[dict, np.ndarray, list[str], str]] = []
    envelope_data: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
    all_gaps: list[np.ndarray] = []

    for ds in DATASETS:
        if not os.path.exists(ds["traj"]):
            continue
        gc = ds["load_gc"]()
        structures = list(load_traj(ds["traj"]))
        gaps = homo_lumo_gaps(structures)
        if not np.isfinite(gaps).any():
            raise RuntimeError(
                f"[{ds['label']}] No {HOMO_LUMO_KEY} in {ds['traj']}. "
                "Rebuild from updated trajectories."
            )
        labels = stoichiometry_overlay_labels(gc["formulas"])
        map_panels.append((gc, gaps, labels, ds["label"]))
        all_gaps.append(gaps[np.isfinite(gaps)])

        xs, ymin, ymax = homo_lumo_gap_minmax_vs_oxidation(structures)
        if xs.size:
            envelope_data.append((ds["label"], xs, ymin, ymax))

    if not map_panels:
        raise FileNotFoundError("No GC data found for figure 10.")

    combined = np.concatenate(all_gaps)
    vmin = float(np.min(combined))
    vmax = float(np.max(combined))
    if vmax <= vmin:
        vmax = vmin + 0.1

    fig = plt.figure(figsize=(FIG_W_MM / 25.4, FIG_H_MM / 25.4), dpi=DPI)
    gs = GridSpec(3, 1, figure=fig, height_ratios=[1.0, 1.0, 0.9], hspace=0.28)
    map_axes = [fig.add_subplot(gs[i, 0]) for i in range(2)]
    envelope_ax = fig.add_subplot(gs[2, 0])

    mesh_ref = None
    for ax, (gc, gaps, overlay_labels, label) in zip(map_axes, map_panels):
        mesh = _gap_panel(
            ax, gc, gaps, overlay_labels,
            f"{label} — ⟨HOMO–LUMO⟩",
            vmin, vmax,
        )
        if mesh_ref is None:
            mesh_ref = mesh

    for ax in map_axes[len(map_panels):]:
        ax.set_axis_off()

    _envelope_panel(envelope_ax, envelope_data)

    fig.subplots_adjust(left=0.14, right=0.84, top=0.97, bottom=0.07, hspace=0.28)
    if mesh_ref is not None:
        add_phase_colorbar(
            fig, mesh_ref,
            cax=margin_colorbar_cax(fig, map_axes[0], map_axes[-1]),
            label="HOMO–LUMO gap (eV)",
        )

    save_fig(fig, OUTPUT_PATH, dpi=DPI)
    print(f"\n[fig10] {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
