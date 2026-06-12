"""Figure 11 – Ensemble ⟨HOMO–LUMO gap⟩ phase maps (PBE | PBE0).

2×1 layout:
  (0): PBE — ⟨HOMO–LUMO⟩ on T × pO₂, stoichiometry overlay
  (1): PBE0 — same

Gap per structure from ``atoms.info['HOMO-LUMO']`` (eV); map field is ``Pgi``-weighted.

Output: ``figures/outputs/fig11_homo_lumo_gap_maps.svg``
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

from figures.config import (
    DPI,
    GC_NPZ,
    GC_NPZ_PBE0,
    OUTPUT_DIR,
    TRAJ_PBE,
    TRAJ_PBE0,
)
from figures.utils.cache import ensure_for_figures
from figures.utils.io import ensure_dirs, load_gc_npz, load_traj
from figures.utils.join import ensemble_weighted_mean_map
from figures.utils.plotting import (
    draw_global_minimum_phase_overlay,
    save_fig,
    stoichiometry_overlay_labels,
    style_phase_ax,
)

OUTPUT_PATH = os.path.join(OUTPUT_DIR, "fig11_homo_lumo_gap_maps.svg")
FIG_W_MM = 88
FIG_H_MM = 130
HOMO_LUMO_KEY = "HOMO-LUMO"
CMAP = "Greens"

DATASETS = [
    {"label": "PBE", "npz": GC_NPZ, "traj": TRAJ_PBE},
    {"label": "PBE0", "npz": GC_NPZ_PBE0, "traj": TRAJ_PBE0},
]


def _homo_lumo_gaps(structures) -> np.ndarray:
    gaps = np.full(len(structures), np.nan, dtype=float)
    for i, atoms in enumerate(structures):
        val = atoms.info.get(HOMO_LUMO_KEY)
        if val is not None:
            gaps[i] = float(val)
    return gaps


def _gap_map(gc, gaps: np.ndarray) -> np.ndarray:
    n = min(gaps.size, gc["Pgi"].shape[0])
    return ensemble_weighted_mean_map(gc["Pgi"][:n], gaps[:n])


def _gap_panel(ax, gc, gaps, overlay_labels, title, vmin, vmax):
    Ts, Ps = gc["Ts"], gc["Ps"]
    Z = _gap_map(gc, gaps)
    if not np.isfinite(Z).any():
        ax.text(0.5, 0.5, "No HOMO–LUMO data", ha="center", va="center",
                fontsize=7, transform=ax.transAxes)
        style_phase_ax(ax)
        ax.set_title(title, fontsize=7)
        return None

    mesh = ax.pcolormesh(
        Ts, Ps, Z, shading="auto",
        cmap=CMAP, norm=Normalize(vmin=vmin, vmax=vmax),
        rasterized=True,
    )
    draw_global_minimum_phase_overlay(
        ax, Ts, Ps, gc["phase_index"], overlay_labels,
    )
    style_phase_ax(ax)
    ax.set_title(title, fontsize=7)
    return mesh


def main() -> None:
    ensure_for_figures([11])
    ensure_dirs(OUTPUT_DIR)

    panels: list[tuple[dict, np.ndarray, list[str], str]] = []
    all_gaps: list[np.ndarray] = []

    for ds in DATASETS:
        if not os.path.exists(ds["npz"]):
            continue
        gc = load_gc_npz(ds["npz"])
        structures = list(load_traj(ds["traj"]))
        gaps = _homo_lumo_gaps(structures)
        if not np.isfinite(gaps).any():
            raise RuntimeError(
                f"[{ds['label']}] No {HOMO_LUMO_KEY} in {ds['traj']}. "
                "Rebuild from updated trajectories."
            )
        labels = stoichiometry_overlay_labels(gc["formulas"])
        panels.append((gc, gaps, labels, ds["label"]))
        all_gaps.append(gaps[np.isfinite(gaps)])

    if not panels:
        raise FileNotFoundError("No GC NPZ files found for figure 9.")

    combined = np.concatenate(all_gaps)
    vmin = float(np.min(combined))
    vmax = float(np.max(combined))
    if vmax <= vmin:
        vmax = vmin + 0.1

    fig, axes = plt.subplots(
        2, 1, figsize=(FIG_W_MM / 25.4, FIG_H_MM / 25.4), dpi=DPI,
    )
    mesh_ref = None
    for ax, (gc, gaps, overlay_labels, label) in zip(axes, panels):
        mesh = _gap_panel(
            ax, gc, gaps, overlay_labels,
            f"{label} — ⟨HOMO–LUMO⟩",
            vmin, vmax,
        )
        if mesh_ref is None:
            mesh_ref = mesh

    for ax in axes[len(panels):]:
        ax.set_axis_off()

    fig.tight_layout(pad=6 / 25.4, rect=[0, 0, 0.84, 1])
    if mesh_ref is not None:
        cbar = fig.colorbar(
            mesh_ref,
            ax=axes[:len(panels)],
            location="right",
            pad=0.06,
            fraction=0.04,
            shrink=0.92,
            label="HOMO–LUMO gap (eV)",
        )
        cbar.ax.tick_params(labelsize=7)

    save_fig(fig, OUTPUT_PATH, dpi=DPI)
    print(f"\n[fig11] {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
