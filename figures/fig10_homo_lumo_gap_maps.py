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
from collections import defaultdict

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
from figures.utils.io import (
    ensure_dirs,
    load_gc_npz,
    load_gc_pbe_novib_view,
    load_traj,
)
from figures.utils.join import ensemble_weighted_mean_map
from figures.utils.plotting import (
    draw_global_minimum_phase_overlay,
    save_fig,
    stoichiometry_overlay_labels,
    style_phase_ax,
)

OUTPUT_PATH = os.path.join(OUTPUT_DIR, "fig10_homo_lumo_gap_maps.svg")
FIG_W_MM = 88
FIG_H_MM = 195
HOMO_LUMO_KEY = "HOMO-LUMO"
CMAP = "Greens"

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


def _gap_minmax_vs_oxidation(structures) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Min/max HOMO–LUMO gap at each stoichiometry (grouped by n_Ir, n_O)."""
    by_stoich: dict[tuple[int, int], list[float]] = defaultdict(list)
    for atoms in structures:
        syms = atoms.get_chemical_symbols()
        n_ir = sum(s == "Ir" for s in syms)
        n_o = sum(s == "O" for s in syms)
        gap = atoms.info.get(HOMO_LUMO_KEY)
        if gap is None or n_ir == 0:
            continue
        by_stoich[(n_ir, n_o)].append(float(gap))

    rows: list[tuple[float, float, float]] = []
    for (n_ir, n_o), gs in by_stoich.items():
        x = n_o / n_ir
        rows.append((x, min(gs), max(gs)))
    rows.sort(key=lambda r: r[0])
    xs = np.asarray([r[0] for r in rows], dtype=float)
    ymin = np.asarray([r[1] for r in rows], dtype=float)
    ymax = np.asarray([r[2] for r in rows], dtype=float)
    return xs, ymin, ymax


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


def _envelope_panel(ax, envelope_data: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]) -> None:
    for label, xs, ymin, ymax in envelope_data:
        color = DATASET_COLORS.get(label, "#333333")
        ax.fill_between(xs, ymin, ymax, color=color, alpha=0.25, linewidth=0)
        ax.plot(xs, ymin, color=color, linewidth=1.0, linestyle="-")
        ax.plot(xs, ymax, color=color, linewidth=1.0, linestyle="-", label=label)

    ax.set_xlabel("O fraction (n_O / n_Ir)", fontsize=7)
    ax.set_ylabel("HOMO–LUMO gap (eV)", fontsize=7)
    ax.set_title("HOMO–LUMO range vs O fraction", fontsize=7)
    ax.tick_params(axis="both", labelsize=6)
    leg = ax.legend(fontsize=5, frameon=False, loc="best", ncol=2)
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
        gaps = _homo_lumo_gaps(structures)
        if not np.isfinite(gaps).any():
            raise RuntimeError(
                f"[{ds['label']}] No {HOMO_LUMO_KEY} in {ds['traj']}. "
                "Rebuild from updated trajectories."
            )
        labels = stoichiometry_overlay_labels(gc["formulas"])
        map_panels.append((gc, gaps, labels, ds["label"]))
        all_gaps.append(gaps[np.isfinite(gaps)])

        xs, ymin, ymax = _gap_minmax_vs_oxidation(structures)
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
        cbar = fig.colorbar(
            mesh_ref,
            ax=map_axes[:len(map_panels)],
            location="right",
            pad=0.04,
            fraction=0.03,
            shrink=0.88,
            label="HOMO–LUMO gap (eV)",
        )
        cbar.ax.tick_params(labelsize=7)

    save_fig(fig, OUTPUT_PATH, dpi=DPI)
    print(f"\n[fig10] {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
