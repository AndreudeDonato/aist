"""Figure 6 – Ambient-pO₂ motif and cluster populations vs temperature.

2×2 layout:
  Row 0: PBE atom motifs | PBE0 atom motifs  (Ir-A* then O-A*, grouped)
  Row 1: PBE clusters    | PBE0 clusters

x-axis: ``T`` at ``FIG06_PRESSURE_PA`` (1 atm). The Ir/O motif boundary tracks oxidation.

Output: ``figures/outputs/fig06_gc_pressure_slices.svg``
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt

from aist.ace.common import ace_atom_label

from figures.config import (
    ACE_FEATURES,
    DPI,
    FIG06_PRESSURE_PA,
    FIG06_SMOOTH_SIGMA,
    GC_NPZ,
    GC_NPZ_PBE0,
    OUTPUT_DIR,
)
from figures.utils.cache import ensure_for_figures
from figures.utils.io import ensure_dirs, load_ace_features, load_gc_npz
from figures.utils.join import (
    atom_motif_population_vs_T_at_pressure,
    build_joined_table,
    cluster_population_vs_T_at_pressure,
)
from figures.utils.plotting import (
    cluster_colors,
    draw_population_stack,
    ir_motif_colors,
    o_motif_colors,
    save_fig,
)

OUTPUT_PATH = os.path.join(OUTPUT_DIR, "fig06_gc_pressure_slices.svg")
FIG_W_MM = 180
ROW_H_MM = 60
FIG_H_MM = 2 * ROW_H_MM

DATASETS = [
    {"label": "PBE", "npz": GC_NPZ, "ace_csv": ACE_FEATURES["PBE"]},
    {"label": "PBE0", "npz": GC_NPZ_PBE0, "ace_csv": ACE_FEATURES["PBE0"]},
]


def _panel_title(label: str, panel_kind: str, pressure_pa: float) -> str:
    return (
        f"{label} — {panel_kind} — "
        rf"$p_{{\mathrm{{O_2}}}}={pressure_pa:.3g}$ Pa"
    )


def _draw_atom_panel(ax, joined: dict, label: str, pressure_pa: float) -> None:
    Ts, p_used, Y = atom_motif_population_vs_T_at_pressure(
        joined, pressure_pa, smooth_sigma=FIG06_SMOOTH_SIGMA,
    )
    k_ir = joined["ir_frac"].shape[1]
    k_o = joined["o_frac"].shape[1]
    colors = ir_motif_colors(k_ir) + o_motif_colors(k_o)
    labels = [ace_atom_label("Ir", c) for c in range(k_ir)]
    labels += [ace_atom_label("O", c) for c in range(k_o)]
    draw_population_stack(
        ax, Ts, Y, colors,
        labels=labels,
        xlabel="T (K)",
        ylabel="Atom motif pop. (%)",
        title=_panel_title(label, "atom motifs", p_used),
        legend_ncol=2,
        boundary_after=k_ir,
    )
    ax.tick_params(axis="both", labelsize=6)


def _draw_cluster_panel(ax, joined: dict, label: str, pressure_pa: float) -> None:
    Ts, p_used, Y = cluster_population_vs_T_at_pressure(
        joined, pressure_pa, smooth_sigma=FIG06_SMOOTH_SIGMA,
    )
    k = int(joined["cluster"].max()) + 1
    colors = cluster_colors(k)
    labels = [f"C{c}" for c in range(k)]
    draw_population_stack(
        ax, Ts, Y, colors,
        labels=labels,
        xlabel="T (K)",
        ylabel="Cluster pop. (%)",
        title=_panel_title(label, "structure clusters", p_used),
        legend_ncol=2,
    )
    ax.tick_params(axis="both", labelsize=6)


def main() -> None:
    ensure_for_figures([6])
    ensure_dirs(OUTPUT_DIR)

    fig, axes = plt.subplots(2, 2, figsize=(FIG_W_MM / 25.4, FIG_H_MM / 25.4), dpi=DPI)

    joined_tables: list[tuple[dict, dict | None]] = []
    for ds in DATASETS:
        if not os.path.exists(ds["npz"]):
            joined_tables.append((ds, None))
            continue
        gc = load_gc_npz(ds["npz"])
        rows = load_ace_features(ds["ace_csv"])
        joined_tables.append((ds, build_joined_table(gc, rows, ds["label"])))

    for idx, (ds, jt) in enumerate(joined_tables):
        ax_atom = axes[0, idx]
        ax_cluster = axes[1, idx]
        if jt is None:
            for ax in (ax_atom, ax_cluster):
                ax.text(
                    0.5, 0.5, "GC data missing",
                    ha="center", va="center", fontsize=7,
                    transform=ax.transAxes,
                )
                ax.set_axis_off()
            continue

        _draw_atom_panel(ax_atom, jt, ds["label"], FIG06_PRESSURE_PA)
        _draw_cluster_panel(ax_cluster, jt, ds["label"], FIG06_PRESSURE_PA)

    fig.tight_layout(pad=6 / 25.4)
    save_fig(fig, OUTPUT_PATH, dpi=DPI)
    print(f"\n[fig06] {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
