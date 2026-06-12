"""Figure 5 – Grand-canonical phase diagrams with ACE motif overlays.

Layout: 4 rows × 2 columns
  Row 1: Ensemble-averaged oxidation ⟨n_O/n_Ir⟩
  Row 2: Structure-cluster population (RGB mix of cluster colours)
  Row 3: Ir atom-motif population (RGB mix of Ir-A* colours)
  Row 4: O atom-motif population (RGB mix of O-A* colours)

Row 1 (PBE | PBE0): global-minimum boundaries + stoichiometry labels.

PBE uses electronic-only GC (no vib/ZPE), same thermo level as PBE0, from
``gc_variants.npz`` novib tensors.
Row 2: same boundaries + ``C*`` cluster label of the global minimum.
Rows 3–4: no phase overlay.

Cluster and atom rows colour each grid point by population-weighted mixing of
category colours (same logic as the fig 6 slice stacks), not ⟨index⟩ averages.

Output: ``figures/outputs/fig05_gc_clustering_maps.svg``
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

from figures.config import (
    ACE_FEATURES,
    GC_NPZ,
    GC_NPZ_PBE0,
    GC_VARIANTS_NPZ,
    OUTPUT_DIR,
    DPI,
)
from figures.utils.io import (
    ensure_dirs,
    load_ace_features,
    load_gc_npz,
    load_gc_pbe_novib_view,
)
from figures.utils.cache import ensure_for_figures
from figures.utils.join import (
    build_joined_table,
    cluster_population_fractions,
    ir_motif_population_fractions,
    o_motif_population_fractions,
)
from aist.ace.common import ace_atom_label

from figures.utils.plotting import (
    cluster_colors,
    cluster_overlay_labels,
    discrete_category_colorbar,
    draw_global_minimum_phase_overlay,
    draw_population_mixed_phase_panel,
    ir_motif_colors,
    o_motif_colors,
    save_fig,
    stoichiometry_overlay_labels,
    style_phase_ax,
)

OUTPUT_PATH = os.path.join(OUTPUT_DIR, "fig05_gc_clustering_maps.svg")

DATASETS = [
    {
        "label": "PBE",
        "load_gc": lambda: load_gc_pbe_novib_view(GC_VARIANTS_NPZ, GC_NPZ),
        "ace_csv": ACE_FEATURES["PBE"],
        "join_tag": "PBE",
    },
    {
        "label": "PBE0",
        "load_gc": lambda: load_gc_npz(GC_NPZ_PBE0),
        "ace_csv": ACE_FEATURES["PBE0"],
        "join_tag": "PBE0",
    },
]

PHASE_CMAP = "coolwarm"


def _scalar_ensemble_panel(
    ax,
    gc,
    field,
    title,
    cmap,
    vmin,
    vmax,
    cbar_label,
    *,
    overlay_labels: list[str] | None = None,
):
    """2D phase map of an ensemble-averaged scalar field."""
    Ts, Ps = gc["Ts"], gc["Ps"]
    Z = np.asarray(field, dtype=float)
    if not np.isfinite(Z).any():
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                fontsize=7, transform=ax.transAxes)
        return None

    if vmin is None:
        vmin = float(np.nanmin(Z))
    if vmax is None:
        vmax = float(np.nanmax(Z))
    if vmax <= vmin:
        vmax = vmin + 1.0

    mesh = ax.pcolormesh(
        Ts, Ps, Z, shading="auto",
        cmap=cmap,
        norm=Normalize(vmin=vmin, vmax=vmax),
        rasterized=True,
    )
    if overlay_labels is not None:
        draw_global_minimum_phase_overlay(
            ax, Ts, Ps, gc["phase_index"], overlay_labels,
        )
    style_phase_ax(ax)
    ax.set_title(title, fontsize=7)
    return mesh


def main() -> None:
    ensure_for_figures([5])
    ensure_dirs(OUTPUT_DIR)

    fig, axes = plt.subplots(4, 2, figsize=(180 / 25.4, 200 / 25.4), dpi=DPI)

    for col, ds in enumerate(DATASETS):
        try:
            gc = ds["load_gc"]()
        except FileNotFoundError:
            for row in range(4):
                axes[row, col].text(0.5, 0.5, "GC data missing",
                                    ha="center", va="center", fontsize=7,
                                    transform=axes[row, col].transAxes)
                axes[row, col].set_axis_off()
            continue
        rows = load_ace_features(ds["ace_csv"])
        jt = build_joined_table(gc, rows, ds.get("join_tag", "PBE"))

        k_struct = int(jt["cluster"].max()) + 1
        k_Ir = jt["ir_frac"].shape[1]
        k_O = jt["o_frac"].shape[1]

        cluster_pop = cluster_population_fractions(jt)
        ir_pop = ir_motif_population_fractions(jt)
        o_pop = o_motif_population_fractions(jt)

        stoich_labels = stoichiometry_overlay_labels(gc["formulas"][:jt["n"]])
        cluster_labels = cluster_overlay_labels(jt["cluster"])

        Ts, Ps = gc["Ts"], gc["Ps"]

        # Row 0: oxidation (from GC NPZ, already ensemble-averaged)
        mesh0 = _scalar_ensemble_panel(
            axes[0, col], gc, gc["x_per_non_o"],
            f"{ds['label']} — ⟨{r'n_O/n_{Ir}'}⟩",
            PHASE_CMAP, None, None, r"$n_O/n_{Ir}$",
            overlay_labels=stoich_labels,
        )
        if mesh0 is not None:
            plt.colorbar(mesh0, ax=axes[0, col], pad=0.02, fraction=0.046,
                         label=r"$n_O/n_{Ir}$").ax.tick_params(labelsize=7)

        struct_colors = cluster_colors(k_struct)
        draw_population_mixed_phase_panel(
            axes[1, col], Ts, Ps, cluster_pop, struct_colors,
            f"{ds['label']} — cluster population",
            overlay_labels=cluster_labels,
            phase_index=gc["phase_index"],
        )
        discrete_category_colorbar(
            axes[1, col],
            struct_colors,
            [f"C{c}" for c in range(k_struct)],
            cbar_label="Cluster",
        )

        ir_colors = ir_motif_colors(k_Ir)
        draw_population_mixed_phase_panel(
            axes[2, col], Ts, Ps, ir_pop, ir_colors,
            f"{ds['label']} — Ir motif population",
        )
        discrete_category_colorbar(
            axes[2, col],
            ir_colors,
            [ace_atom_label("Ir", c) for c in range(k_Ir)],
            cbar_label="Ir motif",
        )

        o_colors = o_motif_colors(k_O)
        draw_population_mixed_phase_panel(
            axes[3, col], Ts, Ps, o_pop, o_colors,
            f"{ds['label']} — O motif population",
        )
        discrete_category_colorbar(
            axes[3, col],
            o_colors,
            [ace_atom_label("O", c) for c in range(k_O)],
            cbar_label="O motif",
        )

    fig.tight_layout(pad=6 / 25.4)
    save_fig(fig, OUTPUT_PATH, dpi=DPI)
    print(f"\n[fig05] {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
