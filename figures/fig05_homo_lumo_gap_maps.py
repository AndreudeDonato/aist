"""Figure 5 – Per-Ir ⟨HOMO–LUMO⟩ gradient bars at low pO₂ + binned gap lines.

3×1 layout (88 mm wide):
  (0): PBE — population-weighted gradient bars vs T (Ir₃…Ir₉) at ``FIG04_LOW_PRESSURE_PA``
  (1): PBE0 — same
  (2): Mean HOMO–LUMO gap vs O fraction — one line per ``n_Ir`` (PBE blues, PBE0 reds)

Output: ``figures/outputs/fig05_homo_lumo_gap_maps.svg``
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

from figures.config import (
    ACE_FEATURES,
    DATASET_COLORS,
    DPI,
    FIG04_LOW_PRESSURE_PA,
    OUTPUT_DIR,
    SUPP_IR_COUNTS,
    TRAJ_PBE,
    TRAJ_PBE0,
)
from figures.utils.cache import ensure_for_figures
from figures.utils.comparison import (
    build_comparison_joined_table,
    homo_lumo_gaps_for_comparison,
    load_gc_comparison_view,
)
from figures.utils.gc_gradient_bars import (
    draw_scalar_gradient_bars,
    ensemble_scalar_vs_T_at_pressure,
    sync_temperature_xaxis_grid,
)
from figures.utils.homo_lumo import (
    HOMO_LUMO_CMAP,
    HOMO_LUMO_KEY,
    POP_OX_XMAX,
    homo_lumo_gap_lines_by_ir_size,
    oxidation_ratio_xgrid,
    structure_oxidation_ratios,
)
from figures.utils.io import ensure_dirs, load_ace_features, load_traj
from figures.utils.plotting import (
    TITLE_FONTSIZE,
    AXIS_LABEL_FONTSIZE,
    TICK_FONTSIZE,
    LEGEND_FONTSIZE,
    add_phase_colorbar,
    margin_colorbar_cax,
    save_fig,
)

OUTPUT_PATH = os.path.join(OUTPUT_DIR, "fig05_homo_lumo_gap_maps.svg")
FIG_W_MM = 88
BAR_ROW_H_MM = 52
LINES_H_MM = 50
FIG_H_MM = 2 * BAR_ROW_H_MM + LINES_H_MM
CBAR_GAP = 0.012
CBAR_WIDTH = 0.025
IR_COLOR_LO = 0.35
IR_COLOR_HI = 1.0

DATASETS = [
    {
        "label": "PBE",
        "load_gc": lambda: load_gc_comparison_view("PBE"),
        "ace_csv": ACE_FEATURES["PBE"],
        "traj": TRAJ_PBE,
        "join_tag": "PBE",
    },
    {
        "label": "PBE0",
        "load_gc": lambda: load_gc_comparison_view("PBE0"),
        "ace_csv": ACE_FEATURES["PBE0"],
        "traj": TRAJ_PBE0,
        "join_tag": "PBE0",
    },
]


def _load_column(ds: dict) -> dict | None:
    try:
        gc = ds["load_gc"]()
    except FileNotFoundError:
        return None
    traj_path = ds["traj"]
    if not os.path.exists(traj_path):
        return None
    rows = load_ace_features(ds["ace_csv"])
    jt = build_comparison_joined_table(gc, rows, ds.get("join_tag", "PBE"))
    gaps = homo_lumo_gaps_for_comparison(ds["label"], traj_path, jt["n"])
    if not np.isfinite(gaps).any():
        raise RuntimeError(
            f"[{ds['label']}] No {HOMO_LUMO_KEY} in {traj_path}. "
            "Rebuild from updated trajectories."
        )
    return {"label": ds["label"], "joined": jt, "gaps": gaps, "traj": traj_path}


def _gap_limits(columns: list[dict | None], pressure_pa: float) -> tuple[float, float]:
    vals: list[float] = []
    for col in columns:
        if col is None:
            continue
        for n in SUPP_IR_COUNTS:
            curve = ensemble_scalar_vs_T_at_pressure(
                col["joined"], n, col["gaps"], pressure_pa,
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
        vmax = vmin + 0.1
    return vmin, vmax


def _style_bar_axis(ax, title: str) -> None:
    ax.set_title(title, fontsize=TITLE_FONTSIZE, pad=3)
    ax.set_yscale("linear")
    ax.set_ylabel(r"$n_{\mathrm{Ir}}$", fontsize=AXIS_LABEL_FONTSIZE)
    ax.tick_params(axis="y", labelsize=TICK_FONTSIZE)


def _ir_size_color(norm, cmap, n_ir: int) -> tuple[float, float, float]:
    t = IR_COLOR_LO + (IR_COLOR_HI - IR_COLOR_LO) * norm(float(n_ir))
    return cmap(t)[:3]


def _lines_panel(
    ax,
    x_grid: np.ndarray,
    line_data: list[tuple[str, dict[int, np.ndarray]]],
) -> None:
    ir_norm = Normalize(vmin=float(min(SUPP_IR_COUNTS)), vmax=float(max(SUPP_IR_COUNTS)))
    blue_cmap = plt.get_cmap("Blues")
    red_cmap = plt.get_cmap("Reds")
    cmap_lookup = {"PBE": blue_cmap, "PBE0": red_cmap}

    y_vals: list[float] = []
    for label, lines in line_data:
        cmap = cmap_lookup[label]
        for n_ir, y in sorted(lines.items()):
            finite = y[np.isfinite(y)]
            if finite.size:
                y_vals.extend(finite.tolist())
            ax.plot(
                x_grid, y,
                color=_ir_size_color(ir_norm, cmap, n_ir),
                linewidth=1.0,
                zorder=2,
            )

    ax.set_xlim(float(x_grid[0]), POP_OX_XMAX)
    if y_vals:
        ymin, ymax = float(min(y_vals)), float(max(y_vals))
        pad = 0.05 * max(ymax - ymin, 0.1)
        ax.set_ylim(ymin - pad, ymax + pad)

    ax.set_xlabel(r"$n_O / n_{Ir}$", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("HOMO–LUMO gap (eV)", fontsize=AXIS_LABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)

    handles = [
        Line2D([], [], color=DATASET_COLORS["PBE"], lw=1.2, label="PBE"),
        Line2D([], [], color=DATASET_COLORS["PBE0"], lw=1.2, label="PBE0"),
    ]
    ax.legend(handles=handles, fontsize=LEGEND_FONTSIZE, frameon=False, loc="best")


def main() -> None:
    ensure_for_figures([5])
    ensure_dirs(OUTPUT_DIR)

    pressure_pa = FIG04_LOW_PRESSURE_PA
    columns = [_load_column(ds) for ds in DATASETS]
    if not any(c is not None for c in columns):
        raise FileNotFoundError("No GC data found for figure 5.")

    hl_vmin, hl_vmax = _gap_limits(columns, pressure_pa)

    loaded: list[tuple[str, list]] = []
    ratio_chunks: list[np.ndarray] = []
    for col in columns:
        if col is None:
            continue
        structures = list(load_traj(col["traj"]))
        loaded.append((col["label"], structures))
        ratio_chunks.append(structure_oxidation_ratios(structures))
    x_grid = oxidation_ratio_xgrid(np.concatenate(ratio_chunks))

    line_data = [
        (label, homo_lumo_gap_lines_by_ir_size(structures, SUPP_IR_COUNTS, x_grid))
        for label, structures in loaded
    ]

    fig = plt.figure(figsize=(FIG_W_MM / 25.4, FIG_H_MM / 25.4), dpi=DPI)
    gs = GridSpec(
        3, 1, figure=fig,
        height_ratios=[BAR_ROW_H_MM, BAR_ROW_H_MM, LINES_H_MM],
        hspace=0.38,
    )
    bar_axes = np.array([fig.add_subplot(gs[i, 0]) for i in range(2)])

    bar_titles = ("PBE — ⟨HOMO–LUMO⟩", "PBE0 — ⟨HOMO–LUMO⟩")
    for row, (col, title) in enumerate(zip(columns, bar_titles)):
        ax = bar_axes[row]
        if col is None:
            ax.text(
                0.5, 0.5, "GC data missing", ha="center", va="center",
                fontsize=AXIS_LABEL_FONTSIZE, transform=ax.transAxes,
            )
            ax.set_axis_off()
            continue
        draw_scalar_gradient_bars(
            ax, col["joined"], pressure_pa, SUPP_IR_COUNTS, col["gaps"],
            cmap=HOMO_LUMO_CMAP, vmin=hl_vmin, vmax=hl_vmax,
            section_kind="none",
        )
        ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
        _style_bar_axis(ax, title)

    ref = next(c for c in columns if c is not None)
    ts_ref = np.asarray(ref["joined"]["Ts"], dtype=float)
    sync_temperature_xaxis_grid(bar_axes.reshape(2, 1), ts_ref)

    lines_ax = fig.add_subplot(gs[2, 0])
    _lines_panel(lines_ax, x_grid, line_data)

    fig.subplots_adjust(left=0.18, right=0.78, top=0.97, bottom=0.08, hspace=0.38)

    mesh_hl = ScalarMappable(norm=Normalize(vmin=hl_vmin, vmax=hl_vmax), cmap=HOMO_LUMO_CMAP)
    mesh_hl.set_array([])
    add_phase_colorbar(
        fig, mesh_hl,
        cax=margin_colorbar_cax(
            fig, bar_axes[0], bar_axes[1],
            gap=CBAR_GAP, width=CBAR_WIDTH,
        ),
        label="HOMO–LUMO gap (eV)",
    )

    save_fig(fig, OUTPUT_PATH, dpi=DPI)
    print(f"\n[fig05] {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
