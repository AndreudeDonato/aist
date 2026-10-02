"""Supplementary – Ranked structure populations by Ir size at low pO₂, PBE | PBE0.

Former Figure 8. 2×4 layout (180 mm wide) at ``FIG04_LOW_PRESSURE_PA``:
  Row 0 — no vib; row 1 — vib (same energy order as row 0).
  Columns — PBE ~273 K | PBE ~500 K | PBE0 ~273 K | PBE0 ~500 K (actual grid ``T``).

Within each panel, one line per ``n_Ir`` (Ir₄…Ir₆): population (%) vs
lowest-energy structures s0…s4 (s0 = global minimum; order fixed by novib G).

Output: ``figures/outputs/supp_ranked_pop_by_ir.svg``
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

from figures.config import (
    DPI,
    FIG04_LOW_PRESSURE_PA,
    OUTPUT_DIR,
    THERMO_IR_COUNTS,
)
from figures.utils.cache import ensure_for_supplementary
from figures.utils.datasets import DATASETS, dataset_inputs
from figures.utils.gc_subset import ranked_structure_pops_by_ir_at_fixed_point
from figures.utils.io import ensure_dirs, load_ace_features, load_gc_npz, load_variants_like
from figures.utils.join import build_joined_table
from figures.utils.plotting import (
    AXIS_LABEL_FONTSIZE,
    LEGEND_FONTSIZE,
    TICK_FONTSIZE,
    TITLE_FONTSIZE,
    save_fig,
)

OUTPUT_PATH = os.path.join(OUTPUT_DIR, "supp_ranked_pop_by_ir.svg")
FIG_W_MM = 180
ROW_H_MM = 48
FIG_H_MM = 2 * ROW_H_MM
IR_COUNTS = THERMO_IR_COUNTS
IR_CMAP = "winter"
TEMPERATURES = (273.0, 500.0)
N_STRUCTURES = 5

ROW_SPECS = (
    ("No vib", "novib"),
    ("Vib", "vib"),
)


def _ir_colormap():
    norm = Normalize(vmin=float(min(IR_COUNTS)), vmax=float(max(IR_COUNTS)))
    return norm, plt.get_cmap(IR_CMAP)


def _format_temperature(t_k: float) -> str:
    return f"{t_k:.0f} K"


def _truncate_lines(
    lines_by_ir: dict[int, tuple[np.ndarray, np.ndarray]],
    n: int = N_STRUCTURES,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    return {
        k: (novib[:n], vib[:n])
        for k, (novib, vib) in lines_by_ir.items()
    }


def _panel_limits(
    lines_by_ir: dict[int, tuple[np.ndarray, np.ndarray]],
    *,
    row_kind: str,
) -> float:
    ymax = 1.0
    key = 0 if row_kind == "novib" else 1
    for pops_n, pops_v in lines_by_ir.values():
        pops = pops_n if key == 0 else pops_v
        if pops.size:
            ymax = max(ymax, float(np.nanmax(pops[:N_STRUCTURES])) * 1.08)
    return ymax


def _edge_to_edge_x(n: int) -> np.ndarray:
    if n <= 1:
        return np.zeros(max(n, 1), dtype=float)
    return np.linspace(0.0, float(n - 1), n, dtype=float)


def _plot_ranked_panel(
    ax,
    lines_by_ir: dict[int, tuple[np.ndarray, np.ndarray]],
    norm,
    cmap,
    *,
    row_kind: str,
    ymax: float,
    show_xlabel: bool,
    show_ylabel: bool,
    panel_title: str,
) -> None:
    key = 0 if row_kind == "novib" else 1
    x = _edge_to_edge_x(N_STRUCTURES)
    for n_ir in IR_COUNTS:
        pair = lines_by_ir.get(n_ir)
        if pair is None:
            continue
        pops = pair[key][:N_STRUCTURES]
        if pops.size == 0:
            continue
        ax.plot(
            x[: pops.size], pops,
            color=cmap(norm(float(n_ir))),
            linewidth=1.2,
            label=f"Ir{n_ir}",
            zorder=2,
            solid_capstyle="butt",
        )

    ax.set_xlim(0.0, float(N_STRUCTURES - 1))
    ax.set_ylim(0.0, ymax)
    ax.set_xticks(x)
    ax.set_xticklabels([f"s{k}" for k in range(N_STRUCTURES)], fontsize=TICK_FONTSIZE)
    ax.set_xlabel("Lowest energy structures" if show_xlabel else "", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Population (%)" if show_ylabel else "", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title(panel_title, fontsize=TITLE_FONTSIZE, loc="left")
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    ax.margins(x=0)


def _dataset_panels(label: str, pressure_pa: float) -> tuple[dict, dict]:
    inp = dataset_inputs(label)
    gc = load_gc_npz(inp.gc_npz, light=True)
    vd = load_variants_like(inp.variants_npz, gc, ("Ts", "Ps", "novib_Pgi", "full_Pgi", "novib_G_tensor"))
    jt = build_joined_table(gc, load_ace_features(inp.ace_features), label)
    n_ir = np.asarray(jt["n_Ir"], dtype=int)
    formulas = np.asarray(jt["formulas_gc"], dtype=object)

    panels, meta = {}, {}
    for T in TEMPERATURES:
        t_used, p_used, lines_by_ir = ranked_structure_pops_by_ir_at_fixed_point(
            vd["novib_Pgi"], vd["full_Pgi"], vd["novib_G_tensor"], n_ir, vd["Ts"], vd["Ps"], T, pressure_pa,
            ir_counts=IR_COUNTS,
            formulas=formulas,
            max_structures=N_STRUCTURES,
        )
        meta[(label, T)] = (t_used, p_used)
        lines_by_ir = _truncate_lines(lines_by_ir)
        panels[(label, T, "novib")] = lines_by_ir
        panels[(label, T, "vib")] = lines_by_ir
    return panels, meta


def main() -> None:
    ensure_for_supplementary(["supp_ranked_pop_by_ir"])
    ensure_dirs(OUTPUT_DIR)

    pressure_pa = FIG04_LOW_PRESSURE_PA
    panels: dict = {}
    meta: dict = {}
    for label in DATASETS:
        p, m = _dataset_panels(label, pressure_pa)
        panels.update(p)
        meta.update(m)
    columns = [(label, T) for label in DATASETS for T in TEMPERATURES]

    norm, cmap = _ir_colormap()
    fig, axes = plt.subplots(
        2, len(columns),
        figsize=(FIG_W_MM / 25.4, FIG_H_MM / 25.4),
        dpi=DPI,
        sharey="row",
    )

    for row, (row_title, row_kind) in enumerate(ROW_SPECS):
        ymax = 1.0
        for label, T in columns:
            ymax = max(
                ymax,
                _panel_limits(panels[(label, T, row_kind)], row_kind=row_kind),
            )
        for col, (label, T) in enumerate(columns):
            lines_by_ir = panels[(label, T, row_kind)]
            t_used, _ = meta[(label, T)]
            panel_title = f"{label}, {row_title.lower()}, {_format_temperature(t_used)}"
            _plot_ranked_panel(
                axes[row, col],
                lines_by_ir,
                norm,
                cmap,
                row_kind=row_kind,
                ymax=ymax,
                show_xlabel=(row == len(ROW_SPECS) - 1),
                show_ylabel=(col == 0),
                panel_title=panel_title,
            )

    handles, labels = axes[0, 1].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            fontsize=LEGEND_FONTSIZE,
            frameon=False,
            ncol=len(IR_COUNTS),
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
        )

    fig.subplots_adjust(left=0.08, right=0.99, top=0.86, bottom=0.12, hspace=0.45, wspace=0.22)
    save_fig(fig, OUTPUT_PATH, dpi=DPI)
    print(f"[supp] {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
