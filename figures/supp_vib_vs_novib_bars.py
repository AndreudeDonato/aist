"""Supplementary – Vib vs no-vib oxidation gradient bars at ambient pO₂, PBE | PBE0.

Former Figure 7 (gradient-bar version). Fig. 3 row-2 style: per-Ir scalar gradient bars of
``n_O/n_Ir`` vs ``T`` at ``AMBIENT_PO2_PA`` (Ir₄…Ir₆), 2×4 (columns PBE full | PBE no vib |
PBE0 full | PBE0 no vib):

  row 0  with ensemble
  row 1  single minimum

Output: figures/outputs/supp_vib_vs_novib_bars.svg
"""
from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in [_ROOT, os.path.join(_ROOT, "analysis")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from figures.config import (
    AMBIENT_PO2_PA,
    DPI,
    OUTPUT_DIR,
    PHASE_CMAP,
    THERMO_IR_COUNTS,
)
from figures.utils.cache import ensure_for_supplementary
from figures.utils.comparison import build_comparison_joined_table, load_gc_comparison_view
from figures.utils.datasets import DATASETS, dataset_inputs
from figures.utils.gc_gradient_bars import (
    draw_scalar_gradient_bars,
    draw_winner_scalar_gradient_bars,
    ensemble_scalar_vs_T_at_pressure,
    sync_temperature_xaxis_grid,
    winner_scalar_vs_T_at_pressure,
)
from figures.utils.io import ensure_dirs, load_ace_features, load_npz, load_variants_like
from figures.utils.plotting import (
    AXIS_LABEL_FONTSIZE,
    TICK_FONTSIZE,
    TITLE_FONTSIZE,
    add_phase_colorbar,
    margin_colorbar_cax,
    save_fig,
)

OUTPUT_PATH = os.path.join(OUTPUT_DIR, "supp_vib_vs_novib_bars.svg")
FIG_W_MM = 180
BAR_ROW_H_MM = 50
FIG_H_MM = 2 * BAR_ROW_H_MM
LAYOUT_RIGHT = 0.92
CBAR_GAP = 0.008
CBAR_WIDTH = 0.012
HSPACE = 0.35
WSPACE = 0.10
IR_COUNTS = list(THERMO_IR_COUNTS)

TITLES = {
    (0, 0): "full (vib + ensemble)",
    (0, 1): "no vib/ZPE, ensemble",
    (1, 0): "with vib, single minimum",
    (1, 1): "no vib/ZPE, single minimum",
}


def _format_po2(pa: float) -> str:
    if pa >= 1.0:
        return f"{pa:.3g}"
    exp = int(np.floor(np.log10(pa)))
    mant = pa / 10.0**exp
    if abs(mant - 1.0) < 1e-6:
        return f"10^{{{exp}}}"
    return f"{mant:.3g}\\times10^{{{exp}}}"


def _joined_with_pgi(jt: dict, Pgi: np.ndarray) -> dict:
    return {**jt, "Pgi": np.asarray(Pgi, dtype=np.float32)}


def _oxidation_limits(
    panels: list[tuple[dict, np.ndarray, str]],
    pressure_pa: float,
) -> tuple[float, float]:
    vals: list[float] = []
    for joined, stoich, mode in panels:
        for n in IR_COUNTS:
            if mode == "ensemble":
                curve = ensemble_scalar_vs_T_at_pressure(
                    joined, n, stoich, pressure_pa,
                )
            else:
                curve = winner_scalar_vs_T_at_pressure(
                    joined, n, stoich, pressure_pa,
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


def _style_bar_axes(axes: np.ndarray) -> None:
    last_row = axes.shape[0] - 1
    for row in range(axes.shape[0]):
        for col in range(axes.shape[1]):
            ax = axes[row, col]
            ax.set_yscale("linear")
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))   # n_Ir rows, not 3.5, 4.0, …
            label = DATASETS[col // 2]
            ax.set_title(f"{label}\n{TITLES[(row, col % 2)]}", fontsize=TITLE_FONTSIZE)
            ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
            if col == 0:
                ax.set_ylabel(r"$n_{\mathrm{Ir}}$", fontsize=AXIS_LABEL_FONTSIZE)
            else:
                ax.set_ylabel("")
                ax.tick_params(labelleft=False)
            if row == last_row:
                ax.set_xlabel("T (K)", fontsize=AXIS_LABEL_FONTSIZE)
            else:
                ax.set_xlabel("")
                ax.tick_params(labelbottom=False)


def _dataset_joined(label: str) -> tuple[dict, dict, dict, np.ndarray]:
    """(reference, full, novib) joined tables and n_O/n_Ir per GC row of one functional."""
    inp = dataset_inputs(label)
    gc_ref = load_gc_comparison_view(label, light=True)
    vd = load_variants_like(inp.variants_npz, gc_ref, ("Ts", "Ps", "stoich", "full_Pgi", "novib_Pgi"))
    rows = load_ace_features(inp.ace_features)
    jt = build_comparison_joined_table(
        gc_ref,
        rows,
        label,
    )
    n_ir = np.asarray(jt["n_Ir"], dtype=float)
    n_o = np.asarray(jt["n_O"], dtype=float)
    stoich = np.where(n_ir > 0, n_o / n_ir, np.nan)
    return jt, _joined_with_pgi(jt, vd["full_Pgi"]), _joined_with_pgi(jt, vd["novib_Pgi"]), stoich


def main() -> None:
    ensure_for_supplementary(["supp_vib_vs_novib_bars"])
    ensure_dirs(OUTPUT_DIR)

    pressure_pa = AMBIENT_PO2_PA
    panel_specs: dict[tuple[int, int], tuple[dict, np.ndarray, str]] = {}
    ts_ref = None
    for i, label in enumerate(DATASETS):
        jt, jt_full, jt_novib, stoich = _dataset_joined(label)
        if ts_ref is None:
            ts_ref = np.asarray(jt["Ts"], dtype=float)
        panel_specs[(0, 2 * i)] = (jt_full, stoich, "ensemble")
        panel_specs[(0, 2 * i + 1)] = (jt_novib, stoich, "ensemble")
        panel_specs[(1, 2 * i)] = (jt_full, stoich, "winner")
        panel_specs[(1, 2 * i + 1)] = (jt_novib, stoich, "winner")
    ox_vmin, ox_vmax = _oxidation_limits(list(panel_specs.values()), pressure_pa)

    fig, axes = plt.subplots(
        2, 2 * len(DATASETS),
        figsize=(FIG_W_MM / 25.4, FIG_H_MM / 25.4),
        dpi=DPI,
        sharex=True,
        sharey=True,
    )
    for (row, col), (joined, stoich, mode) in panel_specs.items():
        ax = axes[row, col]
        draw_kw = dict(
            cmap=PHASE_CMAP,
            vmin=ox_vmin,
            vmax=ox_vmax,
            section_kind="global_min",
        )
        if mode == "ensemble":
            draw_scalar_gradient_bars(
                ax, joined, pressure_pa, IR_COUNTS, stoich, **draw_kw,
            )
        else:
            draw_winner_scalar_gradient_bars(
                ax, joined, pressure_pa, IR_COUNTS, stoich, **draw_kw,
            )

    _style_bar_axes(axes)
    sync_temperature_xaxis_grid(axes, ts_ref)

    fig.subplots_adjust(
        left=0.08,
        right=LAYOUT_RIGHT,
        top=0.84,
        bottom=0.12,
        hspace=0.55,
        wspace=WSPACE,
    )
    fig.suptitle(
        rf"$p_{{\mathrm{{O_2}}}}={_format_po2(pressure_pa)}$ Pa",
        fontsize=TITLE_FONTSIZE,
        y=0.995,
    )

    mesh = ScalarMappable(
        norm=Normalize(vmin=ox_vmin, vmax=ox_vmax),
        cmap=PHASE_CMAP,
    )
    mesh.set_array([])
    add_phase_colorbar(
        fig, mesh,
        cax=margin_colorbar_cax(
            fig, axes[0, 0], axes[1, -1], gap=CBAR_GAP, width=CBAR_WIDTH,
        ),
        label=r"$n_O / n_{Ir}$",
    )
    save_fig(fig, OUTPUT_PATH, dpi=DPI)
    print(f"\n[supp] {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
