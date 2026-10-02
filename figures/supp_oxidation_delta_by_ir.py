"""Supplementary – Thermo-variant oxidation at fixed pO₂, one figure per functional (PBE, PBE0).

Former Figure 7. 180 mm wide, two columns:

  Left:  ``Δ(n_O/n_Ir) = variant − VZE`` vs ``T`` (Ir₄…Ir₆ winter lines).
  Right: Fig. 3 row-2 style gradient bars for Ir₄…Ir₆ at the chosen pO₂.

Thermo code (3 chars): V / Z / E = on; ``\\`` = off. The PBE and PBE0 figures share y-limits and the
oxidation colour scale. This module draws the ambient-pO₂ pair; ``supp_oxidation_delta_by_ir_low_po2``
draws the low-pO₂ pair with ``build``.

Outputs: figures/outputs/supp_oxidation_delta_by_ir_PBE.svg, …_PBE0.svg
"""
from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

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
    BAR_HEIGHT,
    bar_xlim,
    ensemble_scalar_vs_T_at_pressure,
    sync_temperature_xaxis,
    winner_scalar_vs_T_at_pressure,
)
from figures.utils.io import ensure_dirs, load_ace_features, load_npz
from figures.utils.plotting import (
    AXIS_LABEL_FONTSIZE,
    LEGEND_FONTSIZE,
    TICK_FONTSIZE,
    TITLE_FONTSIZE,
    add_phase_colorbar,
    margin_colorbar_cax,
    save_fig,
)

OUTPUT_STEM = "supp_oxidation_delta_by_ir"
FIG_W_MM = 180
DELTA_ROW_H_MM = 36
IR_COUNTS = list(THERMO_IR_COUNTS)
BAR_IR_COUNTS = list(THERMO_IR_COUNTS)
IR_CMAP = "winter"
LAYOUT_RIGHT = 0.90
CBAR_GAP = 0.008
CBAR_WIDTH = 0.012
WSPACE = 0.28
BAR_HSPACE = 0.22
PRESSURE_PA = AMBIENT_PO2_PA

# Left Δ-row order = right bar-band order (top → bottom).
# Code: V/Z/E = on; \ = off  →  VZE, \\E, V\E, \ZE, \\\
VARIANT_ROWS = (
    ("full", "ensemble", "VZE"),
    ("novib", "ensemble", "\\\\E"),
    ("vib_nozpe", "ensemble", "V\\E"),
    ("zpe_only", "ensemble", "\\ZE"),
    ("novib", "winner", "\\\\\\"),
)

N_ENSEMBLE_DELTA = 4
N_DELTA = len(VARIANT_ROWS)
N_BAR = len(BAR_IR_COUNTS)
FIG_H_MM = N_DELTA * DELTA_ROW_H_MM


def _format_po2(pa: float) -> str:
    if pa >= 1.0:
        return f"{pa:.3g}"
    exp = int(np.floor(np.log10(pa)))
    mant = pa / 10.0**exp
    if abs(mant - 1.0) < 1e-6:
        return f"10^{{{exp}}}"
    return f"{mant:.3g}\\times10^{{{exp}}}"


def _ir_colormap():
    norm = Normalize(vmin=float(min(IR_COUNTS)), vmax=float(max(IR_COUNTS)))
    return norm, plt.get_cmap(IR_CMAP)


def _joined_with_pgi(jt: dict, Pgi: np.ndarray) -> dict:
    return {**jt, "Pgi": np.asarray(Pgi, dtype=np.float32)}


def _ox_lines(
    joined: dict,
    stoich: np.ndarray,
    pressure_pa: float,
    *,
    mode: str,
    ir_counts: list[int],
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    Ts = np.asarray(joined["Ts"], dtype=float)
    lines: dict[int, np.ndarray] = {}
    for n in ir_counts:
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
        lines[int(n)] = np.asarray(curve[2], dtype=float)
    return Ts, lines


def _delta_lines(
    variant: dict[int, np.ndarray],
    baseline: dict[int, np.ndarray],
) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    for n in IR_COUNTS:
        if n not in variant or n not in baseline:
            continue
        out[n] = np.asarray(variant[n], dtype=float) - np.asarray(baseline[n], dtype=float)
    return out


def _y_limits(*line_dicts: dict[int, np.ndarray]) -> tuple[float, float]:
    vals: list[float] = []
    for lines in line_dicts:
        for y in lines.values():
            finite = np.asarray(y, dtype=float)
            finite = finite[np.isfinite(finite)]
            if finite.size:
                vals.extend(finite.tolist())
    if not vals:
        return -0.1, 0.1
    vmin, vmax = float(min(vals)), float(max(vals))
    pad = 0.05 * (vmax - vmin) if vmax > vmin else 0.05
    return vmin - pad, vmax + pad


def _plot_delta(
    ax,
    Ts: np.ndarray,
    lines: dict[int, np.ndarray],
    norm,
    cmap,
    *,
    code: str,
    show_xlabel: bool,
    show_legend_labels: bool,
) -> None:
    ax.axhline(0.0, color="black", linewidth=0.6, linestyle="dotted", zorder=0)
    for n_ir in IR_COUNTS:
        y = lines.get(n_ir)
        if y is None:
            continue
        ax.plot(
            Ts, y,
            color=cmap(norm(float(n_ir))),
            linewidth=1.2,
            label=rf"Ir$_{{{n_ir}}}$" if show_legend_labels else None,
            zorder=2,
        )
    ax.set_ylabel(
        rf"$\Delta(n_O/n_{{\mathrm{{Ir}}}})$" + f"\n{code} − VZE",
        fontsize=AXIS_LABEL_FONTSIZE,
    )
    ax.set_xlabel("")  # set via sync_temperature_xaxis on the bottom panel
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE, labelleft=True)
    ax.yaxis.set_tick_params(labelleft=True)
    ax.margins(x=0)


def _draw_variant_gradient_bars(
    ax,
    Ts: np.ndarray,
    bands: list[tuple[str, np.ndarray]],
    *,
    cmap: str,
    vmin: float,
    vmax: float,
    n_ir: int,
) -> None:
    """Fig. 3 row-2 style: one coloured band per thermo code (top → bottom)."""
    norm = Normalize(vmin=vmin, vmax=vmax)
    cm = plt.get_cmap(cmap)
    ts = np.asarray(Ts, dtype=float)
    dT = float(ts[1] - ts[0]) if ts.size > 1 else 1.0
    n_bands = len(bands)
    # Integer y like fig3 n_Ir ticks; first code at the top.
    y_vals = list(range(n_bands))

    for y, (_code, values) in zip(y_vals, bands):
        colors = cm(norm(np.asarray(values, dtype=float)))
        bars = ax.barh(
            np.full_like(ts, float(y), dtype=float),
            width=dT,
            left=ts - 0.5 * dT,
            height=BAR_HEIGHT,
            color=colors,
            edgecolor="none",
            linewidth=0.0,
            antialiased=False,
            align="center",
        )
        for patch in bars:
            patch.set_rasterized(True)

    ax.set_yscale("linear")
    ax.set_yticks(y_vals)
    ax.set_yticklabels([code for code, _ in bands], fontsize=TICK_FONTSIZE)
    # Same pad convention as style_ir_size_bar_yaxis (top → bottom).
    pad = 0.5
    ax.set_ylim(n_bands - 1 + pad, -pad)
    xlo, xhi = bar_xlim(ts)
    ax.set_xlim(xlo, xhi)
    ax.set_ylabel(rf"Ir$_{{{n_ir}}}$", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_xlabel("")  # set via sync_temperature_xaxis on the bottom panel
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)


def _series(label: str, pressure_pa: float) -> dict:
    """Oxidation lines of every thermo variant and their Δ to VZE, for one functional."""
    inp = dataset_inputs(label)
    vd = load_npz(
        inp.variants_npz,
        keys=(
            "Ts", "Ps", "stoich",
            "full_Pgi", "novib_Pgi", "vib_nozpe_Pgi", "zpe_only_Pgi",
        ),
    )
    gc_ref = load_gc_comparison_view(label, light=True)
    rows = load_ace_features(inp.ace_features)
    n_gc = min(
        len(gc_ref["formulas"]),
        vd["full_Pgi"].shape[0],
        vd["novib_Pgi"].shape[0],
        vd["vib_nozpe_Pgi"].shape[0],
        vd["zpe_only_Pgi"].shape[0],
    )
    jt = build_comparison_joined_table(
        {**gc_ref, "Pgi": gc_ref["Pgi"][:n_gc]},
        rows,
        label,
    )
    n_ir = np.asarray(jt["n_Ir"], dtype=float)
    n_o = np.asarray(jt["n_O"], dtype=float)
    stoich = np.where(n_ir > 0, n_o / n_ir, np.nan)

    joined_by_branch = {
        "full": _joined_with_pgi(jt, vd["full_Pgi"][:n_gc]),
        "novib": _joined_with_pgi(jt, vd["novib_Pgi"][:n_gc]),
        "vib_nozpe": _joined_with_pgi(jt, vd["vib_nozpe_Pgi"][:n_gc]),
        "zpe_only": _joined_with_pgi(jt, vd["zpe_only_Pgi"][:n_gc]),
    }

    ox_cache: dict[tuple[str, str], dict[int, np.ndarray]] = {}
    Ts = None
    for branch, mode, _code in VARIANT_ROWS:
        key = (branch, mode)
        if key in ox_cache:
            continue
        Ts, lines = _ox_lines(
            joined_by_branch[branch], stoich, pressure_pa,
            mode=mode, ir_counts=IR_COUNTS,
        )
        ox_cache[key] = lines
    assert Ts is not None

    baseline = ox_cache[("full", "ensemble")]
    row_deltas = [
        _delta_lines(ox_cache[(branch, mode)], baseline)
        for branch, mode, _code in VARIANT_ROWS
    ]
    return {"Ts": Ts, "ox_cache": ox_cache, "row_deltas": row_deltas, "stoich": stoich}


def _ox_range(series: list[dict]) -> tuple[float, float]:
    """Same oxidation colour scale as Fig. 3 / other n_O/n_Ir panels: full stoich range."""
    vals = np.concatenate([d["stoich"][np.isfinite(d["stoich"])] for d in series])
    if vals.size:
        vmin, vmax = float(vals.min()), float(vals.max())
    else:
        vmin, vmax = 0.0, 1.0
    if vmax <= vmin:
        vmax = vmin + 1.0
    return vmin, vmax


def _draw(label: str, d: dict, *, pressure_pa: float, ylim_ens, ylim_min, ox_vmin: float, ox_vmax: float,
          output_path: str) -> None:
    Ts, ox_cache, row_deltas = d["Ts"], d["ox_cache"], d["row_deltas"]
    norm, cmap = _ir_colormap()
    last = N_DELTA - 1

    fig = plt.figure(figsize=(FIG_W_MM / 25.4, FIG_H_MM / 25.4), dpi=DPI)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=WSPACE)
    gs_left = gs[0, 0].subgridspec(
        N_DELTA, 1,
        height_ratios=[DELTA_ROW_H_MM] * N_DELTA,
        hspace=0.45,
    )
    # Squeeze all Ir-size panels into the same vertical span as the left column.
    gs_right = gs[0, 1].subgridspec(
        N_BAR, 1,
        hspace=BAR_HSPACE,
    )

    delta_axes = []
    for row, ((_branch, _mode, code), deltas) in enumerate(zip(VARIANT_ROWS, row_deltas)):
        ax = fig.add_subplot(gs_left[row, 0], sharex=delta_axes[0] if delta_axes else None)
        delta_axes.append(ax)
        _plot_delta(
            ax, Ts, deltas, norm, cmap,
            code=code,
            show_xlabel=(row == last),
            show_legend_labels=(row == 0),
        )
        ax.set_ylim(ylim_ens if row < N_ENSEMBLE_DELTA else ylim_min)
        ax.tick_params(axis="y", labelleft=True)
        # Temperature tick values on every left panel; "T (K)" only on the last.
        sync_temperature_xaxis(
            ax, Ts,
            xlabel="T (K)" if row == last else None,
        )
        ax.tick_params(axis="x", labelbottom=True)

    delta_axes[0].legend(
        fontsize=LEGEND_FONTSIZE, frameon=False, ncol=2, loc="best",
    )

    bar_axes = []
    for i, n in enumerate(BAR_IR_COUNTS):
        ax = fig.add_subplot(
            gs_right[i, 0],
            sharex=bar_axes[0] if bar_axes else None,
        )
        bar_axes.append(ax)
        bands = [
            (code, ox_cache[(branch, mode)][n])
            for branch, mode, code in VARIANT_ROWS
            if n in ox_cache[(branch, mode)]
        ]
        _draw_variant_gradient_bars(
            ax, Ts, bands,
            cmap=PHASE_CMAP, vmin=ox_vmin, vmax=ox_vmax,
            n_ir=n,
        )
        # Temperature tick values on every panel; "T (K)" only on the last.
        sync_temperature_xaxis(
            ax, Ts,
            xlabel="T (K)" if i == N_BAR - 1 else None,
        )
        ax.tick_params(axis="x", labelbottom=True)

    mesh = ScalarMappable(
        norm=Normalize(vmin=ox_vmin, vmax=ox_vmax),
        cmap=PHASE_CMAP,
    )
    mesh.set_array([])
    add_phase_colorbar(
        fig, mesh,
        cax=margin_colorbar_cax(
            fig, bar_axes[0], bar_axes[-1], gap=CBAR_GAP, width=CBAR_WIDTH,
        ),
        label=r"$n_O/n_{Ir}$",
    )

    fig.suptitle(
        rf"{label}: $p_{{\mathrm{{O_2}}}}={_format_po2(pressure_pa)}$ Pa"
        r"  (V/Z/E = on; \ = off)",
        fontsize=TITLE_FONTSIZE,
    )
    fig.subplots_adjust(
        left=0.12, right=LAYOUT_RIGHT, top=0.90, bottom=0.10,
    )
    save_fig(fig, output_path, dpi=DPI)
    print(f"\n[supp] {output_path}")


def build(pressure_pa: float, output_stem: str) -> None:
    """PBE and PBE0 figures at ``pressure_pa``: ``<output_stem>_PBE.svg`` and ``<output_stem>_PBE0.svg``."""
    ensure_dirs(OUTPUT_DIR)
    series = {label: _series(label, pressure_pa) for label in DATASETS}
    ylim_ens = _y_limits(*[r for d in series.values() for r in d["row_deltas"][:N_ENSEMBLE_DELTA]])
    ylim_min = _y_limits(*[d["row_deltas"][N_ENSEMBLE_DELTA] for d in series.values()])
    ox_vmin, ox_vmax = _ox_range(list(series.values()))
    for label, d in series.items():
        _draw(label, d, pressure_pa=pressure_pa, ylim_ens=ylim_ens, ylim_min=ylim_min,
              ox_vmin=ox_vmin, ox_vmax=ox_vmax, output_path=os.path.join(OUTPUT_DIR, f"{output_stem}_{label}.svg"))
    print("  codes:", [code for _b, _m, code in VARIANT_ROWS])


def main() -> None:
    ensure_for_supplementary(["supp_oxidation_delta_by_ir"])
    build(PRESSURE_PA, OUTPUT_STEM)


if __name__ == "__main__":
    main()
