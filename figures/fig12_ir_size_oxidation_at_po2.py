"""Figure 12 – Per-Ir curves vs T at ambient pO₂ (PBE | PBE0 | Δ).

2×3 layout at ``AMBIENT_PO2_PA``:
  Row 1 — ensemble ``⟨n_O/n_Ir⟩`` vs T
  Row 2 — ensemble ``⟨HOMO–LUMO⟩`` vs T (same pressure)
  Columns: PBE, PBE0, PBE0 − PBE (dotted y = 0 behind curves).

Ir₃…Ir₉ curves use the ``winter`` colormap (as the ZPE panel in fig08).

Output: ``figures/outputs/fig12_ir_size_oxidation_at_po2.svg``
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from ase.io import read
from matplotlib.colors import Normalize

from figures.config import (
    ACE_FEATURES,
    AMBIENT_PO2_PA,
    DPI,
    GC_NPZ,
    GC_NPZ_PBE0,
    GC_VARIANTS_NPZ,
    OUTPUT_DIR,
    TRAJ_PBE,
    TRAJ_PBE0,
)
from figures.utils.cache import ensure_for_figures
from figures.utils.gc_subset import (
    homo_lumo_vs_t_by_ir_at_pressure,
    oxidation_vs_t_by_ir_at_pressure,
)
from figures.utils.homo_lumo import homo_lumo_gaps
from figures.utils.io import (
    ensure_dirs,
    load_ace_features,
    load_gc_npz,
    load_gc_pbe_novib_view,
    load_npz,
)
from figures.utils.plotting import (
    save_fig,
    TITLE_FONTSIZE,
    AXIS_LABEL_FONTSIZE,
    TICK_FONTSIZE,
    LEGEND_FONTSIZE,
)

OUTPUT_PATH = os.path.join(OUTPUT_DIR, "fig12_ir_size_oxidation_at_po2.svg")
FIG_W_MM = 180
ROW_H_MM = 60
FIG_H_MM = 2 * ROW_H_MM
IR_COUNTS = list(range(3, 10))
ZPE_CMAP = "winter"

ROW_SPECS = (
    ("oxidation", r"$n_O/n_{\mathrm{Ir}}$", r"$\Delta(n_O/n_{\mathrm{Ir}})$ (PBE0 − PBE)"),
    ("homo_lumo", "HOMO–LUMO gap (eV)", r"$\Delta(\mathrm{HOMO-LUMO})$ (PBE0 − PBE, eV)"),
)


def _load_dataset(label: str) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    traj = TRAJ_PBE if label == "PBE" else TRAJ_PBE0
    if label == "PBE":
        gc = load_gc_pbe_novib_view(GC_VARIANTS_NPZ, GC_NPZ)
        stoich = np.asarray(load_npz(GC_VARIANTS_NPZ, keys=("stoich",))["stoich"], dtype=float)
    else:
        gc = load_gc_npz(GC_NPZ_PBE0, light=True)
        stoich = np.asarray(gc["stoich"], dtype=float)
    rows = load_ace_features(ACE_FEATURES[label])
    n = gc["Pgi"].shape[0]
    n_ir = np.array([r["n_Ir"] for r in rows[:n]], dtype=int)
    gaps = homo_lumo_gaps(list(read(traj, ":"))[:n])
    return gc, stoich, n_ir, gaps


def _ir_colormap():
    norm = Normalize(vmin=float(min(IR_COUNTS)), vmax=float(max(IR_COUNTS)))
    cmap = plt.get_cmap(ZPE_CMAP)
    return norm, cmap


def _format_po2(pa: float) -> str:
    if pa >= 1.0:
        return f"{pa:.3g}"
    exp = int(np.floor(np.log10(pa)))
    mant = pa / 10.0**exp
    if abs(mant - 1.0) < 1e-6:
        return f"10^{{{exp}}}"
    return f"{mant:.3g}\\times10^{{{exp}}}"


def _plot_ir_lines(
    ax,
    Ts,
    lines: dict[int, np.ndarray],
    norm,
    cmap,
    *,
    title: str,
    ylabel: str,
    show_xlabel: bool,
    show_ylabel: bool,
) -> None:
    for n_ir in IR_COUNTS:
        y = lines.get(n_ir)
        if y is None:
            continue
        ax.plot(
            Ts, y,
            color=cmap(norm(float(n_ir))),
            linewidth=1.2,
            label=f"Ir{n_ir}",
            zorder=2,
        )
    ax.set_xlabel("T (K)" if show_xlabel else "", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(ylabel if show_ylabel else "", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title(title, fontsize=TITLE_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    ax.margins(x=0)


def _plot_diff_panel(
    ax,
    Ts,
    lines_pbe: dict[int, np.ndarray],
    lines_pbe0: dict[int, np.ndarray],
    norm,
    cmap,
    *,
    ylabel: str,
    show_xlabel: bool,
    show_ylabel: bool,
) -> None:
    ax.axhline(0.0, color="black", linewidth=0.6, linestyle="dotted", zorder=0)
    for n_ir in IR_COUNTS:
        if n_ir not in lines_pbe or n_ir not in lines_pbe0:
            continue
        diff = np.asarray(lines_pbe0[n_ir], dtype=float) - np.asarray(lines_pbe[n_ir], dtype=float)
        ax.plot(
            Ts, diff,
            color=cmap(norm(float(n_ir))),
            linewidth=1.2,
            label=f"Ir{n_ir}",
            zorder=2,
        )
    ax.set_xlabel("T (K)" if show_xlabel else "", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(ylabel if show_ylabel else "", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title("PBE0 − PBE", fontsize=TITLE_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    ax.margins(x=0)


def _y_limits(*line_dicts: dict[int, np.ndarray], include_diff: bool = False) -> tuple[float, float]:
    vals: list[float] = []
    for lines in line_dicts:
        for y in lines.values():
            finite = np.asarray(y, dtype=float)
            finite = finite[np.isfinite(finite)]
            if finite.size:
                vals.extend(finite.tolist())
    if include_diff and len(line_dicts) >= 2:
        common = set(line_dicts[0]) & set(line_dicts[1])
        for n in common:
            d = np.asarray(line_dicts[1][n], dtype=float) - np.asarray(line_dicts[0][n], dtype=float)
            finite = d[np.isfinite(d)]
            if finite.size:
                vals.extend(finite.tolist())
    if not vals:
        return 0.0, 1.0
    vmin, vmax = float(min(vals)), float(max(vals))
    pad = 0.05 * (vmax - vmin) if vmax > vmin else 0.05
    return vmin - pad, vmax + pad


def main() -> None:
    ensure_for_figures([12])
    ensure_dirs(OUTPUT_DIR)

    gc_pbe, stoich_pbe, n_ir_pbe, gaps_pbe = _load_dataset("PBE")
    gc_pbe0, stoich_pbe0, n_ir_pbe0, gaps_pbe0 = _load_dataset("PBE0")

    Ts, p_used, lines_pbe_ox = oxidation_vs_t_by_ir_at_pressure(
        gc_pbe, stoich_pbe, n_ir_pbe, AMBIENT_PO2_PA, ir_counts=IR_COUNTS,
    )
    _, _, lines_pbe0_ox = oxidation_vs_t_by_ir_at_pressure(
        gc_pbe0, stoich_pbe0, n_ir_pbe0, AMBIENT_PO2_PA, ir_counts=IR_COUNTS,
    )
    _, _, lines_pbe_gap = homo_lumo_vs_t_by_ir_at_pressure(
        gc_pbe, gaps_pbe, n_ir_pbe, AMBIENT_PO2_PA, ir_counts=IR_COUNTS,
    )
    _, _, lines_pbe0_gap = homo_lumo_vs_t_by_ir_at_pressure(
        gc_pbe0, gaps_pbe0, n_ir_pbe0, AMBIENT_PO2_PA, ir_counts=IR_COUNTS,
    )
    del gc_pbe, gc_pbe0

    row_data = {
        "oxidation": (lines_pbe_ox, lines_pbe0_ox),
        "homo_lumo": (lines_pbe_gap, lines_pbe0_gap),
    }

    norm, cmap = _ir_colormap()
    po2_tex = _format_po2(p_used)

    fig, axes = plt.subplots(
        len(ROW_SPECS), 3,
        figsize=(FIG_W_MM / 25.4, FIG_H_MM / 25.4),
        dpi=DPI,
        sharex=True,
    )

    last_row = len(ROW_SPECS) - 1
    for row, (row_key, ylabel, diff_ylabel) in enumerate(ROW_SPECS):
        lines_pbe, lines_pbe0 = row_data[row_key]
        ylim = _y_limits(lines_pbe, lines_pbe0, include_diff=True)

        _plot_ir_lines(
            axes[row, 0], Ts, lines_pbe, norm, cmap,
            title=rf"PBE — $p_{{\mathrm{{O_2}}}}={po2_tex}$ Pa",
            ylabel=ylabel,
            show_xlabel=(row == last_row),
            show_ylabel=True,
        )
        _plot_ir_lines(
            axes[row, 1], Ts, lines_pbe0, norm, cmap,
            title=rf"PBE0 — $p_{{\mathrm{{O_2}}}}={po2_tex}$ Pa",
            ylabel=ylabel,
            show_xlabel=(row == last_row),
            show_ylabel=False,
        )
        _plot_diff_panel(
            axes[row, 2], Ts, lines_pbe, lines_pbe0, norm, cmap,
            ylabel=diff_ylabel,
            show_xlabel=(row == last_row),
            show_ylabel=False,
        )
        for ax in axes[row, :]:
            ax.set_ylim(ylim)

    axes[0, 0].legend(fontsize=LEGEND_FONTSIZE, frameon=False, ncol=2, loc="best")

    fig.tight_layout(pad=6 / 25.4)
    save_fig(fig, OUTPUT_PATH, dpi=DPI)
    print(f"\n[fig12] {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
