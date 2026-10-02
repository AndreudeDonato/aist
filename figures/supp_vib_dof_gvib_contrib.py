"""Supplementary – Boltzmann G_vib vs N_vib and per-Ir ΔG_vib(T) panels, PBE | PBE0.

Former Figure 9 rows 2–5. 180 mm wide, 4 rows × 2 columns (PBE | PBE0, same y range per row):

  1. Boltzmann ⟨G_vib⟩ vs ``N_vib`` (by Ir, 273 K).
  2. Boltzmann ⟨G_vib⟩ vs ``N_vib`` (by Ir, 500 K).
  3. Vibrational ΔG contribution vs ``T`` at ambient ``pO₂``.
  4. Same as row 3 at low ``pO₂`` (``FIG04_LOW_PRESSURE_PA``).

Output: ``figures/outputs/supp_vib_dof_gvib_contrib.svg``
"""
from __future__ import annotations

import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

from aist.core.vibrational import boltzmann_weighted_vib_stoichiometry_by_ir_content

from figures.config import (
    DPI,
    FIG04_LOW_PRESSURE_PA,
    FIG08_PRESSURE_PA,
    FIG08_VIB_G_TEMP_K,
    OUTPUT_DIR,
    THERMO_IR_COUNTS,
)
from figures.utils.cache import ensure_for_supplementary
from figures.utils.datasets import DATASETS, dataset_inputs, gc_structures
from figures.utils.io import ensure_dirs, load_gc_npz
from figures.utils.plotting import (
    AXIS_LABEL_FONTSIZE,
    LEGEND_FONTSIZE,
    TICK_FONTSIZE,
    TITLE_FONTSIZE,
    save_fig,
)
from figures.utils.validation import check_vib_availability
from figures.utils.vib_contrib_by_ir import contribution_lines_by_ir_at_pressure

OUTPUT_PATH = os.path.join(OUTPUT_DIR, "supp_vib_dof_gvib_contrib.svg")
FIG_W_MM = 180
ROW_H_MM = 42
FIG_H_MM = 4 * ROW_H_MM
VIB_TERM_LABEL = "Vibrational"
DOF_XLABEL = r"Vibrational DOF ($N_{\mathrm{vib}}$)"
FIG09_HIGH_T_K = 500.0


def _condition_label(temperature_k: float) -> str:
    return (
        rf"$T={temperature_k:.0f}$ K, "
        rf"$p_{{\mathrm{{O_2}}}}={FIG08_PRESSURE_PA:.3g}$ Pa"
    )


def _pin_line_xlim(ax, xmin: float, xmax: float) -> None:
    ax.set_xlim(xmin, xmax)
    ax.set_xmargin(0)


def _ir_cmap(grouped: dict[int, list]) -> tuple[list[int], Normalize, plt.Colormap]:
    ir_vals = sorted(grouped)
    norm = Normalize(vmin=float(min(ir_vals)), vmax=float(max(ir_vals)))
    return ir_vals, norm, plt.get_cmap("winter")


def _plot_ir_series_vs_vib_dof(
    ax,
    grouped: dict[int, list[dict[str, Any]]],
    *,
    y_key: str,
    ylabel: str,
    title: str,
    zero_at_ref: bool = True,
    show_zero_line: bool = False,
) -> tuple[float, float] | None:
    if not grouped:
        ax.text(
            0.5, 0.5, "No data", ha="center", va="center",
            fontsize=AXIS_LABEL_FONTSIZE, transform=ax.transAxes,
        )
        return None

    ir_vals, norm, cmap = _ir_cmap(grouped)
    x_all: list[float] = []
    for n_ir in ir_vals:
        rows = sorted(grouped[n_ir], key=lambda r: (r["n_vib_modes"], r["n_o"]))
        x = np.array([r["n_vib_modes"] for r in rows], dtype=float)
        y = np.array([r[y_key] for r in rows], dtype=float)
        if zero_at_ref:
            ref = next((r for r in rows if int(r["n_o"]) == 0), None)
            if ref is not None:
                x0 = float(ref["n_vib_modes"])
                if x.size == 0 or not np.isclose(x[0], x0):
                    x = np.concatenate(([x0], x))
                    y = np.concatenate(([0.0], y))
                else:
                    y[0] = 0.0
        finite = np.isfinite(x) & np.isfinite(y)
        if not np.any(finite):
            continue
        ax.plot(
            x[finite], y[finite],
            color=cmap(norm(float(n_ir))), linewidth=1.2, label=f"Ir{n_ir}",
        )
        x_all.extend(x[finite].tolist())

    ax.set_xlabel(DOF_XLABEL, fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title(title, fontsize=TITLE_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    ax.legend(fontsize=LEGEND_FONTSIZE, frameon=False, ncol=3, loc="best")
    if show_zero_line:
        ax.axhline(0.0, color="0.75", linewidth=0.6, zorder=0)
    if not x_all:
        return None
    return float(min(x_all)), float(max(x_all))


def _plot_vib_contrib_by_ir(
    ax,
    ir_lines: dict[int, list[tuple[str, np.ndarray]]],
    Ts: np.ndarray,
    *,
    pressure_pa: float,
) -> tuple[float, float] | None:
    if not ir_lines:
        ax.text(
            0.5, 0.5, "No ΔG decomposition data", ha="center", va="center",
            fontsize=AXIS_LABEL_FONTSIZE, transform=ax.transAxes,
        )
        return None

    ir_vals = sorted(ir_lines)
    norm = Normalize(vmin=float(min(ir_vals)), vmax=float(max(ir_vals)))
    cmap = plt.get_cmap("winter")

    for n_ir in ir_vals:
        vib_curve = next(
            ((term, y) for term, y in ir_lines[n_ir] if term == VIB_TERM_LABEL),
            None,
        )
        if vib_curve is None:
            continue
        _, y = vib_curve
        ax.plot(
            Ts, y,
            color=cmap(norm(float(n_ir))),
            linewidth=1.2,
            label=f"Ir{n_ir}",
        )

    ax.set_xlabel("T (K)", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(r"Vibrational $\Delta G$ contribution (eV)", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title(
        rf"Vibrational $\Delta G$ by Ir$_n$ at "
        rf"$p_{{\mathrm{{O_2}}}}={pressure_pa:.3g}$ Pa",
        fontsize=TITLE_FONTSIZE,
    )
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    ax.legend(fontsize=LEGEND_FONTSIZE, frameon=False, ncol=3, loc="best")
    return float(Ts[0]), float(Ts[-1])


def _load_grouped(label: str, temperature_k: float) -> dict[int, list[dict[str, Any]]]:
    gc = load_gc_npz(dataset_inputs(label).gc_npz, light=True, extra_keys=("G_tensor",))
    structures = gc_structures(label)
    if len(structures) != gc["Pgi"].shape[0]:
        raise ValueError(
            f"{label} structure count {len(structures)} != GC Pgi rows {gc['Pgi'].shape[0]}"
        )
    return boltzmann_weighted_vib_stoichiometry_by_ir_content(
        structures,
        gc,
        temperature_k=temperature_k,
        pressure_pa=FIG08_PRESSURE_PA,
    )


def _load_contrib_by_ir(
    label: str,
    pressure_pa: float,
) -> tuple[np.ndarray, float, dict[int, list[tuple[str, np.ndarray]]]]:
    inp = dataset_inputs(label)
    gc = load_gc_npz(inp.gc_npz, light=True)
    structures = gc_structures(label)
    n_ir = np.array([
        sum(1 for s in a.get_chemical_symbols() if s == "Ir") for a in structures
    ], dtype=int)
    formulas = gc.get("formulas")
    return contribution_lines_by_ir_at_pressure(
        gc,
        structures,
        n_ir,
        pressure_pa,
        ir_counts=THERMO_IR_COUNTS,
        formulas=formulas,
        config=inp.grid_config(include_decomposition=True),
        o2_path=inp.o2_ref,
    )


def _shared_ylim(axes_row) -> None:
    lo = min(ax.get_ylim()[0] for ax in axes_row)
    hi = max(ax.get_ylim()[1] for ax in axes_row)
    for ax in axes_row:
        ax.set_ylim(lo, hi)


def main() -> None:
    ensure_for_supplementary(["supp_vib_dof_gvib_contrib"])
    ensure_dirs(OUTPUT_DIR)

    cond_273 = _condition_label(FIG08_VIB_G_TEMP_K)
    cond_500 = _condition_label(FIG09_HIGH_T_K)
    gvib_ylabel = rf"$\langle G_{{\mathrm{{vib}}}}\rangle$ (eV)"

    def _gvib_title(label: str, cond: str) -> str:
        return (
            rf"{label}: $\langle G_{{\mathrm{{vib}}}}\rangle$ vs "
            rf"$N_{{\mathrm{{vib}}}}$ (by Ir, {cond})"
        )

    fig, axes = plt.subplots(
        4, len(DATASETS),
        figsize=(FIG_W_MM / 25.4, FIG_H_MM / 25.4),
        dpi=DPI,
        sharex=False,
    )

    for col, label in enumerate(DATASETS):
        check_vib_availability(gc_structures(label), label, required=True)
        grouped_273 = _load_grouped(label, FIG08_VIB_G_TEMP_K)
        grouped_500 = _load_grouped(label, FIG09_HIGH_T_K)
        Ts, p_ambient, ir_lines_ambient = _load_contrib_by_ir(label, FIG08_PRESSURE_PA)
        _, p_low, ir_lines_low = _load_contrib_by_ir(label, FIG04_LOW_PRESSURE_PA)

        xlims = [
            _plot_ir_series_vs_vib_dof(
                axes[0, col], grouped_273, y_key="g_vib", ylabel=gvib_ylabel,
                title=_gvib_title(label, cond_273), zero_at_ref=False,
            ),
            _plot_ir_series_vs_vib_dof(
                axes[1, col], grouped_500, y_key="g_vib", ylabel=gvib_ylabel,
                title=_gvib_title(label, cond_500), zero_at_ref=False,
            ),
            _plot_vib_contrib_by_ir(axes[2, col], ir_lines_ambient, Ts, pressure_pa=p_ambient),
            _plot_vib_contrib_by_ir(axes[3, col], ir_lines_low, Ts, pressure_pa=p_low),
        ]
        for row, xlim in enumerate(xlims):
            if xlim is not None:
                _pin_line_xlim(axes[row, col], *xlim)
        for row in (2, 3):
            axes[row, col].set_title(f"{label}: " + axes[row, col].get_title(), fontsize=TITLE_FONTSIZE)

    for row in range(axes.shape[0]):
        _shared_ylim(axes[row])
    for ax in axes.ravel():
        ax.tick_params(axis="x", labelbottom=True)
        ax.tick_params(axis="y", labelleft=True)

    fig.tight_layout(pad=6 / 25.4)
    fig.subplots_adjust(hspace=0.55, wspace=0.25)

    save_fig(fig, OUTPUT_PATH, dpi=DPI, bbox_inches=None, pad_inches=0)
    print(f"[supp] {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
