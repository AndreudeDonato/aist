"""Supplementary – Boltzmann G_vib vs N_vib and per-Ir ΔG_vib(T) panels (PBE).

Former Figure 9 rows 2–5. Single column (88 mm wide, 4 rows):

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
    SUPP_IR_COUNTS,
    VIB_GC_NPZ,
)
from figures.utils.cache import ensure_for_supplementary
from figures.utils.io import ensure_dirs, load_gc_npz, load_pbe_gc_structures
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
FIG_W_MM = 88
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


def _load_grouped(temperature_k: float) -> dict[int, list[dict[str, Any]]]:
    gc = load_gc_npz(VIB_GC_NPZ, light=True)
    structures = load_pbe_gc_structures(include_gas=False)
    if len(structures) != gc["Pgi"].shape[0]:
        raise ValueError(
            f"PBE structure count {len(structures)} != GC Pgi rows {gc['Pgi'].shape[0]}"
        )
    return boltzmann_weighted_vib_stoichiometry_by_ir_content(
        structures,
        gc["Pgi"],
        gc["Ts"],
        gc["Ps"],
        temperature_k=temperature_k,
        pressure_pa=FIG08_PRESSURE_PA,
    )


def _load_contrib_by_ir(
    pressure_pa: float,
) -> tuple[np.ndarray, float, dict[int, list[tuple[str, np.ndarray]]]]:
    gc = load_gc_npz(VIB_GC_NPZ, light=True)
    structures = load_pbe_gc_structures(include_gas=False)
    n_ir = np.array([
        sum(1 for s in a.get_chemical_symbols() if s == "Ir") for a in structures
    ], dtype=int)
    formulas = gc.get("formulas")
    return contribution_lines_by_ir_at_pressure(
        gc,
        structures,
        n_ir,
        pressure_pa,
        ir_counts=SUPP_IR_COUNTS,
        formulas=formulas,
    )


def main() -> None:
    ensure_for_supplementary(["supp_vib_dof_gvib_contrib"])
    ensure_dirs(OUTPUT_DIR)

    structures = load_pbe_gc_structures(include_gas=False)
    check_vib_availability(structures, "PBE", required=False)

    grouped_273 = _load_grouped(FIG08_VIB_G_TEMP_K)
    grouped_500 = _load_grouped(FIG09_HIGH_T_K)
    Ts, p_ambient, ir_lines_ambient = _load_contrib_by_ir(FIG08_PRESSURE_PA)
    _, p_low, ir_lines_low = _load_contrib_by_ir(FIG04_LOW_PRESSURE_PA)
    cond_273 = _condition_label(FIG08_VIB_G_TEMP_K)
    cond_500 = _condition_label(FIG09_HIGH_T_K)
    gvib_ylabel = rf"$\langle G_{{\mathrm{{vib}}}}\rangle$ (eV)"

    def _gvib_title(cond: str) -> str:
        return (
            rf"Boltzmann $\langle G_{{\mathrm{{vib}}}}\rangle$ vs "
            rf"$N_{{\mathrm{{vib}}}}$ (by Ir, {cond})"
        )

    fig, axes = plt.subplots(
        4, 1,
        figsize=(FIG_W_MM / 25.4, FIG_H_MM / 25.4),
        dpi=DPI,
        sharex=False,
    )

    gvib_273_xlim = _plot_ir_series_vs_vib_dof(
        axes[0],
        grouped_273,
        y_key="g_vib",
        ylabel=gvib_ylabel,
        title=_gvib_title(cond_273),
        zero_at_ref=False,
    )
    gvib_500_xlim = _plot_ir_series_vs_vib_dof(
        axes[1],
        grouped_500,
        y_key="g_vib",
        ylabel=gvib_ylabel,
        title=_gvib_title(cond_500),
        zero_at_ref=False,
    )
    t_xlim_ambient = _plot_vib_contrib_by_ir(
        axes[2], ir_lines_ambient, Ts, pressure_pa=p_ambient,
    )
    t_xlim_low = _plot_vib_contrib_by_ir(
        axes[3], ir_lines_low, Ts, pressure_pa=p_low,
    )

    for ax in axes:
        ax.tick_params(axis="x", labelbottom=True)
        ax.tick_params(axis="y", labelleft=True)

    if gvib_273_xlim is not None:
        _pin_line_xlim(axes[0], *gvib_273_xlim)
    if gvib_500_xlim is not None:
        _pin_line_xlim(axes[1], *gvib_500_xlim)
    if t_xlim_ambient is not None:
        _pin_line_xlim(axes[2], *t_xlim_ambient)
    if t_xlim_low is not None:
        _pin_line_xlim(axes[3], *t_xlim_low)

    fig.tight_layout(pad=6 / 25.4)
    fig.subplots_adjust(hspace=0.42)

    save_fig(fig, OUTPUT_PATH, dpi=DPI, bbox_inches=None, pad_inches=0)
    print(f"[supp] {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
