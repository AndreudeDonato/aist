"""Figure 9 – Boltzmann-weighted vib/ZPE stoichiometry + per-Ir ΔG terms (PBE).

Single column (88 mm wide, 3 rows):

  1. Boltzmann ⟨ΔZPE⟩ vs vibrational DOF ``N_vib`` (by Ir).
  2. Boltzmann ⟨Δ(G_vib/n_Ir)⟩ vs ``N_vib`` (by Ir).
  3. Vibrational ΔG contribution vs ``T`` at fixed ``pO₂``, one line per Ir.

Output: ``figures/outputs/fig09_vib_zpe_stoichiometry.svg``
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
    FIG08_PRESSURE_PA,
    FIG08_VIB_G_TEMP_K,
    OUTPUT_DIR,
    SUPP_IR_COUNTS,
    VIB_GC_NPZ,
)
from figures.utils.cache import ensure_for_figures
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

OUTPUT_PATH = os.path.join(OUTPUT_DIR, "fig09_vib_zpe_stoichiometry.svg")
FIG_W_MM = 88
FIG_H_MM = 135
VIB_TERM_LABEL = "Vibrational"
DOF_XLABEL = r"Vibrational DOF ($N_{\mathrm{vib}}$)"


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
        rf"Vibrational $\Delta G$ by Ir$_n$ at $p_{{\mathrm{{O_2}}}}={pressure_pa:.3g}$ Pa",
        fontsize=TITLE_FONTSIZE,
    )
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    ax.legend(fontsize=LEGEND_FONTSIZE, frameon=False, ncol=3, loc="best")
    return float(Ts[0]), float(Ts[-1])


def _load_grouped() -> dict[int, list[dict[str, Any]]]:
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
        temperature_k=FIG08_VIB_G_TEMP_K,
        pressure_pa=FIG08_PRESSURE_PA,
    )


def _load_contrib_by_ir() -> tuple[np.ndarray, float, dict[int, list[tuple[str, np.ndarray]]]]:
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
        FIG08_PRESSURE_PA,
        ir_counts=SUPP_IR_COUNTS,
        formulas=formulas,
    )


def main() -> None:
    ensure_for_figures([9])
    ensure_dirs(OUTPUT_DIR)

    structures = load_pbe_gc_structures(include_gas=False)
    check_vib_availability(structures, "PBE", required=False)

    grouped = _load_grouped()
    Ts, p_used, ir_lines = _load_contrib_by_ir()
    cond = (
        rf"$T={FIG08_VIB_G_TEMP_K:.0f}$ K, "
        rf"$p_{{\mathrm{{O_2}}}}={FIG08_PRESSURE_PA:.3g}$ Pa"
    )

    fig, axes = plt.subplots(
        3, 1,
        figsize=(FIG_W_MM / 25.4, FIG_H_MM / 25.4),
        dpi=DPI,
    )

    zpe_xlim = _plot_ir_series_vs_vib_dof(
        axes[0],
        grouped,
        y_key="zpe_diff",
        ylabel=r"$\langle\Delta\mathrm{ZPE}\rangle$ (eV)",
        title=rf"Boltzmann $\langle\Delta\mathrm{{ZPE}}\rangle$ vs "
              rf"$N_{{\mathrm{{vib}}}}$ (by Ir, {cond})",
    )
    gvib_dof_xlim = _plot_ir_series_vs_vib_dof(
        axes[1],
        grouped,
        y_key="g_vib_per_ir_diff",
        ylabel=rf"$\langle\Delta G_{{\mathrm{{vib}}}}/n_{{\mathrm{{Ir}}}}\rangle$ (eV)",
        title=rf"Boltzmann $\langle G_{{\mathrm{{vib}}}}/n_{{\mathrm{{Ir}}}}\rangle$ vs "
              rf"$N_{{\mathrm{{vib}}}}$ (by Ir, {cond})",
        show_zero_line=True,
    )
    t_xlim = _plot_vib_contrib_by_ir(axes[2], ir_lines, Ts, pressure_pa=p_used)

    axes[2].set_xlabel("T (K)", fontsize=AXIS_LABEL_FONTSIZE)
    axes[0].tick_params(labelbottom=False)
    axes[1].tick_params(labelbottom=False)

    fig.tight_layout(pad=6 / 25.4)
    if zpe_xlim is not None:
        _pin_line_xlim(axes[0], *zpe_xlim)
    if gvib_dof_xlim is not None:
        _pin_line_xlim(axes[1], *gvib_dof_xlim)
    if t_xlim is not None:
        _pin_line_xlim(axes[2], *t_xlim)

    save_fig(fig, OUTPUT_PATH, dpi=DPI)
    print(f"\n[fig09] {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
