"""Figure 7 – Boltzmann ΔZPE vs N_vib + reaction ⟨ΔG_vib⟩(T) by stoichiometry (PBE).

Single column (88 mm wide, 4 rows):

  1. Boltzmann ⟨ΔZPE⟩ vs vibrational DOF ``N_vib`` (by Ir).
  2–4. Boltzmann reaction ⟨ΔG_vib⟩(T) for Ir₃, Ir₆, Ir₉
     (stoichiometry colour; GC reaction normalised to one cluster product).

Output: ``figures/outputs/fig07_vib_zpe_stoichiometry.svg``
"""
from __future__ import annotations

import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from aist.core.vibrational import boltzmann_weighted_vib_stoichiometry_by_ir_content

from figures.config import (
    DPI,
    FIG08_PRESSURE_PA,
    FIG08_VIB_G_TEMP_K,
    OUTPUT_DIR,
    PHASE_CMAP,
    VIB_GC_NPZ,
)
from figures.utils.cache import ensure_for_figures
from figures.utils.io import ensure_dirs, load_gc_npz, load_pbe_gc_structures
from figures.utils.plotting import (
    AXIS_LABEL_FONTSIZE,
    LEGEND_FONTSIZE,
    TICK_FONTSIZE,
    TITLE_FONTSIZE,
    add_phase_colorbar,
    margin_colorbar_cax,
    save_fig,
)
from figures.utils.validation import check_vib_availability
from figures.utils.vib_contrib_by_ir import boltzmann_reaction_vib_curves_by_stoichiometry

OUTPUT_PATH = os.path.join(OUTPUT_DIR, "fig07_vib_zpe_stoichiometry.svg")
FIG_W_MM = 88
ROW_H_MM = 42
FIG_H_MM = 4 * ROW_H_MM
IR_TARGETS = (3, 6, 9)
LAYOUT_RECT = [0, 0, 0.84, 1]
CBAR_GAP = 0.01
CBAR_WIDTH = 0.012


def _pin_line_xlim(ax, xmin: float, xmax: float) -> None:
    ax.set_xlim(xmin, xmax)
    ax.set_xmargin(0)


def _ir_cmap(grouped: dict[int, list]) -> tuple[list[int], Normalize, plt.Colormap]:
    ir_vals = sorted(grouped)
    norm = Normalize(vmin=float(min(ir_vals)), vmax=float(max(ir_vals)))
    return ir_vals, norm, plt.get_cmap("winter")


def _plot_zpe_vs_vib_dof(
    ax,
    grouped: dict[int, list[dict[str, Any]]],
    *,
    title: str,
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
        y = np.array([r["zpe_diff"] for r in rows], dtype=float)
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

    ax.set_xlabel(r"Vibrational DOF ($N_{\mathrm{vib}}$)", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(r"$\langle\Delta\mathrm{ZPE}\rangle$ (eV)", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title(title, fontsize=TITLE_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE, labelbottom=True, labelleft=True)
    ax.legend(fontsize=LEGEND_FONTSIZE, frameon=False, ncol=3, loc="best")
    if not x_all:
        return None
    return float(min(x_all)), float(max(x_all))


def _stoich_limits(
    curves_by_ir: dict[int, list[dict]],
    ir_targets: tuple[int, ...],
) -> tuple[float, float]:
    vals: list[float] = []
    for n_ir in ir_targets:
        for row in curves_by_ir.get(n_ir, []):
            vals.append(float(row["stoich"]))
    if not vals:
        return 0.0, 1.0
    vmin, vmax = float(min(vals)), float(max(vals))
    if vmax <= vmin:
        vmax = vmin + 1.0
    return vmin, vmax


def _plot_reaction_vib_vs_t(
    ax,
    Ts: np.ndarray,
    rows: list[dict],
    norm: Normalize,
    cmap,
    *,
    title: str,
) -> None:
    if not rows:
        ax.text(
            0.5, 0.5, "No data", ha="center", va="center",
            fontsize=AXIS_LABEL_FONTSIZE, transform=ax.transAxes,
        )
        ax.set_title(title, fontsize=TITLE_FONTSIZE)
        return

    for row in rows:
        y = np.asarray(row["dg_vib"], dtype=float)
        finite = np.isfinite(y)
        if not np.any(finite):
            continue
        ax.plot(
            Ts[finite], y[finite],
            color=cmap(norm(float(row["stoich"]))),
            linewidth=1.2,
            zorder=2,
        )

    ax.set_title(title, fontsize=TITLE_FONTSIZE)
    ax.set_ylabel(
        r"$\langle \Delta G_{\mathrm{vib}}\rangle$ (eV)",
        fontsize=AXIS_LABEL_FONTSIZE,
    )
    ax.set_xlabel("T (K)", fontsize=AXIS_LABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE, labelbottom=True, labelleft=True)


def main() -> None:
    ensure_for_figures([7])
    ensure_dirs(OUTPUT_DIR)

    gc = load_gc_npz(VIB_GC_NPZ, light=True)
    structures = load_pbe_gc_structures(include_gas=False)
    if len(structures) != gc["Pgi"].shape[0]:
        raise ValueError(
            f"PBE structure count {len(structures)} != GC Pgi rows {gc['Pgi'].shape[0]}"
        )
    check_vib_availability(structures, "PBE", required=False)

    grouped_273 = boltzmann_weighted_vib_stoichiometry_by_ir_content(
        structures,
        gc["Pgi"],
        gc["Ts"],
        gc["Ps"],
        temperature_k=FIG08_VIB_G_TEMP_K,
        pressure_pa=FIG08_PRESSURE_PA,
    )
    Ts, _p_used, curves_by_ir = boltzmann_reaction_vib_curves_by_stoichiometry(
        structures,
        gc["Pgi"],
        gc["Ts"],
        gc["Ps"],
        pressure_pa=FIG08_PRESSURE_PA,
    )

    ox_vmin, ox_vmax = _stoich_limits(curves_by_ir, IR_TARGETS)
    ox_norm = Normalize(vmin=ox_vmin, vmax=ox_vmax)
    ox_cmap = plt.get_cmap(PHASE_CMAP)

    fig, axes = plt.subplots(
        4, 1,
        figsize=(FIG_W_MM / 25.4, FIG_H_MM / 25.4),
        dpi=DPI,
        sharex=False,
        sharey=False,
    )

    zpe_xlim = _plot_zpe_vs_vib_dof(
        axes[0],
        grouped_273,
        title=r"Boltzmann $\langle\Delta\mathrm{ZPE}\rangle$ vs "
              r"$N_{\mathrm{vib}}$ (by Ir)",
    )

    for row, n_ir in enumerate(IR_TARGETS, start=1):
        _plot_reaction_vib_vs_t(
            axes[row],
            Ts,
            curves_by_ir.get(n_ir, []),
            ox_norm,
            ox_cmap,
            title=rf"Ir$_{{{n_ir}}}$",
        )

    if zpe_xlim is not None:
        _pin_line_xlim(axes[0], *zpe_xlim)
    for ax in axes[1:]:
        ax.set_xlim(float(Ts[0]), float(Ts[-1]))
        ax.set_xmargin(0)

    y_vals = [
        float(y)
        for n_ir in IR_TARGETS
        for row in curves_by_ir.get(n_ir, [])
        for y in np.asarray(row["dg_vib"], dtype=float)
        if np.isfinite(y)
    ]
    if y_vals:
        ymin, ymax = float(min(y_vals)), float(max(y_vals))
        for ax in axes[1:]:
            ax.set_ylim(ymin, ymax)

    for ax in axes:
        ax.tick_params(axis="x", labelbottom=True)
        ax.tick_params(axis="y", labelleft=True)

    fig.tight_layout(pad=6 / 25.4, rect=LAYOUT_RECT)
    fig.subplots_adjust(hspace=0.42)

    add_phase_colorbar(
        fig,
        ScalarMappable(norm=ox_norm, cmap=ox_cmap),
        cax=margin_colorbar_cax(fig, axes[1], axes[-1], gap=CBAR_GAP, width=CBAR_WIDTH),
        label=r"$n_O/n_{\mathrm{Ir}}$",
    )

    save_fig(fig, OUTPUT_PATH, dpi=DPI, bbox_inches=None, pad_inches=0)
    print(f"\n[fig07] {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
