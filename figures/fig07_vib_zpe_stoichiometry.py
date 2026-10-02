"""Figure 7 – Boltzmann ΔZPE vs N_vib + reaction ⟨ΔG_vib⟩(T) by stoichiometry, PBE | PBE0.

180 mm wide, 4 rows × 2 columns (PBE | PBE0, same y range per row):

  1. Boltzmann ⟨ΔZPE⟩ vs vibrational DOF ``N_vib`` (by Ir).
  2–4. Boltzmann reaction ⟨ΔG_vib⟩(T) for Ir₄, Ir₅, Ir₆
     (stoichiometry colour; GC reaction normalised to one cluster product).

Each functional uses its own grand-canonical ensemble (Ir4–Ir6; PBE0 without the structures that have
no spectrum) and its own Ir and O₂ references.

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
    THERMO_IR_COUNTS,
    DPI,
    FIG08_PRESSURE_PA,
    FIG08_VIB_G_TEMP_K,
    OUTPUT_DIR,
    PHASE_CMAP,
)
from figures.utils.cache import ensure_for_figures
from figures.utils.datasets import DATASETS, dataset_inputs, gc_structures
from figures.utils.io import ensure_dirs, load_gc_npz
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
FIG_W_MM = 180
ROW_H_MM = 42
FIG_H_MM = 4 * ROW_H_MM
IR_TARGETS = THERMO_IR_COUNTS   # was (3, 6, 9)
LAYOUT_RECT = [0, 0, 0.92, 1]
CBAR_GAP = 0.01
CBAR_WIDTH = 0.008


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


def _dataset_series(label: str) -> dict:
    """ΔZPE rows (by Ir) and reaction ⟨ΔG_vib⟩(T) curves (by stoichiometry) of one functional."""
    inp = dataset_inputs(label)
    gc = load_gc_npz(inp.gc_npz, light=True, extra_keys=("G_tensor",))
    structures = gc_structures(label)
    if len(structures) != gc["Pgi"].shape[0]:
        raise ValueError(f"{label} structure count {len(structures)} != GC Pgi rows {gc['Pgi'].shape[0]}")
    check_vib_availability(structures, label, required=True)
    grouped = boltzmann_weighted_vib_stoichiometry_by_ir_content(
        structures, gc, temperature_k=FIG08_VIB_G_TEMP_K, pressure_pa=FIG08_PRESSURE_PA,
    )
    Ts, _p_used, curves_by_ir = boltzmann_reaction_vib_curves_by_stoichiometry(
        structures, gc, pressure_pa=FIG08_PRESSURE_PA,
        config=inp.grid_config(include_decomposition=True), o2_path=inp.o2_ref,
    )
    return {"grouped": grouped, "Ts": Ts, "curves": curves_by_ir}


def _finite_range(values: list[float]) -> tuple[float, float] | None:
    vals = [v for v in values if np.isfinite(v)]
    if not vals:
        return None
    lo, hi = float(min(vals)), float(max(vals))
    pad = 0.04 * (hi - lo if hi > lo else 1.0)
    return lo - pad, hi + pad


def main() -> None:
    ensure_for_figures([7])
    ensure_dirs(OUTPUT_DIR)

    series = {label: _dataset_series(label) for label in DATASETS}

    all_curves = {n: [r for d in series.values() for r in d["curves"].get(n, [])] for n in IR_TARGETS}
    ox_vmin, ox_vmax = _stoich_limits(all_curves, IR_TARGETS)
    ox_norm = Normalize(vmin=ox_vmin, vmax=ox_vmax)
    ox_cmap = plt.get_cmap(PHASE_CMAP)

    fig, axes = plt.subplots(
        4, len(DATASETS),
        figsize=(FIG_W_MM / 25.4, FIG_H_MM / 25.4),
        dpi=DPI,
        sharex=False,
        sharey=False,
    )

    for col, label in enumerate(DATASETS):
        d = series[label]
        zpe_xlim = _plot_zpe_vs_vib_dof(
            axes[0, col],
            d["grouped"],
            title=rf"{label}: Boltzmann $\langle\Delta\mathrm{{ZPE}}\rangle$ vs "
                  r"$N_{\mathrm{vib}}$ (by Ir)",
        )
        if zpe_xlim is not None:
            _pin_line_xlim(axes[0, col], *zpe_xlim)
        for row, n_ir in enumerate(IR_TARGETS, start=1):
            _plot_reaction_vib_vs_t(
                axes[row, col],
                d["Ts"],
                d["curves"].get(n_ir, []),
                ox_norm,
                ox_cmap,
                title=rf"{label}: Ir$_{{{n_ir}}}$",
            )
            axes[row, col].set_xlim(float(d["Ts"][0]), float(d["Ts"][-1]))
            axes[row, col].set_xmargin(0)

    # Same y range in both columns of a row, so the functionals compare directly.
    zpe_y = _finite_range([float(r["zpe_diff"]) for d in series.values() for rows in d["grouped"].values()
                           for r in rows] + [0.0])
    if zpe_y is not None:
        for ax in axes[0]:
            ax.set_ylim(*zpe_y)
    g_y = _finite_range([float(y) for curves in all_curves.values() for r in curves
                         for y in np.asarray(r["dg_vib"], dtype=float)])
    if g_y is not None:
        for ax in axes[1:].ravel():
            ax.set_ylim(*g_y)

    for ax in axes.ravel():
        ax.tick_params(axis="x", labelbottom=True)
        ax.tick_params(axis="y", labelleft=True)

    fig.tight_layout(pad=6 / 25.4, rect=LAYOUT_RECT)
    fig.subplots_adjust(hspace=0.62, wspace=0.22)

    add_phase_colorbar(
        fig,
        ScalarMappable(norm=ox_norm, cmap=ox_cmap),
        cax=margin_colorbar_cax(fig, axes[1, -1], axes[-1, -1], gap=CBAR_GAP, width=CBAR_WIDTH),
        label=r"$n_O/n_{\mathrm{Ir}}$",
    )

    save_fig(fig, OUTPUT_PATH, dpi=DPI, bbox_inches=None, pad_inches=0)
    print(f"\n[fig07] {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
