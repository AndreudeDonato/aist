"""Figure 8 – Vibrational contributions and characteristic temperature maps (PBE).

PBE structures carry ``atoms.info['frequencies']``; PBE0 vib data are not used.

2×2 layout:
  (0,0): ΔG contributions vs T at fixed pO₂ (1D cut from G_contributions_3x2)
  (0,1): Vibrational Gibbs contribution map (E_vib_avg)
  (1,0): ZPE vs oxygen content
  (1,1): Ensemble energy-weighted cluster Θ_eff(T) map (no O₂)

Output: ``figures/outputs/fig08_vib_contrib_tchar.svg``
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

from aist.core.grandcanonical import contribution_lines_at_pressure

from figures.config import (
    DPI,
    FIG08_PRESSURE_PA,
    OUTPUT_DIR,
    VIB_GC_NPZ,
    VIB_TRAJ,
)
from figures.utils.cache import ensure_for_figures
from figures.utils.io import ensure_dirs, load_gc_npz, load_traj
from figures.utils.plotting import (
    TITLE_FONTSIZE,
    AXIS_LABEL_FONTSIZE,
    TICK_FONTSIZE,
    LEGEND_FONTSIZE,
    CBAR_LABEL_FONTSIZE,
    draw_global_minimum_phase_overlay,
    draw_tchar_boundary,
    save_fig,
    stoichiometry_overlay_labels,
    style_phase_ax,
)
from figures.utils.validation import check_vib_availability
from figures.utils.vib_metrics import tchar_energy_weighted_ensemble_map

OUTPUT_PATH = os.path.join(OUTPUT_DIR, "fig08_vib_contrib_tchar.svg")


def _pin_line_xlim(ax, xmin: float, xmax: float) -> None:
    """Tight x limits for line plots — no white margin at axis ends."""
    ax.set_xlim(xmin, xmax)
    ax.set_xmargin(0)


def _contributions_at_p_panel(ax, gc, pressure_pa: float) -> tuple[float, float] | None:
    """Top-left: all ΔG contribution lines vs T at fixed pO₂."""
    try:
        Ts, p_used, curves = contribution_lines_at_pressure(gc, pressure_pa)
        for label, y in curves:
            ax.plot(Ts, y, linewidth=1.2, label=label)
        ax.set_xlabel("T (K)", fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_ylabel(r"$\Delta G$ contribution (eV)", fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_title(rf"$\Delta G$ terms at $p_{{\mathrm{{O_2}}}}={p_used:.3g}$ Pa", fontsize=TITLE_FONTSIZE)
        ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
        ax.legend(fontsize=LEGEND_FONTSIZE, frameon=False, loc="best")
        return float(Ts[0]), float(Ts[-1])
    except KeyError as exc:
        ax.text(
            0.5, 0.5, str(exc),
            ha="center", va="center", fontsize=AXIS_LABEL_FONTSIZE, transform=ax.transAxes, wrap=True,
        )
        return None


def _gcontrib_map_panel(ax, gc, key: str, title: str, cmap: str = "BrBG"):
    Ts, Ps = gc["Ts"], gc["Ps"]
    Z = gc[key].astype(float)
    vmax = float(np.nanpercentile(np.abs(Z), 97))
    norm = Normalize(vmin=-vmax, vmax=vmax)
    mesh = ax.pcolormesh(Ts, Ps, Z, shading="auto",
                         cmap=cmap, norm=norm, rasterized=True)
    overlay_labels = stoichiometry_overlay_labels(gc["formulas"])
    draw_global_minimum_phase_overlay(
        ax, Ts, Ps, gc["phase_index"], overlay_labels,
    )
    style_phase_ax(ax)
    ax.set_title(title, fontsize=TITLE_FONTSIZE)
    return mesh


def _zpe_panel(ax, structures) -> tuple[float, float] | None:
    from aist.core.vibrational import zpe_global_minima_by_ir_content

    grouped = zpe_global_minima_by_ir_content(structures)

    if not grouped:
        ax.text(0.5, 0.5, "No ZPE data", ha="center", va="center",
                fontsize=AXIS_LABEL_FONTSIZE, transform=ax.transAxes)
        return None

    ir_vals = sorted(grouped)
    norm = Normalize(vmin=float(min(ir_vals)), vmax=float(max(ir_vals)))
    cmap = plt.get_cmap("winter")

    for n_ir in ir_vals:
        rows = sorted(grouped[n_ir], key=lambda r: r["n_o"])
        x = np.array([r["n_o"] for r in rows], dtype=float)
        y = np.array([r["zpe_diff"] for r in rows], dtype=float)
        ax.plot(x, y, color=cmap(norm(float(n_ir))),
                linewidth=1.2, label=f"Ir{n_ir}")

    ax.set_xlabel("Number of O atoms", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(r"$\Delta$ZPE vs Ir$_n$ (eV)", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title("Global-minimum ZPE vs oxygen (by Ir)", fontsize=TITLE_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    ax.legend(fontsize=LEGEND_FONTSIZE, frameon=False, ncol=3, loc="best")
    all_o = [r["n_o"] for n_ir in ir_vals for r in grouped[n_ir]]
    return float(min(all_o)), float(max(all_o))


def _tchar_panel(ax, gc, structures) -> None:
    Ts, Ps = gc["Ts"], gc["Ps"]
    tc_map = tchar_energy_weighted_ensemble_map(structures, gc["Pgi"], Ts)

    if np.all(np.isnan(tc_map)):
        ax.text(0.5, 0.5, "No vib data — Θ_eff unavailable",
                ha="center", va="center", fontsize=AXIS_LABEL_FONTSIZE, transform=ax.transAxes)
        style_phase_ax(ax)
        return

    vmin = float(np.nanmin(tc_map))
    vmax = float(np.nanmax(tc_map))
    mesh = ax.pcolormesh(Ts, Ps, tc_map, shading="auto",
                         cmap="plasma", norm=Normalize(vmin=vmin, vmax=vmax),
                         rasterized=True)
    draw_tchar_boundary(ax, Ts, Ps, tc_map)
    style_phase_ax(ax)
    ax.set_title(r"Cluster $\langle\Theta_{\mathrm{eff}}(T)\rangle$", fontsize=TITLE_FONTSIZE)
    cbar = plt.colorbar(mesh, ax=ax, pad=0.02, fraction=0.046)
    cbar.set_label(r"$\langle\Theta_{\mathrm{eff}}\rangle$ (K)", fontsize=CBAR_LABEL_FONTSIZE)
    cbar.ax.tick_params(labelsize=TICK_FONTSIZE)


def main() -> None:
    ensure_for_figures([8])
    ensure_dirs(OUTPUT_DIR)

    gc = load_gc_npz(VIB_GC_NPZ)
    structures = list(load_traj(VIB_TRAJ))
    check_vib_availability(structures, "PBE", required=False)

    fig, axes = plt.subplots(2, 2, figsize=(180 / 25.4, 130 / 25.4), dpi=DPI)

    # (0,0): 1D ΔG contributions at fixed pO₂
    dg_xlim = _contributions_at_p_panel(axes[0, 0], gc, FIG08_PRESSURE_PA)

    # (0,1): vibrational Gibbs map
    if "E_vib_avg" in gc:
        m2 = _gcontrib_map_panel(
            axes[0, 1], gc, "E_vib_avg",
            r"Vibrational $\langle G_{vib}\rangle$ (eV)",
        )
        cbar = plt.colorbar(m2, ax=axes[0, 1], pad=0.02, fraction=0.046)
        cbar.set_label("eV", fontsize=CBAR_LABEL_FONTSIZE)
        cbar.ax.tick_params(labelsize=TICK_FONTSIZE)
    else:
        axes[0, 1].text(0.5, 0.5, "E_vib_avg not in NPZ",
                        ha="center", va="center", fontsize=AXIS_LABEL_FONTSIZE,
                        transform=axes[0, 1].transAxes)

    # (1,0): ZPE
    zpe_xlim = _zpe_panel(axes[1, 0], structures)

    # (1,1): Tchar
    _tchar_panel(axes[1, 1], gc, structures)

    fig.tight_layout(pad=6 / 25.4)
    if dg_xlim is not None:
        _pin_line_xlim(axes[0, 0], *dg_xlim)
    if zpe_xlim is not None:
        _pin_line_xlim(axes[1, 0], *zpe_xlim)
    save_fig(fig, OUTPUT_PATH, dpi=DPI)
    print(f"\n[fig08] {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
