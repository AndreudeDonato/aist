"""Supplementary – Vibrational contributions and characteristic temperature maps, per functional.

Former Figure 8. One figure per functional (PBE, PBE0), each on its pooled Ir₄–Ir₆ grand-canonical
ensemble (PBE0 without the structures that have no spectrum).

2×2 layout:
  (0,0): ΔG contributions vs T at fixed pO₂ (1D cut from G_contributions_3x2)
  (0,1): Ensemble vibrational Gibbs contribution map <G_vib>
  (1,0): ZPE vs oxygen content
  (1,1): Ensemble energy-weighted cluster Θ_eff(T) map (no O₂)

Ensemble terms are per-structure contributions weighted with the final populations; the two figures
share their colour and y scales.

Outputs: ``figures/outputs/supp_vib_contrib_tchar_PBE.svg``, ``…_PBE0.svg``
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

from figures.config import (
    DPI,
    FIG08_PRESSURE_PA,
    OUTPUT_DIR,
)
from figures.utils.cache import ensure_for_supplementary
from figures.utils.datasets import DATASETS, dataset_inputs, gc_structures
from figures.utils.io import ensure_dirs, load_gc_npz
from figures.utils.plotting import (
    TITLE_FONTSIZE,
    AXIS_LABEL_FONTSIZE,
    TICK_FONTSIZE,
    LEGEND_FONTSIZE,
    CBAR_LABEL_FONTSIZE,
    draw_global_minimum_phase_overlay,
    draw_vib_reference_contours,
    save_fig,
    stoichiometry_overlay_labels,
    style_phase_ax,
)
from figures.utils.validation import check_vib_availability
from figures.utils.vib_contrib_by_ir import compute_per_structure_e_vib_tensor, contribution_lines_pooled_at_pressure
from figures.utils.vib_metrics import tchar_energy_weighted_ensemble_map

OUTPUT_STEM = "supp_vib_contrib_tchar"


def _pin_line_xlim(ax, xmin: float, xmax: float) -> None:
    """Tight x limits for line plots — no white margin at axis ends."""
    ax.set_xlim(xmin, xmax)
    ax.set_xmargin(0)


def _contributions_at_p_panel(ax, curves_data, ylim) -> tuple[float, float]:
    """Top-left: all ΔG contribution lines vs T at fixed pO₂."""
    Ts, p_used, curves = curves_data
    for label, y in curves:
        ax.plot(Ts, y, linewidth=1.2, label=label)
    ax.set_ylim(*ylim)
    ax.set_xlabel("T (K)", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(r"$\Delta G$ contribution (eV)", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title(rf"$\Delta G$ terms at $p_{{\mathrm{{O_2}}}}={p_used:.3g}$ Pa", fontsize=TITLE_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    ax.legend(fontsize=LEGEND_FONTSIZE, frameon=False, loc="best")
    return float(Ts[0]), float(Ts[-1])


def _gcontrib_map_panel(
    ax, gc, Z, title: str, *, vmax: float, tchar_map=None, cmap: str = "BrBG",
):
    Ts, Ps = gc["Ts"], gc["Ps"]
    norm = Normalize(vmin=-vmax, vmax=vmax)
    mesh = ax.pcolormesh(Ts, Ps, Z, shading="auto",
                         cmap=cmap, norm=norm, rasterized=True)
    overlay_labels = stoichiometry_overlay_labels(gc["formulas"])
    draw_global_minimum_phase_overlay(
        ax, Ts, Ps, gc["phase_index"], overlay_labels,
    )
    draw_vib_reference_contours(ax, Ts, Ps, tchar_map=tchar_map, gvib_map=Z)
    style_phase_ax(ax)
    ax.set_title(title, fontsize=TITLE_FONTSIZE)
    return mesh


def _zpe_panel(ax, structures, ylim) -> tuple[float, float] | None:
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

    ax.set_ylim(*ylim)
    ax.set_xlabel("Number of O atoms", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(r"$\Delta$ZPE vs Ir$_n$ (eV)", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title("Global-minimum ZPE vs oxygen (by Ir)", fontsize=TITLE_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    ax.legend(fontsize=LEGEND_FONTSIZE, frameon=False, ncol=3, loc="best")
    all_o = [r["n_o"] for n_ir in ir_vals for r in grouped[n_ir]]
    return float(min(all_o)), float(max(all_o))


def _tchar_panel(ax, gc, tc_map, gvib_map, *, tc_lim) -> None:
    Ts, Ps = gc["Ts"], gc["Ps"]
    if np.all(np.isnan(tc_map)):
        ax.text(0.5, 0.5, "No vib data — Θ_eff unavailable",
                ha="center", va="center", fontsize=AXIS_LABEL_FONTSIZE, transform=ax.transAxes)
        style_phase_ax(ax)
        return

    mesh = ax.pcolormesh(Ts, Ps, tc_map, shading="auto",
                         cmap="plasma", norm=Normalize(vmin=tc_lim[0], vmax=tc_lim[1]),
                         rasterized=True)
    draw_vib_reference_contours(ax, Ts, Ps, tchar_map=tc_map, gvib_map=gvib_map)
    style_phase_ax(ax)
    ax.set_title(r"Cluster $\langle\Theta_{\mathrm{eff}}(T)\rangle$", fontsize=TITLE_FONTSIZE)
    cbar = plt.colorbar(mesh, ax=ax, pad=0.02, fraction=0.046)
    cbar.set_label(r"$\langle\Theta_{\mathrm{eff}}\rangle$ (K)", fontsize=CBAR_LABEL_FONTSIZE)
    cbar.ax.tick_params(labelsize=TICK_FONTSIZE)


def _dataset(label: str) -> dict:
    """Ensemble ΔG terms, <G_vib> and Θ_eff maps of one functional (final populations)."""
    inp = dataset_inputs(label)
    gc = load_gc_npz(inp.gc_npz)
    structures = gc_structures(label)
    if len(structures) != gc["Pgi"].shape[0]:
        raise ValueError(
            f"{label} structure count {len(structures)} != GC Pgi rows {gc['Pgi'].shape[0]}"
        )
    check_vib_availability(structures, label, required=True)
    cfg = inp.grid_config(include_decomposition=True)
    pgi = np.asarray(gc["Pgi"], dtype=float)
    e_vib = compute_per_structure_e_vib_tensor(structures, gc["Ts"], gc["Ps"], config=cfg, o2_path=inp.o2_ref)
    with np.errstate(invalid="ignore", divide="ignore"):
        gvib_map = np.einsum("ijk,ijk->jk", pgi, e_vib) / pgi.sum(axis=0)
    from aist.core.vibrational import zpe_global_minima_by_ir_content

    zpe = [r["zpe_diff"] for rows in zpe_global_minima_by_ir_content(structures).values() for r in rows]
    return {
        "label": label,
        "gc": gc,
        "structures": structures,
        "curves": contribution_lines_pooled_at_pressure(
            gc, structures, FIG08_PRESSURE_PA, config=cfg, o2_path=inp.o2_ref,
        ),
        "gvib_map": gvib_map,
        "tc_map": tchar_energy_weighted_ensemble_map(structures, pgi, gc["Ts"]),
        "zpe": zpe,
    }


def _padded(lo: float, hi: float) -> tuple[float, float]:
    pad = 0.04 * (hi - lo if hi > lo else 1.0)
    return lo - pad, hi + pad


def build(d: dict, limits: dict) -> None:
    label, gc = d["label"], d["gc"]
    output_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_STEM}_{label}.svg")
    fig, axes = plt.subplots(2, 2, figsize=(180 / 25.4, 130 / 25.4), dpi=DPI)

    # (0,0): 1D ΔG contributions at fixed pO₂
    dg_xlim = _contributions_at_p_panel(axes[0, 0], d["curves"], limits["dg_ylim"])

    # (0,1): vibrational Gibbs map
    m2 = _gcontrib_map_panel(
        axes[0, 1], gc, d["gvib_map"],
        r"Vibrational $\langle G_{vib}\rangle$ (eV)",
        vmax=limits["gvib_vmax"], tchar_map=d["tc_map"],
    )
    cbar = plt.colorbar(m2, ax=axes[0, 1], pad=0.02, fraction=0.046)
    cbar.set_label("eV", fontsize=CBAR_LABEL_FONTSIZE)
    cbar.ax.tick_params(labelsize=TICK_FONTSIZE)

    # (1,0): ZPE
    zpe_xlim = _zpe_panel(axes[1, 0], d["structures"], limits["zpe_ylim"])

    # (1,1): Tchar
    _tchar_panel(axes[1, 1], gc, d["tc_map"], d["gvib_map"], tc_lim=limits["tc_lim"])

    fig.suptitle(label, fontsize=TITLE_FONTSIZE)
    fig.tight_layout(pad=6 / 25.4, rect=[0, 0, 1, 0.96])
    _pin_line_xlim(axes[0, 0], *dg_xlim)
    if zpe_xlim is not None:
        _pin_line_xlim(axes[1, 0], *zpe_xlim)
    save_fig(fig, output_path, dpi=DPI)
    print(f"[supp] {output_path}")


def main() -> None:
    ensure_for_supplementary(["supp_vib_contrib_tchar"])
    ensure_dirs(OUTPUT_DIR)
    data = [_dataset(label) for label in DATASETS]
    lines = [y for d in data for _, y in d["curves"][2]]
    limits = {
        "dg_ylim": _padded(min(float(np.nanmin(y)) for y in lines), max(float(np.nanmax(y)) for y in lines)),
        "gvib_vmax": max(float(np.nanpercentile(np.abs(d["gvib_map"]), 97)) for d in data),
        "tc_lim": (min(float(np.nanmin(d["tc_map"])) for d in data), max(float(np.nanmax(d["tc_map"])) for d in data)),
        "zpe_ylim": _padded(min(min(d["zpe"]) for d in data), max(max(d["zpe"]) for d in data)),
    }
    for d in data:
        build(d, limits)


if __name__ == "__main__":
    main()
