"""Supplementary – G vs μ_O₂ per structure at fixed T (PBE | PBE0).

7×2 panels (Ir₃…Ir₉ × PBE | PBE0):
  one ``G_i`` line per structure vs ``μ_O₂``, coloured by that structure's
  ``n_O/n_Ir`` (novib / electronic level for PBE, as supp Ir-size oxidation maps).

Outputs:
  ``figures/outputs/supp_gc_g_vs_mu_273K.svg``
  ``figures/outputs/supp_gc_g_vs_mu_500K.svg``
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

from figures.config import (
    ACE_FEATURES,
    DPI,
    GC_NPZ,
    GC_NPZ_PBE0,
    GC_VARIANTS_NPZ,
    O2_REF_PBE,
    O2_REF_PBE0,
    OUTPUT_DIR,
    PHASE_CMAP,
)
from figures.utils.cache import ensure_for_supplementary
from figures.utils.gc_subset import structure_g_curves_by_ir_at_temperature
from figures.utils.gc_thermo import o2_chemical_potential_grid
from figures.utils.io import (
    ensure_dirs,
    load_ace_features,
    load_gc_pbe_novib_g_tensor,
    load_npz,
)
from figures.utils.plotting import save_fig, stoichiometry_overlay_labels

FIG_W_MM = 180
ROW_H_MM = 32
FIG_H_MM = 7 * ROW_H_MM
IR_COUNTS = list(range(3, 10))
TEMPERATURES_K = (273.0, 500.0)
LINE_WIDTH = 0.7
ENVELOPE_LINE_WIDTH = 0.35
LABEL_FONTSIZE = 7

Panel = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def _output_path(temperature_k: float) -> str:
    return os.path.join(OUTPUT_DIR, f"supp_gc_g_vs_mu_{int(temperature_k)}K.svg")


def _stoich_limits(stoich: np.ndarray, n_ir: np.ndarray) -> tuple[float, float]:
    vals: list[float] = []
    for n in IR_COUNTS:
        mask = n_ir == n
        if np.any(mask):
            vals.extend(np.asarray(stoich[mask], dtype=float).tolist())
    if not vals:
        return 0.0, 1.0
    vmin, vmax = float(min(vals)), float(max(vals))
    if vmax <= vmin:
        vmax = vmin + 1.0
    return vmin, vmax


def _mu_span(mu: np.ndarray) -> tuple[float, float]:
    mu = np.asarray(mu, dtype=float)
    return float(mu[0]), float(mu[-1])


def _phase_segment_bounds(phase_idx: np.ndarray) -> list[tuple[int, int]]:
    n = int(phase_idx.size)
    if n == 0:
        return []
    bounds = [0]
    for i in range(1, n):
        if phase_idx[i] != phase_idx[i - 1]:
            bounds.append(i)
    bounds.append(n)
    return [(bounds[k], bounds[k + 1]) for k in range(len(bounds) - 1)]


def _segment_top_edge(
    mu: np.ndarray,
    g_min: np.ndarray,
    start: int,
    end: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Envelope samples for band ``[start, end)``, extended to the right vertical at ``mu[end]``."""
    if end < mu.size:
        mu_top = mu[start:end + 1]
        g_top = g_min[start:end + 1]
    else:
        mu_top = mu[start:end]
        g_top = g_min[start:end]
    return mu_top, g_top


def _band_label_position(
    mu: np.ndarray,
    g_min: np.ndarray,
    start: int,
    end: int,
    y_bottom: float,
) -> tuple[float, float]:
    """Centre of a phase band: midway between verticals and between envelope and bottom."""
    mu_left = float(mu[start])
    mu_right = float(mu[end] if end < mu.size else mu[end - 1])
    mu_center = 0.5 * (mu_left + mu_right)
    mu_top, g_top = _segment_top_edge(mu, g_min, start, end)
    g_envelope = float(np.interp(mu_center, mu_top, g_top))
    y_center = 0.5 * (g_envelope + y_bottom)
    return mu_center, y_center


def _draw_envelope_overlay(
    ax,
    mu: np.ndarray,
    G_sub: np.ndarray,
    stoich_sub: np.ndarray,
    formulas_sub: np.ndarray,
    norm: Normalize,
    cmap,
    y_bottom: float,
) -> None:
    mu = np.asarray(mu, dtype=float)
    G_sub = np.asarray(G_sub, dtype=float)
    g_min = np.min(G_sub, axis=0)
    phase_idx = np.argmin(G_sub, axis=0)
    labels = stoichiometry_overlay_labels(formulas_sub)

    for start, end in _phase_segment_bounds(phase_idx):
        if end <= start:
            continue
        winner = int(phase_idx[start])
        color = cmap(norm(float(stoich_sub[winner])))
        mu_top, g_top = _segment_top_edge(mu, g_min, start, end)
        x_poly = np.concatenate([mu_top, mu_top[-1:], mu_top[:1]])
        y_poly = np.concatenate([g_top, [y_bottom, y_bottom]])
        ax.fill(
            x_poly,
            y_poly,
            color=color,
            zorder=1,
            linewidth=0,
            antialiased=False,
        )
        mu_label, y_label = _band_label_position(mu, g_min, start, end, y_bottom)
        ax.text(
            mu_label,
            y_label,
            labels[winner],
            color="black",
            fontsize=LABEL_FONTSIZE,
            ha="center",
            va="center",
            zorder=4,
        )

    for start, end in _phase_segment_bounds(phase_idx):
        if start == 0:
            continue
        i = start
        ax.plot(
            [mu[i], mu[i]],
            [g_min[i], y_bottom],
            color="black",
            linewidth=ENVELOPE_LINE_WIDTH,
            zorder=3,
        )

    ax.plot(
        mu,
        g_min,
        color="black",
        linewidth=ENVELOPE_LINE_WIDTH,
        solid_capstyle="round",
        zorder=3,
    )


def _plot_structure_lines(
    ax,
    mu: np.ndarray,
    G_sub: np.ndarray,
    stoich_sub: np.ndarray,
    formulas_sub: np.ndarray,
    norm: Normalize,
    cmap,
) -> LineCollection | None:
    mu = np.asarray(mu, dtype=float)
    segments_list: list[np.ndarray] = []
    colors_list: list[np.ndarray] = []
    for ox, g_curve in zip(stoich_sub, G_sub):
        points = np.column_stack([mu, np.asarray(g_curve, dtype=float)])
        segs = np.stack([points[:-1], points[1:]], axis=1)
        segments_list.append(segs)
        colors_list.append(np.full(segs.shape[0], float(ox), dtype=float))

    if not segments_list:
        return None

    segments = np.concatenate(segments_list, axis=0)
    seg_colors = np.concatenate(colors_list, axis=0)
    lc = LineCollection(
        segments,
        cmap=cmap,
        norm=norm,
        linewidth=LINE_WIDTH,
        capstyle="round",
        zorder=2,
    )
    lc.set_array(seg_colors)
    ax.add_collection(lc)

    ymin = float(np.nanmin(G_sub))
    ymax = float(np.nanmax(G_sub))
    pad = 0.05 * (ymax - ymin) if ymax > ymin else 0.05
    y_bottom = ymin - pad
    ax.set_ylim(y_bottom, ymax + pad)

    if formulas_sub.size:
        _draw_envelope_overlay(
            ax, mu, G_sub, stoich_sub, formulas_sub, norm, cmap, y_bottom,
        )

    # Rasterise all panel artists (lines, fills, envelope, labels) in vector export.
    ax.set_rasterization_zorder(5)

    return lc


def _panel_data(
    G_tensor: np.ndarray,
    stoich: np.ndarray,
    n_ir: np.ndarray,
    mu: np.ndarray,
    t_idx: int,
    formulas: np.ndarray,
) -> dict[int, Panel]:
    panels: dict[int, Panel] = {}
    for n in IR_COUNTS:
        result = structure_g_curves_by_ir_at_temperature(
            G_tensor, stoich, n_ir, t_idx, n, formulas=formulas,
        )
        if result is None:
            continue
        G_sub, stoich_sub, formulas_sub = result
        panels[int(n)] = (mu, G_sub, stoich_sub, formulas_sub)
    return panels


def _build_figure(
    temperature_k: float,
    pbe_panels: dict[int, Panel],
    pbe0_panels: dict[int, Panel],
    t_used: float,
    norm: Normalize,
    cmap,
    *,
    xlim_pbe: tuple[float, float],
    xlim_pbe0: tuple[float, float],
) -> None:
    fig, axes = plt.subplots(
        len(IR_COUNTS), 2,
        figsize=(FIG_W_MM / 25.4, FIG_H_MM / 25.4),
        dpi=DPI,
        sharex="col",
    )

    lc_ref = None
    col_labels = ("PBE", "PBE0")
    col_panels = (pbe_panels, pbe0_panels)
    col_xlims = (xlim_pbe, xlim_pbe0)
    last_row = len(IR_COUNTS) - 1

    for row, n_ir in enumerate(IR_COUNTS):
        for col, (label, panels) in enumerate(zip(col_labels, col_panels)):
            ax = axes[row, col]
            panel = panels.get(n_ir)
            if panel is None:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        fontsize=7, transform=ax.transAxes)
                ax.set_axis_off()
                continue

            mu, G_sub, stoich_sub, formulas_sub = panel
            lc = _plot_structure_lines(
                ax, mu, G_sub, stoich_sub, formulas_sub, norm, cmap,
            )
            if lc_ref is None:
                lc_ref = lc

            if row == 0:
                ax.set_title(label, fontsize=8, pad=3)
            if col == 0:
                ax.set_ylabel(f"Ir$_{{{n_ir}}}$\n$G$ (eV)", fontsize=8)
            else:
                ax.set_ylabel("")
                ax.tick_params(labelleft=False)
            if row < last_row:
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel(r"$\mu_{\mathrm{O_2}}$ (eV)", fontsize=8)
            ax.tick_params(axis="both", labelsize=7)

    for row in range(len(IR_COUNTS)):
        for col, xlim in enumerate(col_xlims):
            ax = axes[row, col]
            if ax.get_visible():
                ax.set_xlim(xlim)
                ax.margins(x=0)

    fig.suptitle(f"$T = {t_used:.0f}$ K", fontsize=9, y=1.01)

    fig.tight_layout(pad=6 / 25.4, rect=[0, 0, 0.88, 1])
    if lc_ref is not None:
        cbar = fig.colorbar(lc_ref, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)
        cbar.set_label(r"$n_O/n_{\mathrm{Ir}}$", fontsize=8)
        cbar.ax.tick_params(labelsize=7)

    out = _output_path(temperature_k)
    save_fig(fig, out, dpi=DPI)
    print(f"\n[supp] {out}")


def main() -> None:
    ensure_for_supplementary(["supp_gc_g_vs_mu"])
    ensure_dirs(OUTPUT_DIR)

    grid = load_npz(GC_VARIANTS_NPZ, keys=("Ts", "Ps"))
    Ts, Ps = grid["Ts"], grid["Ps"]
    mu_pbe = o2_chemical_potential_grid(
        O2_REF_PBE, Ts, Ps, novib=True, relative_to_molecule=True,
    )

    G_pbe, gc_pbe = load_gc_pbe_novib_g_tensor(GC_VARIANTS_NPZ)
    stoich_pbe = np.asarray(load_npz(GC_VARIANTS_NPZ, keys=("stoich",))["stoich"], dtype=float)
    formulas_pbe = load_npz(GC_NPZ, keys=("formulas",))["formulas"]
    rows_pbe = load_ace_features(ACE_FEATURES["PBE"])
    n_ir_pbe = np.array([r["n_Ir"] for r in rows_pbe[: gc_pbe["Pgi"].shape[0]]], dtype=int)

    gc0_data = load_npz(
        GC_NPZ_PBE0,
        keys=("Ts", "Ps", "G_tensor", "stoich", "formulas"),
    )
    mu_pbe0 = o2_chemical_potential_grid(
        O2_REF_PBE0, gc0_data["Ts"], gc0_data["Ps"],
        novib=True, relative_to_molecule=True,
    )
    stoich_pbe0 = np.asarray(gc0_data["stoich"], dtype=float)
    rows_pbe0 = load_ace_features(ACE_FEATURES["PBE0"])
    n_ir_pbe0 = np.array([r["n_Ir"] for r in rows_pbe0[: gc0_data["G_tensor"].shape[0]]], dtype=int)

    vmin, vmax = _stoich_limits(stoich_pbe, n_ir_pbe)
    vmin0, vmax0 = _stoich_limits(stoich_pbe0, n_ir_pbe0)
    norm = Normalize(vmin=min(vmin, vmin0), vmax=max(vmax, vmax0))
    cmap = plt.get_cmap(PHASE_CMAP)

    from aist.core.grandcanonical import temperature_grid_index

    for temperature_k in TEMPERATURES_K:
        t_idx_pbe, t_used = temperature_grid_index(Ts, temperature_k)
        mu = np.asarray(mu_pbe[:, t_idx_pbe], dtype=float)
        pbe_panels = _panel_data(
            G_pbe, stoich_pbe, n_ir_pbe, mu, t_idx_pbe, formulas_pbe,
        )

        t_idx_pbe0, _ = temperature_grid_index(gc0_data["Ts"], temperature_k)
        mu0 = np.asarray(mu_pbe0[:, t_idx_pbe0], dtype=float)
        pbe0_panels = _panel_data(
            gc0_data["G_tensor"], stoich_pbe0, n_ir_pbe0, mu0, t_idx_pbe0,
            gc0_data["formulas"],
        )
        _build_figure(
            temperature_k,
            pbe_panels,
            pbe0_panels,
            t_used,
            norm,
            cmap,
            xlim_pbe=_mu_span(mu),
            xlim_pbe0=_mu_span(mu0),
        )

    del G_pbe, gc_pbe, gc0_data


if __name__ == "__main__":
    main()
