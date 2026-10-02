"""Supplementary – G vs μ_O₂ per structure at fixed T (PBE | PBE0).

7×2 panels (Ir₃…Ir₉ × PBE | PBE0):
  one ``G_i`` line per structure vs ``μ_O₂``, coloured by that structure's
  ``n_O/n_Ir``. Both functionals at ``GC_COMPARISON_LEVEL``: "full" = G and μ_O₂ with vibrations
  (grand-canonical NPZs), "novib" = without (variants NPZs).

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
    DPI,
    GC_COMPARISON_LEVEL,
    OUTPUT_DIR,
    PHASE_CMAP,
    THERMO_IR_COUNTS,
)
from figures.utils.comparison import build_comparison_joined_table, load_gc_comparison_view
from figures.utils.cache import ensure_for_supplementary
from figures.utils.datasets import dataset_inputs
from figures.utils.gc_subset import ir_size_reservoir_mask, structure_g_curves_by_ir_at_temperature
from figures.utils.gc_thermo import o2_chemical_potential_grid
from figures.utils.io import (
    ensure_dirs,
    load_ace_features,
    load_gc_npz,
    load_gc_pbe_novib_g_tensor,
    load_npz,
)
from figures.utils.plotting import (
    save_fig,
    stoichiometry_overlay_labels,
    TITLE_FONTSIZE,
    AXIS_LABEL_FONTSIZE,
    TICK_FONTSIZE,
    CBAR_LABEL_FONTSIZE,
)

FIG_W_MM = 180
ROW_H_MM = 32
FIG_H_MM = len(THERMO_IR_COUNTS) * ROW_H_MM
IR_COUNTS = THERMO_IR_COUNTS
TEMPERATURES_K = (273.0, 500.0)
LINE_WIDTH = 0.7
ENVELOPE_LINE_WIDTH = 0.35

Panel = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def _output_path(temperature_k: float) -> str:
    return os.path.join(OUTPUT_DIR, f"supp_gc_g_vs_mu_{int(temperature_k)}K.svg")


def _stoich_limits(stoich: np.ndarray, n_ir: np.ndarray, formulas: np.ndarray) -> tuple[float, float]:
    vals: list[float] = [2.0, 3.0]
    for n in IR_COUNTS:
        mask = ir_size_reservoir_mask(n_ir, n, formulas)
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
            fontsize=AXIS_LABEL_FONTSIZE,
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
                        fontsize=AXIS_LABEL_FONTSIZE, transform=ax.transAxes)
                ax.set_axis_off()
                continue

            mu, G_sub, stoich_sub, formulas_sub = panel
            lc = _plot_structure_lines(
                ax, mu, G_sub, stoich_sub, formulas_sub, norm, cmap,
            )
            if lc_ref is None:
                lc_ref = lc

            if row == 0:
                ax.set_title(label, fontsize=TITLE_FONTSIZE, pad=3)
            if col == 0:
                ax.set_ylabel(f"Ir$_{{{n_ir}}}$\n$G$ (eV)", fontsize=AXIS_LABEL_FONTSIZE)
            else:
                ax.set_ylabel("")
                ax.tick_params(labelleft=False)
            if row < last_row:
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel(r"$\mu_{\mathrm{O_2}}$ (eV)", fontsize=AXIS_LABEL_FONTSIZE)
            ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)

    for row in range(len(IR_COUNTS)):
        for col, xlim in enumerate(col_xlims):
            ax = axes[row, col]
            if ax.get_visible():
                ax.set_xlim(xlim)
                ax.margins(x=0)

    fig.suptitle(f"$T = {t_used:.0f}$ K", fontsize=TITLE_FONTSIZE, y=1.01)

    fig.tight_layout(pad=6 / 25.4, rect=[0, 0, 0.88, 1])
    if lc_ref is not None:
        cbar = fig.colorbar(lc_ref, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)
        cbar.set_label(r"$n_O/n_{\mathrm{Ir}}$", fontsize=CBAR_LABEL_FONTSIZE)
        cbar.ax.tick_params(labelsize=TICK_FONTSIZE)

    out = _output_path(temperature_k)
    save_fig(fig, out, dpi=DPI)
    print(f"\n[supp] {out}")


def _g_vs_mu_inputs(label: str) -> dict:
    """G tensor, grid, stoichiometry, μ_O₂ − E_O₂ and per-row n_Ir / formulas of one functional."""
    inp = dataset_inputs(label)
    novib = GC_COMPARISON_LEVEL != "full"
    if novib:
        G, gc = load_gc_pbe_novib_g_tensor(inp.variants_npz)
        stoich = load_npz(inp.variants_npz, keys=("stoich",))["stoich"]
    else:
        gc = load_gc_npz(inp.gc_npz)
        G, stoich = gc["G_tensor"], gc["stoich"]
    jt = build_comparison_joined_table(load_gc_comparison_view(label), load_ace_features(inp.ace_features), label)
    return {
        "G": G,
        "Ts": gc["Ts"],
        "stoich": np.asarray(stoich, dtype=float),
        "mu": o2_chemical_potential_grid(inp.o2_ref, gc["Ts"], gc["Ps"], novib=novib, relative_to_molecule=True),
        "n_ir": np.asarray(jt["n_Ir"], dtype=int),
        "formulas": jt["formulas_gc"],
    }


def main() -> None:
    ensure_for_supplementary(["supp_gc_g_vs_mu"])
    ensure_dirs(OUTPUT_DIR)

    pbe, pbe0 = _g_vs_mu_inputs("PBE"), _g_vs_mu_inputs("PBE0")
    vmin, vmax = _stoich_limits(pbe["stoich"], pbe["n_ir"], pbe["formulas"])
    vmin0, vmax0 = _stoich_limits(pbe0["stoich"], pbe0["n_ir"], pbe0["formulas"])
    norm = Normalize(vmin=min(vmin, vmin0), vmax=max(vmax, vmax0))
    cmap = plt.get_cmap(PHASE_CMAP)

    from aist.core.grandcanonical import temperature_grid_index

    for temperature_k in TEMPERATURES_K:
        panels, mus, t_used = {}, {}, None
        for key, d in (("PBE", pbe), ("PBE0", pbe0)):
            t_idx, t = temperature_grid_index(d["Ts"], temperature_k)
            t_used = t if t_used is None else t_used
            mus[key] = np.asarray(d["mu"][:, t_idx], dtype=float)
            panels[key] = _panel_data(d["G"], d["stoich"], d["n_ir"], mus[key], t_idx, d["formulas"])
        _build_figure(
            temperature_k,
            panels["PBE"],
            panels["PBE0"],
            t_used,
            norm,
            cmap,
            xlim_pbe=_mu_span(mus["PBE"]),
            xlim_pbe0=_mu_span(mus["PBE0"]),
        )

    del pbe, pbe0


if __name__ == "__main__":
    main()
