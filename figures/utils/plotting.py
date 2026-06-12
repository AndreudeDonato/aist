"""Shared plotting utilities for paper figures (layout/styling only)."""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Patch
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize, to_rgb

from aist.ace.common import (
    DATASET_COLORS,
    discrete_palette,
    gradient_colormap,
    struct_cluster_palette,
)

plt.rcParams["svg.fonttype"] = "none"


# ---------------------------------------------------------------------------
# Phase-diagram axis helpers
# ---------------------------------------------------------------------------

def style_phase_ax(ax, xlabel="T (K)", ylabel=r"$p_{\mathrm{O_2}}$ (Pa)"):
    ax.set_yscale("log")
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(axis="both", labelsize=7)


def draw_phase_boundaries(ax, Ts, Ps, phase_index,
                          linewidth=1.0, color="black", alpha=1.0, linestyle="solid"):
    """Contour lines where the global-minimum structure index changes."""
    unique = np.unique(phase_index)
    if unique.size < 2:
        return
    levels = 0.5 * (unique[:-1] + unique[1:])
    ax.contour(Ts, Ps, phase_index, levels=levels,
               colors=color, linewidths=linewidth, alpha=alpha, linestyles=linestyle)


def stoichiometry_overlay_labels(formulas) -> list[str]:
    """Per-structure LaTeX formula labels for global-minimum phase overlays."""
    from aist.utils import to_latex_formula

    labels: list[str] = []
    for f in formulas:
        if hasattr(f, "item"):
            f = f.item()
        labels.append(to_latex_formula(str(f)))
    return labels


def cluster_overlay_labels(clusters: np.ndarray) -> list[str]:
    """Per-structure ``C*`` labels aligned with GC structure indices."""
    return [f"C{int(c)}" for c in np.asarray(clusters, dtype=int)]


def add_global_minimum_phase_labels(
    ax,
    Ts: np.ndarray,
    Ps: np.ndarray,
    phase_index: np.ndarray,
    structure_labels: list[str],
    *,
    fontsize: float = 7,
) -> None:
    """Label each global-minimum phase region (one label per structure index)."""
    phase_index = np.asarray(phase_index)
    for i in np.unique(phase_index):
        i = int(i)
        mask = phase_index == i
        if not np.any(mask):
            continue
        p_idx, t_idx = np.where(mask)
        t_center = float(Ts[int(np.round(np.mean(t_idx)))])
        p_center = float(Ps[int(np.round(np.mean(p_idx)))])
        ax.text(
            t_center,
            p_center,
            structure_labels[i],
            color="black",
            fontsize=fontsize,
            ha="center",
            va="center",
        )


def draw_global_minimum_phase_overlay(
    ax,
    Ts: np.ndarray,
    Ps: np.ndarray,
    phase_index: np.ndarray,
    structure_labels: list[str],
    *,
    linewidth: float = 1.0,
    label_fontsize: float = 7,
) -> None:
    """Phase boundaries + labels for the global-minimum structure at each region."""
    draw_phase_boundaries(ax, Ts, Ps, phase_index, linewidth=linewidth)
    add_global_minimum_phase_labels(
        ax, Ts, Ps, phase_index, structure_labels, fontsize=label_fontsize
    )


def draw_tchar_boundary(ax, Ts, Ps, tchar_map,
                        linewidth=1.5, color="black", alpha=1.0):
    """Draw T = Tchar contour (where T/Tchar crosses 1)."""
    Tg, _ = np.meshgrid(Ts, Ps, indexing="xy")
    ratio  = Tg / tchar_map
    ax.contour(Ts, Ps, ratio, levels=[1.0],
               colors=color, linewidths=linewidth, alpha=alpha, linestyles="solid")


# ---------------------------------------------------------------------------
# Colour helpers (from aist.ace.common)
# ---------------------------------------------------------------------------

def _stackplot_no_edges(
    ax,
    x: np.ndarray,
    Y: np.ndarray,
    colors,
    *,
    labels=None,
    zorder: int = 2,
):
    """Stacked population bands without per-band edge lines."""
    polys = ax.stackplot(
        x, Y,
        colors=colors,
        labels=labels,
        linewidth=0.0,
        edgecolor="none",
        zorder=zorder,
    )
    for pc in polys:
        pc.set_edgecolor("none")
        pc.set_linewidth(0.0)
    return polys


def population_weighted_rgb(pop_fractions: np.ndarray, colors: list) -> np.ndarray:
    """RGB field ``(nP, nT, 3)`` by mixing category colours with population fractions."""
    pop = np.asarray(pop_fractions, dtype=float)
    rgb = np.array([to_rgb(c) for c in colors], dtype=float)
    return np.einsum("kpj,kc->pjc", pop, rgb)


def draw_population_mixed_phase_panel(
    ax,
    Ts: np.ndarray,
    Ps: np.ndarray,
    pop_fractions: np.ndarray,
    colors: list,
    title: str,
    *,
    overlay_labels: list[str] | None = None,
    phase_index: np.ndarray | None = None,
):
    """Phase map coloured by population-weighted RGB mix of ``colors``."""
    RGB = population_weighted_rgb(pop_fractions, colors)
    mesh = ax.pcolormesh(Ts, Ps, RGB, shading="auto", rasterized=True)
    if overlay_labels is not None and phase_index is not None:
        draw_global_minimum_phase_overlay(ax, Ts, Ps, phase_index, overlay_labels)
    style_phase_ax(ax)
    ax.set_title(title, fontsize=7)
    return mesh


def draw_twin_population_vs_T_panel(
    ax,
    Ts: np.ndarray,
    x_avg: np.ndarray,
    x_phase: np.ndarray,
    Y: np.ndarray,
    stack_colors: list,
    stack_labels: list[str],
    *,
    title: str = "",
    pop_ylabel: str = "Population (%)",
    ratio_label: str = r"$n_O/n_{Ir}$",
    legend_ncol: int = 2,
):
    """Twin-axis slice: black oxidation lines (left y) + coloured population stack (right y)."""
    ax2 = ax.twinx()
    ax2.set_zorder(1)
    ax.set_zorder(2)
    ax.patch.set_visible(False)

    ax.plot(
        Ts, x_avg,
        color="black", linewidth=2, alpha=0.5,
        label=rf"$\langle {ratio_label}\rangle$ (ensemble)",
        zorder=3,
    )
    ax.step(
        Ts, x_phase,
        where="mid", color="black",
        linewidth=1.5, linestyle="--",
        label=f"Stable phase {ratio_label}",
        zorder=3,
    )

    if Y is not None and Y.size > 0:
        _stackplot_no_edges(ax2, Ts, Y, stack_colors, zorder=1)
    else:
        ax2.text(
            0.5, 0.5, "No population data",
            transform=ax2.transAxes,
            ha="center", va="center",
            fontsize=7, color="gray",
        )

    ax.set_xlabel("Temperature (K)", fontsize=8)
    ax.set_ylabel(ratio_label, fontsize=8)
    ax2.set_ylabel(pop_ylabel, fontsize=8)
    ax.set_xlim(float(Ts[0]), float(Ts[-1]))
    ax2.set_ylim(0.0, 100.0)
    if title:
        ax.set_title(title, fontsize=7)

    ox_handles, ox_labels = ax.get_legend_handles_labels()
    stack_handles = [
        Patch(facecolor=c, edgecolor="none", alpha=0.85)
        for c in stack_colors
    ]
    ax.legend(
        ox_handles + stack_handles,
        ox_labels + stack_labels,
        fontsize=5, frameon=False, loc="lower left", ncol=legend_ncol,
    )
    ax.tick_params(axis="both", labelsize=7)
    ax2.tick_params(axis="y", labelsize=7)


def draw_isomer_population_vs_T_panel(
    ax,
    Ts: np.ndarray,
    x_avg: np.ndarray,
    x_phase: np.ndarray,
    Y: np.ndarray | None,
    color_values: np.ndarray | None,
    *,
    stoich_cmap,
    stoich_norm: Normalize,
    pressure_pa: float,
    title: str = "",
    ratio_label: str = r"$n_O/n_{Ir}$",
):
    """Twin-axis panel: ensemble + stable-phase oxidation vs T, isomer pop. stack (right y)."""
    ax2 = ax.twinx()
    ax2.set_zorder(1)
    ax.set_zorder(2)
    ax.patch.set_visible(False)

    ax.plot(
        Ts, x_avg,
        color="black", linewidth=2, alpha=0.5,
        label=rf"$\langle {ratio_label}\rangle$ (ensemble)",
        zorder=3,
    )
    ax.step(
        Ts, x_phase,
        where="mid", color="black",
        linewidth=1.5, linestyle="--",
        label=f"Stable phase {ratio_label}",
        zorder=3,
    )

    if Y is not None and color_values is not None and Y.size > 0:
        colors = [stoich_cmap(stoich_norm(float(c))) for c in color_values]
        _stackplot_no_edges(ax2, Ts, Y, colors, zorder=1)
    else:
        ax2.text(
            0.5, 0.5,
            "No isomers above threshold",
            transform=ax2.transAxes,
            ha="center", va="center",
            fontsize=7, color="gray",
        )

    ax.set_xlabel("Temperature (K)", fontsize=8)
    ax.set_ylabel(ratio_label, fontsize=8)
    ax2.set_ylabel("Population (%)", fontsize=8)
    ax.set_xlim(float(Ts[0]), float(Ts[-1]))
    ax2.set_ylim(0.0, 100.0)
    if title:
        ax.set_title(title, fontsize=7)
    else:
        ax.set_title(
            rf"$p_{{\mathrm{{O_2}}}} = {pressure_pa:.3g}$ Pa",
            fontsize=7,
        )
    ax.legend(fontsize=6, loc="lower left", frameon=False)
    ax.tick_params(axis="both", labelsize=7)
    ax2.tick_params(axis="y", labelsize=7)

    sm = ScalarMappable(cmap=stoich_cmap, norm=stoich_norm)
    sm.set_array([])
    cb = plt.colorbar(sm, ax=[ax, ax2], pad=0.02, fraction=0.046)
    cb.set_label(f"{ratio_label} fraction", fontsize=8)
    cb.ax.tick_params(labelsize=7)


def draw_population_stack(
    ax,
    x_grid: np.ndarray,
    Y: np.ndarray,
    colors,
    *,
    labels=None,
    xlabel: str = r"$n_O / n_{Ir}$",
    ylabel: str = "Pop. (%)",
    title: str = "",
    legend_ncol: int = 2,
    boundary_after: int | None = None,
):
    """Draw a kernel-smoothed population stack (``Y`` shape ``(k, n_bins)``).

    ``boundary_after``: if set, draw a line at the cumulative top of rows ``0:boundary_after``
    (e.g. Ir motif count for an Ir | O stack).
    """
    k, n_bins = Y.shape
    bottom = np.zeros(n_bins)

    def _fill_layer(c: int) -> None:
        lab = labels[c] if labels is not None else f"C{c}"
        ax.fill_between(
            x_grid, bottom, bottom + Y[c],
            color=colors[c], alpha=0.85, label=lab,
            zorder=2,
            edgecolor="none",
            linewidth=0.0,
        )
        bottom[:] = bottom + Y[c]

    if boundary_after is not None and 0 < boundary_after < k:
        for c in range(boundary_after):
            _fill_layer(c)
        ir_top = bottom.copy()
        ax.plot(
            x_grid, ir_top,
            color="black", linewidth=0.6, solid_capstyle="butt",
            zorder=5, clip_on=False,
        )
        for c in range(boundary_after, k):
            _fill_layer(c)
    else:
        for c in range(k):
            _fill_layer(c)
    ax.set_xlim(float(x_grid[0]), float(x_grid[-1]))
    ax.set_ylim(0, 100)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=7)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=7)
    if title:
        ax.set_title(title, fontsize=7)
    ax.tick_params(axis="both", labelsize=6)
    if labels is not None:
        ax.legend(fontsize=5, frameon=False, ncol=legend_ncol, loc="upper left")

def cluster_colors(k: int):
    """tab10/tab20 colours for structure clusters."""
    cmap = struct_cluster_palette(k)
    return [cmap(c % cmap.N) for c in range(k)]


def ir_motif_colors(k: int):
    return discrete_palette("coolwarm", k)


def o_motif_colors(k: int):
    return discrete_palette("plasma_r", k)


def ensemble_motif_colormap(colors: list, k: int | None = None):
    """Smooth gradient through all motif/cluster colours (for ⟨index⟩ phase maps)."""
    k = k if k is not None else len(colors)
    return gradient_colormap(colors[:k], name="ensemble_motif")


def dominant_motif_cmap(colors: list, k: int):
    """Discrete ListedColormap + BoundaryNorm for a dominant-motif phase map."""
    cmap = ListedColormap([c for c in colors[:k]])
    bounds = np.arange(-0.5, k, 1.0)
    norm = BoundaryNorm(bounds, cmap.N)
    return cmap, norm


def discrete_category_colorbar(
    ax,
    colors: list,
    labels: list[str],
    *,
    cbar_label: str | None = None,
    pad: float = 0.02,
    fraction: float = 0.046,
):
    """Solid-block colorbar with one colour and tick label per category."""
    k = len(colors)
    if k == 0 or len(labels) != k:
        return None
    cmap = ListedColormap([c for c in colors[:k]])
    bounds = np.arange(-0.5, k, 1.0)
    norm = BoundaryNorm(bounds, cmap.N)
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = plt.colorbar(
        sm, ax=ax, pad=pad, fraction=fraction, ticks=np.arange(k),
    )
    cb.ax.set_yticklabels(labels, fontsize=7)
    cb.ax.tick_params(labelsize=7)
    if cbar_label:
        cb.set_label(cbar_label, fontsize=8)
    return cb


# ---------------------------------------------------------------------------
# Per-figure save helper
# ---------------------------------------------------------------------------

def save_fig(fig, path: str, dpi: int = 800):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {path}")
