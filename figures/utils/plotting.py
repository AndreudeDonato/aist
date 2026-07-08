"""Shared plotting utilities for paper figures (layout/styling only)."""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize, to_rgb

from aist.ace.common import (
    discrete_palette,
    struct_cluster_palette,
)
from aist.core.grandcanonical import global_minimum_population_map

plt.rcParams["svg.fonttype"] = "none"

# Paper figure typography (titles / axis labels / ticks / legends)
TITLE_FONTSIZE = 8
AXIS_LABEL_FONTSIZE = 7
TICK_FONTSIZE = 6
LEGEND_FONTSIZE = 6
CBAR_LABEL_FONTSIZE = AXIS_LABEL_FONTSIZE
ANNOTATION_FONTSIZE = AXIS_LABEL_FONTSIZE  # placeholder text, phase-map overlays

GLOBAL_MIN_POP_CMAP = "PuBu"
PHASE_COLORBAR_KW = {
    "location": "right",
    "pad": 0.06,
    "fraction": 0.04,
    "shrink": 0.92,
}


def add_phase_colorbar(
    fig,
    mappable,
    ax=None,
    label: str | None = None,
    *,
    cax=None,
    label_fontsize: float = CBAR_LABEL_FONTSIZE,
    tick_fontsize: float = TICK_FONTSIZE,
):
    """Shared vertical colorbar style (length/padding matched to Figure 7).

    Pass ``cax`` to draw in a fixed axes (plot panels keep their size).
    Pass ``ax`` to attach to panel axes (shrinks them, as in Figure 7).
    """
    if cax is not None:
        cbar = fig.colorbar(mappable, cax=cax)
    elif ax is not None:
        cbar = fig.colorbar(mappable, ax=ax, **PHASE_COLORBAR_KW)
    else:
        raise ValueError("add_phase_colorbar requires ax or cax")
    if label:
        cbar.set_label(label, fontsize=label_fontsize)
    cbar.ax.tick_params(labelsize=tick_fontsize)
    return cbar


def margin_colorbar_cax(
    fig,
    ax_left,
    ax_right,
    *,
    gap: float = 0.015,
    width: float = 0.012,
):
    """Dedicated colorbar axes in the tight_layout right margin (no panel resize)."""
    pos_l = ax_left.get_position()
    pos_r = ax_right.get_position()
    x = pos_r.x1 + gap
    return fig.add_axes([x, pos_l.y0, width, pos_l.height])


def add_fig06_interface_pbe_pbe0_diff_axis(
    ax,
    Ts: np.ndarray,
    pbe_y: np.ndarray,
    pbe0_y: np.ndarray,
) -> None:
    """Twin y-axis: dotted zero line + thin black PBE−PBE0 difference (behind main curves)."""
    ax2 = ax.twinx()
    ax2.set_zorder(0)
    ax.set_zorder(1)
    ax.patch.set_visible(False)

    diff = np.asarray(pbe_y, dtype=float) - np.asarray(pbe0_y, dtype=float)
    ax2.axhline(
        0.0, color="black", linewidth=0.5, linestyle="dotted", zorder=1,
    )
    ax2.plot(Ts, diff, color="black", linewidth=0.6, zorder=2)
    dmin, dmax = float(np.min(diff)), float(np.max(diff))
    ax2.set_ylim(min(dmin, 0.0), max(dmax, 0.0))
    ax2.tick_params(axis="y", labelsize=TICK_FONTSIZE, colors="black")
    ax2.set_ylabel("PBE−PBE0", fontsize=AXIS_LABEL_FONTSIZE)


# ---------------------------------------------------------------------------
# Phase-diagram axis helpers
# ---------------------------------------------------------------------------

def apply_gc_temperature_xticks(ax) -> None:
    """Fixed ``T`` ticks for the main GC grid (100–1000 K)."""
    from figures.config import GC_T_TICKS

    ax.set_xticks(GC_T_TICKS)


def style_gradient_bar_panel_ax(
    ax,
    row: int,
    col: int,
    last_row: int,
    *,
    title: str = "",
) -> None:
    """Shared-axis layout for per-size gradient-bar panels (linear ``n_Ir`` axis)."""
    ax.set_yscale("linear")
    if row == 0 and title:
        ax.set_title(title, fontsize=TITLE_FONTSIZE, pad=3)
    if col == 0:
        ax.set_ylabel(r"$n_{\mathrm{Ir}}$", fontsize=AXIS_LABEL_FONTSIZE)
    else:
        ax.set_ylabel("")
        ax.tick_params(labelleft=False)
    if row < last_row:
        ax.tick_params(labelbottom=False)
        ax.set_xlabel("")
    else:
        ax.set_xlabel("T (K)", fontsize=AXIS_LABEL_FONTSIZE)
    apply_gc_temperature_xticks(ax)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)


def style_shared_gc_panel_ax(
    ax,
    row: int,
    col: int,
    last_row: int,
    *,
    title: str = "",
) -> None:
    """Shared-axis GC map grid (e.g. Fig. 3: PBE | PBE0 columns)."""
    ax.set_yscale("log")
    if row == 0 and title:
        ax.set_title(title, fontsize=TITLE_FONTSIZE, pad=3)
    if col == 0:
        ax.set_ylabel(r"$p_{\mathrm{O_2}}$ (Pa)", fontsize=AXIS_LABEL_FONTSIZE)
    else:
        ax.set_ylabel("")
        ax.tick_params(labelleft=False)
    if row < last_row:
        ax.tick_params(labelbottom=False)
        ax.set_xlabel("")
    else:
        ax.set_xlabel("T (K)", fontsize=AXIS_LABEL_FONTSIZE)
    apply_gc_temperature_xticks(ax)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)


def style_phase_ax(
    ax,
    xlabel="T (K)",
    ylabel=r"$p_{\mathrm{O_2}}$ (Pa)",
    *,
    apply_gc_t_ticks: bool = True,
):
    ax.set_yscale("log")
    ax.set_xlabel(xlabel, fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    if apply_gc_t_ticks:
        apply_gc_temperature_xticks(ax)


def style_ir_size_panel_ax(
    ax,
    n_ir: int,
    col: int,
    row: int,
    last_row: int,
    *,
    title: str = "",
) -> None:
    """Ir-size GC grid panels (shared-axis layout, matching supp_gc_g_vs_mu)."""
    ax.set_yscale("log")
    if row == 0 and title:
        ax.set_title(title, fontsize=TITLE_FONTSIZE, pad=3)
    if col == 0:
        ax.set_ylabel(
            f"Ir$_{{{n_ir}}}$\n$p_{{\\mathrm{{O_2}}}}$ (Pa)",
            fontsize=AXIS_LABEL_FONTSIZE,
        )
    else:
        ax.set_ylabel("")
        ax.tick_params(labelleft=False)
    if row < last_row:
        ax.tick_params(labelbottom=False)
        ax.set_xlabel("")
    else:
        ax.set_xlabel("T (K)", fontsize=AXIS_LABEL_FONTSIZE)
        apply_gc_temperature_xticks(ax)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)


def draw_phase_boundaries(ax, Ts, Ps, phase_index,
                          linewidth=1.0, color="black", alpha=1.0, linestyle="solid"):
    """Contour lines where the global-minimum structure index changes."""
    unique = np.unique(phase_index)
    if unique.size < 2:
        return
    levels = 0.5 * (unique[:-1] + unique[1:])
    ax.contour(Ts, Ps, phase_index, levels=levels,
               colors=color, linewidths=linewidth, alpha=alpha, linestyles=linestyle)


def stoichiometry_ratio_overlay_labels(
    n_O,
    n_Ir,
    *,
    decimals: int = 1,
) -> list[str]:
    """Per-structure ``n_O/n_Ir`` labels for global-minimum phase overlays."""
    n_o = np.asarray(n_O, dtype=float)
    n_ir = np.asarray(n_Ir, dtype=float)
    ratio = np.where(n_ir > 0, n_o / n_ir, np.nan)
    labels: list[str] = []
    for r in ratio:
        if np.isfinite(r):
            labels.append(f"{float(r):.{decimals}f}")
        else:
            labels.append("—")
    return labels


def stoichiometry_overlay_labels(formulas) -> list[str]:
    """Per-structure LaTeX formula labels for global-minimum phase overlays."""
    from aist.utils import to_latex_formula

    gas_labels = {
        "IrO2": r"IrO$_2$(g)",
        "IrO3": r"IrO$_3$(g)",
        "Ir2O4": r"IrO$_2$(rutile)",
    }
    labels: list[str] = []
    for f in formulas:
        if hasattr(f, "item"):
            f = f.item()
        s = str(f)
        if s in gas_labels:
            labels.append(gas_labels[s])
        else:
            labels.append(to_latex_formula(s))
    return labels


def cluster_overlay_labels(clusters: np.ndarray) -> list[str]:
    """Per-structure ``C*`` labels aligned with GC structure indices."""
    return [f"C{int(c)}" for c in np.asarray(clusters, dtype=int)]


def majority_category_index(pop_fractions: np.ndarray) -> np.ndarray:
    """Index of the dominant category at each ``(p, T)``; shape ``(nP, nT)``."""
    return np.argmax(np.asarray(pop_fractions, dtype=float), axis=0)


def add_majority_category_phase_labels(
    ax,
    Ts: np.ndarray,
    Ps: np.ndarray,
    pop_fractions: np.ndarray,
    category_labels: list[str],
    *,
    fontsize: float = ANNOTATION_FONTSIZE,
) -> None:
    """Label each majority-category region with ``category_labels[c]``."""
    idx = majority_category_index(pop_fractions)
    add_global_minimum_phase_labels(
        ax, Ts, Ps, idx, category_labels, fontsize=fontsize,
    )


def draw_majority_category_phase_overlay(
    ax,
    Ts: np.ndarray,
    Ps: np.ndarray,
    pop_fractions: np.ndarray,
    category_labels: list[str],
    *,
    linewidth: float = 1.0,
    label_fontsize: float = ANNOTATION_FONTSIZE,
) -> None:
    """Phase boundaries + labels for the population-majority category at each region."""
    idx = majority_category_index(pop_fractions)
    draw_phase_boundaries(ax, Ts, Ps, idx, linewidth=linewidth)
    add_majority_category_phase_labels(
        ax, Ts, Ps, pop_fractions, category_labels, fontsize=label_fontsize,
    )


def add_global_minimum_phase_labels(
    ax,
    Ts: np.ndarray,
    Ps: np.ndarray,
    phase_index: np.ndarray,
    structure_labels: list[str],
    *,
    fontsize: float = ANNOTATION_FONTSIZE,
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
    label_fontsize: float = ANNOTATION_FONTSIZE,
) -> None:
    """Phase boundaries + labels for the global-minimum structure at each region."""
    draw_phase_boundaries(ax, Ts, Ps, phase_index, linewidth=linewidth)
    add_global_minimum_phase_labels(
        ax, Ts, Ps, phase_index, structure_labels, fontsize=label_fontsize
    )


def draw_global_minimum_population_panel(
    ax,
    gc: dict,
    overlay_labels: list[str],
    *,
    title: str = "",
    title_fontsize: float = TITLE_FONTSIZE,
):
    """PuBu map of global-minimum structure population (%) with phase overlay."""
    Ts, Ps = gc["Ts"], gc["Ps"]
    minima_pop = global_minimum_population_map(gc)
    mesh = ax.pcolormesh(
        Ts, Ps, minima_pop, shading="auto",
        cmap=GLOBAL_MIN_POP_CMAP,
        norm=Normalize(vmin=0.0, vmax=100.0),
        rasterized=True,
    )
    draw_global_minimum_phase_overlay(
        ax, Ts, Ps, gc["phase_index"], overlay_labels,
    )
    style_phase_ax(ax)
    if title:
        ax.set_title(title, fontsize=title_fontsize)
    return mesh


def draw_tchar_boundary(ax, Ts, Ps, tchar_map,
                        linewidth=1.5, color="black", alpha=1.0):
    """Draw T = Tchar contour (where T/Tchar crosses 1)."""
    Tg, _ = np.meshgrid(Ts, Ps, indexing="xy")
    ratio  = Tg / tchar_map
    ax.contour(Ts, Ps, ratio, levels=[1.0],
               colors=color, linewidths=linewidth, alpha=alpha, linestyles="solid")


def draw_gvib_zero_contour(
    ax,
    Ts,
    Ps,
    gvib_map,
    *,
    linewidth: float = 1.4,
    color: str = "#ff1a1a",
    alpha: float = 1.0,
) -> None:
    """Draw ⟨G_vib⟩ = 0 contour on vibrational Gibbs contribution maps."""
    Z = np.asarray(gvib_map, dtype=float)
    if not np.isfinite(Z).any():
        return
    ax.contour(
        Ts, Ps, Z, levels=[0.0],
        colors=color, linewidths=linewidth, alpha=alpha, linestyles="solid",
    )


def draw_vib_reference_contours(
    ax,
    Ts,
    Ps,
    *,
    tchar_map=None,
    gvib_map=None,
) -> None:
    """Overlay T = ⟨Θ_eff⟩ (black) and ⟨G_vib⟩ = 0 (red) on vib panels."""
    if tchar_map is not None and np.isfinite(np.asarray(tchar_map)).any():
        draw_tchar_boundary(ax, Ts, Ps, tchar_map)
    if gvib_map is not None and np.isfinite(np.asarray(gvib_map)).any():
        draw_gvib_zero_contour(ax, Ts, Ps, gvib_map)


# ---------------------------------------------------------------------------
# Colour helpers (from aist.ace.common)
# ---------------------------------------------------------------------------

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
    majority_labels: list[str] | None = None,
):
    """Phase map coloured by population-weighted RGB mix of ``colors``."""
    RGB = population_weighted_rgb(pop_fractions, colors)
    mesh = ax.pcolormesh(Ts, Ps, RGB, shading="auto", rasterized=True)
    if majority_labels is not None:
        draw_majority_category_phase_overlay(ax, Ts, Ps, pop_fractions, majority_labels)
    elif overlay_labels is not None and phase_index is not None:
        draw_global_minimum_phase_overlay(ax, Ts, Ps, phase_index, overlay_labels)
    style_phase_ax(ax)
    if title:
        ax.set_title(title, fontsize=TITLE_FONTSIZE)
    return mesh


def draw_shape_extent_rgb_phase_panel(
    ax,
    Ts: np.ndarray,
    Ps: np.ndarray,
    rgb_field: np.ndarray,
    title: str,
    *,
    overlay_labels: list[str] | None = None,
    phase_index: np.ndarray | None = None,
):
    """Phase map from precomputed RGB field ``(nP, nT, 3)`` (inertia shape × extent)."""
    RGB = np.clip(np.asarray(rgb_field, dtype=float), 0.0, 1.0)
    mesh = ax.pcolormesh(Ts, Ps, RGB, shading="auto", rasterized=True)
    if overlay_labels is not None and phase_index is not None:
        draw_global_minimum_phase_overlay(ax, Ts, Ps, phase_index, overlay_labels)
    style_phase_ax(ax)
    if title:
        ax.set_title(title, fontsize=TITLE_FONTSIZE)
    return mesh


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
        ax.set_xlabel(xlabel, fontsize=AXIS_LABEL_FONTSIZE)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FONTSIZE)
    if title:
        ax.set_title(title, fontsize=TITLE_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    if labels is not None:
        ax.legend(fontsize=LEGEND_FONTSIZE, frameon=False, ncol=legend_ncol, loc="upper left")

def cluster_colors(k: int):
    """tab10/tab20 colours for structure clusters."""
    cmap = struct_cluster_palette(k)
    return [cmap(c % cmap.N) for c in range(k)]


def ir_motif_colors(k: int):
    return discrete_palette("coolwarm", k)


def o_motif_colors(k: int):
    return discrete_palette("plasma_r", k)


def discrete_category_colorbar(
    colors: list,
    labels: list[str],
    *,
    ax=None,
    cbar_label: str | None = None,
    pad: float = 0.06,
    fraction: float = 0.04,
    shrink: float = 0.92,
    cax=None,
    label_fontsize: float = CBAR_LABEL_FONTSIZE,
):
    """Solid-block colorbar with one colour and tick label per category.

    ``ax`` may be a single axes or an array/list of axes (shared colorbar).
    When ``cax`` is given, the colorbar is drawn on that axes instead.
    """
    k = len(colors)
    if k == 0 or len(labels) != k:
        return None
    cmap = ListedColormap([c for c in colors[:k]])
    bounds = np.arange(-0.5, k, 1.0)
    norm = BoundaryNorm(bounds, cmap.N)
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    if cax is not None:
        cb = plt.colorbar(sm, cax=cax, ticks=np.arange(k))
    elif ax is not None:
        cb = plt.colorbar(
            sm, ax=ax, pad=pad, fraction=fraction, shrink=shrink,
            ticks=np.arange(k),
        )
    else:
        return None
    cb.ax.set_yticklabels(labels, fontsize=TICK_FONTSIZE)
    cb.ax.tick_params(labelsize=TICK_FONTSIZE)
    if cbar_label:
        cb.set_label(cbar_label, fontsize=label_fontsize)
    return cb


# ---------------------------------------------------------------------------
# Per-figure save helper
# ---------------------------------------------------------------------------

def save_fig(fig, path: str, dpi: int = 800, pad_inches: float = 0.1):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=pad_inches)
    plt.close(fig)
    print(f"[save] {path}")
