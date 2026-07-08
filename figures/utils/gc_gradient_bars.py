"""Population-weighted gradient bars vs T (ensemble field, no global-minimum band)."""
from __future__ import annotations

from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize, to_rgb

from aist.core.grandcanonical import pressure_grid_index

from figures.utils.gc_subset import ir_size_reservoir_mask, subset_joined_table_by_n_ir
from figures.utils.comparison import gas_structure_mask, scalar_field_rgb_with_gas
from figures.utils.join import (
    cluster_population_fractions,
    ensemble_o_motif_per_ir_map,
    ensemble_weighted_mean_map,
    ensemble_weighted_rgb_map,
    ir_motif_population_fractions,
    o_motif_population_fractions,
)
from figures.utils.plotting import (
    ANNOTATION_FONTSIZE,
    AXIS_LABEL_FONTSIZE,
    TICK_FONTSIZE,
    stoichiometry_ratio_overlay_labels,
)

SectionKind = Literal["none", "global_min", "majority_cluster", "majority_ir", "majority_o"]

BAR_HEIGHT = 0.85


def ir_bar_heights(ir_counts: list[int]) -> dict[int, float]:
    """Uniform band height for every ``n_Ir`` row."""
    return {int(n): BAR_HEIGHT for n in ir_counts}


def style_ir_size_bar_yaxis(ax, ir_counts: list[int]) -> None:
    """``n_Ir`` bands top → bottom: Ir₃ … Ir₉ on a **linear** axis."""
    counts = list(ir_counts)
    ax.set_yscale("linear")
    ax.set_yticks(counts)
    ax.set_yticklabels([rf"Ir$_{{{n}}}$" for n in counts])
    pad = 0.5
    ax.set_ylim(max(counts) + pad, min(counts) - pad)


def _temperature_step(Ts: np.ndarray) -> float:
    if len(Ts) > 1:
        return float(Ts[1] - Ts[0])
    return 1.0


def temperature_xticks(ts: np.ndarray | None = None) -> np.ndarray:
    """Standard ``T`` axis ticks for the main GC grid (100–1000 K)."""
    from figures.config import GC_T_TICKS

    return np.asarray(GC_T_TICKS, dtype=int)


def bar_xlim(ts: np.ndarray) -> tuple[float, float]:
    """``xlim`` aligned with gradient-bar cell edges (half ``dT`` pad)."""
    ts = np.asarray(ts, dtype=float)
    dT = _temperature_step(ts)
    return float(ts[0] - 0.5 * dT), float(ts[-1] + 0.5 * dT)


def sync_temperature_xaxis(
    ax,
    ts: np.ndarray,
    *,
    xlabel: str | None = None,
) -> None:
    """``T`` tick **values** on every panel; optional ``xlabel`` (e.g. bottom row only)."""
    xlo, xhi = bar_xlim(ts)
    ax.set_xlim(xlo, xhi)
    ax.set_xticks(temperature_xticks(ts))
    ax.tick_params(axis="x", labelbottom=True, labelsize=TICK_FONTSIZE, pad=1.5)
    ax.set_xlabel(xlabel or "", fontsize=AXIS_LABEL_FONTSIZE)


def sync_temperature_xaxis_grid(
    axes: np.ndarray,
    ts: np.ndarray,
    *,
    xlabel: str = "T (K)",
) -> None:
    """Apply ``sync_temperature_xaxis``; ``xlabel`` only on the last grid row."""
    last_row = axes.shape[0] - 1
    for row in range(axes.shape[0]):
        for col in range(axes.shape[1]):
            ax = axes[row, col]
            if not ax.axison:
                continue
            sync_temperature_xaxis(
                ax, ts,
                xlabel=xlabel if row == last_row else None,
            )


def _subset_at_pressure(
    joined: dict,
    n_ir: int,
    pressure_pa: float,
    *,
    include_gas_reservoir: bool | None = None,
) -> tuple[dict, int, np.ndarray] | None:
    jt_sub = subset_joined_table_by_n_ir(
        joined, n_ir, include_gas_reservoir=include_gas_reservoir,
    )
    if jt_sub is None:
        return None
    p_idx, _ = pressure_grid_index(jt_sub["Ps"], pressure_pa)
    return jt_sub, p_idx, np.asarray(jt_sub["Ts"], dtype=float)


def _subset_per_structure(
    joined: dict,
    n_ir: int,
    per_structure: np.ndarray,
    *,
    include_gas_reservoir: bool | None = None,
) -> np.ndarray | None:
    formulas = joined.get("formulas_gc")
    mask = ir_size_reservoir_mask(
        joined["n_Ir"], n_ir, formulas, include_gas_reservoir=include_gas_reservoir,
    )
    if not np.any(mask):
        return None
    return np.asarray(per_structure, dtype=float)[mask]


def _section_index_1d(
    jt_sub: dict,
    p_idx: int,
    kind: SectionKind,
    *,
    section_labels: list[str] | None,
) -> tuple[np.ndarray, list[str]] | None:
    if kind == "none":
        return None
    if kind == "global_min":
        idx = np.asarray(jt_sub["phase_index"][p_idx, :], dtype=int)
        if section_labels is not None:
            labels = section_labels
        else:
            labels = stoichiometry_ratio_overlay_labels(
                jt_sub["n_O"], jt_sub["n_Ir"],
            )
        return idx, labels
    if kind == "majority_cluster":
        pop = cluster_population_fractions(jt_sub)
        idx = np.argmax(pop[:, p_idx, :], axis=0)
        labels = section_labels or [f"C{c}" for c in range(pop.shape[0])]
        return idx, labels
    if kind == "majority_ir":
        pop = ir_motif_population_fractions(jt_sub)
        idx = np.argmax(pop[:, p_idx, :], axis=0)
        labels = section_labels or [f"Ir-A{c}" for c in range(pop.shape[0])]
        return idx, labels
    if kind == "majority_o":
        pop = o_motif_population_fractions(jt_sub)
        idx = np.argmax(pop[:, p_idx, :], axis=0)
        labels = section_labels or [f"O-A{c}" for c in range(pop.shape[0])]
        return idx, labels
    return None


def draw_vertical_bar_sections(
    ax,
    Ts: np.ndarray,
    idx_1d: np.ndarray,
    y_center: float,
    bar_height: float,
    labels: list[str],
    *,
    fontsize: float = ANNOTATION_FONTSIZE,
) -> None:
    """Vertical dividers + region labels (phase-map style, at fixed ``pO₂``)."""
    idx_1d = np.asarray(idx_1d, dtype=int)
    ts = np.asarray(Ts, dtype=float)
    n_t = ts.size
    if n_t == 0:
        return

    y0 = y_center - 0.5 * bar_height
    y1 = y_center + 0.5 * bar_height

    for j in range(n_t - 1):
        if idx_1d[j] != idx_1d[j + 1]:
            x = 0.5 * (ts[j] + ts[j + 1])
            ax.plot([x, x], [y0, y1], color="black", linewidth=0.7, zorder=5)

    j = 0
    while j < n_t:
        k = j + 1
        while k < n_t and idx_1d[k] == idx_1d[j]:
            k += 1
        t_center = 0.5 * (ts[j] + ts[k - 1])
        lab_idx = int(idx_1d[j])
        if 0 <= lab_idx < len(labels):
            text = labels[lab_idx]
        else:
            text = str(lab_idx)
        ax.text(
            t_center, y_center, text,
            ha="center", va="center",
            fontsize=fontsize, color="black", zorder=6,
        )
        j = k


def cluster_only_phase_index_1d(
    joined: dict,
    n_ir: int,
    pressure_pa: float,
    *,
    include_gas_reservoir: bool | None = None,
) -> tuple[np.ndarray, list[str]] | None:
    """Global minimum over cluster rows only (gas excluded from phase boundaries)."""
    ctx = _subset_at_pressure(
        joined, n_ir, pressure_pa, include_gas_reservoir=include_gas_reservoir,
    )
    if ctx is None:
        return None
    jt_sub, p_idx, ts = ctx
    gas_mask = gas_structure_mask(jt_sub["formulas_gc"])
    cluster_idx = np.where(~gas_mask)[0]
    if cluster_idx.size == 0:
        return None
    pgi_slab = np.asarray(jt_sub["Pgi"][:, p_idx, :], dtype=float)
    local_argmax = np.argmax(pgi_slab[cluster_idx, :], axis=0)
    phase_index_1d = cluster_idx[local_argmax]
    labels = stoichiometry_ratio_overlay_labels(
        jt_sub["n_O"], jt_sub["n_Ir"],
    )
    return phase_index_1d, labels


def gas_fraction_vs_T_at_pressure(
    joined: dict,
    n_ir: int,
    pressure_pa: float,
    *,
    include_gas_reservoir: bool | None = None,
) -> tuple[np.ndarray, float, np.ndarray] | None:
    """Gas-reservoir population fraction vs ``T`` at fixed ``pO₂``."""
    ctx = _subset_at_pressure(
        joined, n_ir, pressure_pa, include_gas_reservoir=include_gas_reservoir,
    )
    if ctx is None:
        return None
    jt_sub, p_idx, ts = ctx
    gas_mask = gas_structure_mask(jt_sub["formulas_gc"])
    if not np.any(gas_mask):
        return ts, float(jt_sub["Ps"][p_idx]), np.zeros(ts.size, dtype=float)
    f_gas = np.asarray(jt_sub["Pgi"][gas_mask, p_idx, :], dtype=float).sum(axis=0)
    return ts, float(jt_sub["Ps"][p_idx]), f_gas


def _scalar_colors_with_gas_1d(
    values: np.ndarray,
    f_gas: np.ndarray,
    *,
    cmap: str,
    vmin: float,
    vmax: float,
) -> np.ndarray:
    """1D bar colours with white mixed in proportional to gas population."""
    rgb_2d = scalar_field_rgb_with_gas(
        np.asarray(values, dtype=float)[None, :],
        np.asarray(f_gas, dtype=float)[None, :],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    return rgb_2d[0]


def _annotate_sections(
    ax,
    joined: dict,
    n_ir: int,
    pressure_pa: float,
    *,
    kind: SectionKind,
    section_labels: list[str] | None,
    include_gas_reservoir: bool | None = None,
    phase_index_1d: np.ndarray | None = None,
) -> None:
    ctx = _subset_at_pressure(
        joined, n_ir, pressure_pa, include_gas_reservoir=include_gas_reservoir,
    )
    if ctx is None:
        return
    jt_sub, p_idx, ts = ctx
    heights = ir_bar_heights([n_ir])

    if phase_index_1d is not None:
        labels = section_labels or stoichiometry_ratio_overlay_labels(
            jt_sub["n_O"], jt_sub["n_Ir"],
        )
        draw_vertical_bar_sections(
            ax, jt_sub["Ts"], np.asarray(phase_index_1d, dtype=int),
            float(n_ir), heights[n_ir], labels,
        )
        return

    parsed = _section_index_1d(jt_sub, p_idx, kind, section_labels=section_labels)
    if parsed is None:
        return
    idx_1d, labels = parsed
    draw_vertical_bar_sections(
        ax, jt_sub["Ts"], idx_1d, float(n_ir), heights[n_ir], labels,
    )


def ensemble_scalar_vs_T_at_pressure(
    joined: dict,
    n_ir: int,
    per_structure: np.ndarray,
    pressure_pa: float,
    *,
    include_gas_reservoir: bool | None = None,
) -> tuple[np.ndarray, float, np.ndarray] | None:
    """Ensemble scalar vs ``T`` for one ``n_Ir`` bucket at fixed ``pO₂``."""
    ctx = _subset_at_pressure(
        joined, n_ir, pressure_pa, include_gas_reservoir=include_gas_reservoir,
    )
    if ctx is None:
        return None
    jt_sub, p_idx, ts = ctx
    per_sub = _subset_per_structure(
        joined, n_ir, per_structure, include_gas_reservoir=include_gas_reservoir,
    )
    if per_sub is None:
        return None
    field = ensemble_weighted_mean_map(jt_sub["Pgi"], per_sub)
    return ts, float(jt_sub["Ps"][p_idx]), np.asarray(field[p_idx, :], dtype=float)


def ensemble_o_motif_vs_T_at_pressure(
    joined: dict,
    n_ir: int,
    o_motif_index: int,
    pressure_pa: float,
) -> tuple[np.ndarray, float, np.ndarray] | None:
    """``⟨n_O^{motif} / n_Ir⟩`` vs ``T`` for one Ir-size bucket at fixed ``pO₂``."""
    ctx = _subset_at_pressure(joined, n_ir, pressure_pa)
    if ctx is None:
        return None
    jt_sub, p_idx, ts = ctx
    field = ensemble_o_motif_per_ir_map(jt_sub, o_motif_index)
    return ts, float(jt_sub["Ps"][p_idx]), np.asarray(field[p_idx, :], dtype=float)


def ensemble_rgb_vs_T_at_pressure(
    joined: dict,
    n_ir: int,
    per_structure_rgb: np.ndarray,
    pressure_pa: float,
) -> tuple[np.ndarray, float, np.ndarray] | None:
    """Population-weighted RGB vs ``T``; returns ``(Ts, p_used, rgb[n_T, 3])``."""
    ctx = _subset_at_pressure(joined, n_ir, pressure_pa)
    if ctx is None:
        return None
    jt_sub, p_idx, ts = ctx
    per_sub = _subset_per_structure(joined, n_ir, per_structure_rgb)
    if per_sub is None:
        return None
    rgb_map = ensemble_weighted_rgb_map(jt_sub["Pgi"], per_sub)
    return ts, float(jt_sub["Ps"][p_idx]), np.asarray(rgb_map[p_idx, :, :], dtype=float)


def per_structure_rgb_from_fractions(frac: np.ndarray, colors: list[str]) -> np.ndarray:
    """Mix discrete motif/cluster colours by per-structure fraction rows."""
    palette = np.array([to_rgb(c) for c in colors], dtype=float)
    return np.asarray(frac, dtype=float) @ palette


def per_structure_cluster_rgb(cluster: np.ndarray, colors: list[str]) -> np.ndarray:
    palette = np.array([to_rgb(c) for c in colors], dtype=float)
    return palette[np.asarray(cluster, dtype=int)]


def _draw_bar_layers(
    ax,
    joined: dict,
    pressure_pa: float,
    ir_counts: list[int],
    *,
    bar_colors_fn,
    section_kind: SectionKind = "none",
    section_labels: list[str] | None = None,
    include_gas_reservoir: bool | None = None,
    section_phase_index_by_ir: dict[int, np.ndarray] | None = None,
    xlim: tuple[float, float] | None = None,
) -> float:
    ts_ref = np.asarray(joined["Ts"], dtype=float)
    dT = _temperature_step(ts_ref)
    heights = ir_bar_heights(ir_counts)

    for n in ir_counts:
        t_plot, colors = bar_colors_fn(int(n))
        if t_plot is None:
            continue
        h = heights[int(n)]
        bars = ax.barh(
            np.full_like(t_plot, float(n), dtype=float),
            width=dT,
            left=t_plot - 0.5 * dT,
            height=h,
            color=colors,
            edgecolor="none",
            linewidth=0.0,
            antialiased=False,
            align="center",
        )
        for patch in bars:
            patch.set_rasterized(True)
        phase_override = (
            section_phase_index_by_ir.get(int(n))
            if section_phase_index_by_ir is not None else None
        )
        _annotate_sections(
            ax, joined, int(n), pressure_pa,
            kind=section_kind, section_labels=section_labels,
            include_gas_reservoir=include_gas_reservoir,
            phase_index_1d=phase_override,
        )

    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])
    else:
        xlo, xhi = bar_xlim(ts_ref)
        ax.set_xlim(xlo, xhi)
    style_ir_size_bar_yaxis(ax, ir_counts)
    return dT


def draw_scalar_gradient_bars(
    ax,
    joined: dict,
    pressure_pa: float,
    ir_counts: list[int],
    per_structure: np.ndarray,
    *,
    cmap: str,
    vmin: float,
    vmax: float,
    section_kind: SectionKind = "global_min",
    section_labels: list[str] | None = None,
    include_gas_reservoir: bool | None = None,
    gas_white_mix: bool = False,
    section_phase_index_by_ir: dict[int, np.ndarray] | None = None,
    xlim: tuple[float, float] | None = None,
) -> float:
    """Horizontal bars vs ``T``: one band per ``n_Ir``, colour = ensemble scalar."""
    norm = Normalize(vmin=vmin, vmax=vmax)
    cm = plt.get_cmap(cmap)

    def colors_fn(n: int):
        curve = ensemble_scalar_vs_T_at_pressure(
            joined, n, per_structure, pressure_pa,
            include_gas_reservoir=include_gas_reservoir,
        )
        if curve is None:
            return None, None
        t_plot, _, values = curve
        if not gas_white_mix:
            return t_plot, cm(norm(values))
        gas_curve = gas_fraction_vs_T_at_pressure(
            joined, n, pressure_pa, include_gas_reservoir=include_gas_reservoir,
        )
        if gas_curve is None:
            return t_plot, cm(norm(values))
        _, _, f_gas = gas_curve
        return t_plot, _scalar_colors_with_gas_1d(
            values, f_gas, cmap=cmap, vmin=vmin, vmax=vmax,
        )

    return _draw_bar_layers(
        ax, joined, pressure_pa, ir_counts,
        bar_colors_fn=colors_fn,
        section_kind=section_kind,
        section_labels=section_labels,
        include_gas_reservoir=include_gas_reservoir,
        section_phase_index_by_ir=section_phase_index_by_ir,
        xlim=xlim,
    )


def draw_rgb_gradient_bars(
    ax,
    joined: dict,
    pressure_pa: float,
    ir_counts: list[int],
    per_structure_rgb: np.ndarray,
    *,
    section_kind: SectionKind = "none",
    section_labels: list[str] | None = None,
) -> float:
    """Horizontal bars vs ``T``: one band per ``n_Ir``, colour = ensemble RGB mix."""

    def colors_fn(n: int):
        curve = ensemble_rgb_vs_T_at_pressure(
            joined, n, per_structure_rgb, pressure_pa,
        )
        if curve is None:
            return None, None
        t_plot, _, rgb = curve
        return t_plot, rgb

    return _draw_bar_layers(
        ax, joined, pressure_pa, ir_counts,
        bar_colors_fn=colors_fn,
        section_kind=section_kind,
        section_labels=section_labels,
    )


def draw_o_motif_gradient_bars(
    ax,
    joined: dict,
    pressure_pa: float,
    o_motif_index: int,
    ir_counts: list[int],
    *,
    cmap: str,
    vmin: float,
    vmax: float,
    section_kind: SectionKind = "global_min",
    section_labels: list[str] | None = None,
) -> float:
    """Horizontal bars coloured by ``⟨n_O^{motif} / n_Ir⟩`` (Fig. 4 supplementary)."""
    norm = Normalize(vmin=vmin, vmax=vmax)
    cm = plt.get_cmap(cmap)

    def colors_fn(n: int):
        curve = ensemble_o_motif_vs_T_at_pressure(
            joined, n, o_motif_index, pressure_pa,
        )
        if curve is None:
            return None, None
        t_plot, _, values = curve
        return t_plot, cm(norm(values))

    return _draw_bar_layers(
        ax, joined, pressure_pa, ir_counts,
        bar_colors_fn=colors_fn,
        section_kind=section_kind,
        section_labels=section_labels,
    )
