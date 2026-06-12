"""Figure 9 – Hirshfeld parity plot + O |spin| KDE.

2×2 layout:
  Row 0: Hirshfeld q vs |spin| scatter (PBE | PBE0), coloured by atom motif
  Row 1: O atom |spin| KDE by O motif (PBE | PBE0)

Output: ``figures/outputs/fig09_tchar_motif_hs.svg``
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

from aist.ace.io import load_motif_atoms_json

from figures.config import DPI, MOTIF_JSON, OUTPUT_DIR, TRAJ_PBE, TRAJ_PBE0
from figures.utils.cache import ensure_for_figures
from figures.utils.io import ensure_dirs, load_traj
from figures.utils.plotting import save_fig

OUTPUT_PATH = os.path.join(OUTPUT_DIR, "fig09_tchar_motif_hs.svg")
FIG_W_MM = 180
ROW_H_MM = 60
FIG_H_MM = 2 * ROW_H_MM

DATASETS = [
    {
        "label": "PBE",
        "traj": TRAJ_PBE,
        "motif_json": MOTIF_JSON["PBE"],
    },
    {
        "label": "PBE0",
        "traj": TRAJ_PBE0,
        "motif_json": MOTIF_JSON["PBE0"],
    },
]

XLABEL = "Hirshfeld charge (e)"
YLABEL = "|Hirshfeld spin| (e)"


def _motif_marker(motif_tag: str) -> str:
    return "s" if motif_tag.startswith("Ir") else "o"


def _motif_order(motif: dict) -> list[str]:
    return list(motif.get("ir_motifs", [])) + list(motif.get("o_motifs", []))


def _o_motif_tags(motif: dict) -> list[str]:
    return list(motif.get("o_motifs", []))


def _hirshfeld_by_motif(
    traj_path: str,
    motif_json_path: str,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict]:
    motif = load_motif_atoms_json(motif_json_path)
    motif_order = _motif_order(motif)
    buckets: dict[str, list[tuple[float, float]]] = {tag: [] for tag in motif_order}

    for fi, atoms in enumerate(load_traj(traj_path)):
        labels = motif.get("frames", {}).get(str(fi))
        if labels is None:
            continue
        h = atoms.info.get("hirshfeld")
        if not isinstance(h, dict) or "charges" not in h or "spins" not in h:
            continue
        charges = np.asarray(h["charges"], dtype=float)
        spins = np.abs(np.asarray(h["spins"], dtype=float))
        if len(labels) != len(charges):
            continue
        for mot, q, s in zip(labels, charges, spins):
            if mot not in buckets:
                continue
            buckets[mot].append((float(q), float(s)))

    points = {
        tag: (
            np.array([p[0] for p in pts], dtype=float),
            np.array([p[1] for p in pts], dtype=float),
        )
        for tag, pts in buckets.items()
    }
    return points, motif


def _global_hirshfeld_limits(
    datasets: list[dict],
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Exact min/max Hirshfeld charge and |spin| across all atoms (both datasets)."""
    all_q: list[float] = []
    all_s: list[float] = []
    for ds in datasets:
        if not os.path.exists(ds["traj"]) or not os.path.exists(ds["motif_json"]):
            continue
        points, motif = _hirshfeld_by_motif(ds["traj"], ds["motif_json"])
        for tag in _motif_order(motif):
            q, s = points.get(tag, (np.array([]), np.array([])))
            if q.size:
                all_q.extend(q.tolist())
            if s.size:
                all_s.extend(s.tolist())
    if not all_q or not all_s:
        return None
    return (
        (float(np.min(all_q)), float(np.max(all_q))),
        (float(np.min(all_s)), float(np.max(all_s))),
    )


def _pin_limits(
    ax,
    *,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
) -> None:
    """Force axes to start/end at the given limits (no autoscale margin)."""
    ax.autoscale(enable=False)
    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])
    ax.margins(x=0, y=0)
    ax.set_xmargin(0)
    ax.set_ymargin(0)


def _charge_spin_panel(
    ax,
    points: dict[str, tuple[np.ndarray, np.ndarray]],
    motif: dict,
    title: str,
    *,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
) -> None:
    palette = motif.get("colors", {})
    motif_order = _motif_order(motif)
    n_total = sum(len(points[t][0]) for t in motif_order)
    if n_total == 0:
        ax.text(0.5, 0.5, "No Hirshfeld / motif data", ha="center", va="center",
                fontsize=7, transform=ax.transAxes)
        ax.set_title(title, fontsize=8)
        return

    for tag in motif_order:
        q, s = points.get(tag, (np.array([]), np.array([])))
        if q.size == 0:
            continue
        ax.scatter(
            q, s,
            c=palette.get(tag, "#888888"),
            marker=_motif_marker(tag),
            s=6,
            alpha=0.35,
            edgecolors="none",
            label=tag,
            rasterized=True,
        )

    ax.set_xlabel(XLABEL, fontsize=8)
    ax.set_ylabel(YLABEL, fontsize=8)
    ax.set_title(title, fontsize=8)
    ax.tick_params(axis="both", labelsize=7, colors="black")
    for spine in ax.spines.values():
        spine.set_color("black")
    ax.legend(fontsize=5, frameon=False, ncol=2, loc="best", markerscale=2)
    _pin_limits(ax, xlim=xlim, ylim=ylim)


def _kde_x_at_cumulative_fraction(
    data: np.ndarray,
    eval_xlim: tuple[float, float],
    fraction: float,
    *,
    n_pts: int = 400,
) -> float | None:
    """Rightmost x on the KDE grid where cumulative probability reaches ``fraction``."""
    curve = _kde_curve(data, n_pts=n_pts, xlim=eval_xlim)
    if curve is None:
        return None
    x, y = np.asarray(curve[0], dtype=float), np.asarray(curve[1], dtype=float)
    if x.size < 2 or not np.any(y > 0):
        return None
    dx = np.diff(x)
    mass = np.cumsum(0.5 * (y[:-1] + y[1:]) * dx)
    total = float(mass[-1])
    if total <= 0:
        return None
    idx = int(np.searchsorted(mass / total, fraction, side="left"))
    idx = min(idx, x.size - 1)
    return float(x[idx])


def _kde_spin_xlim_coverage(
    points: dict[str, tuple[np.ndarray, np.ndarray]],
    tags: list[str],
    eval_xlim: tuple[float, float],
    *,
    fraction: float = 0.95,
    n_pts: int = 400,
) -> tuple[float, float] | None:
    """x-axis from spin min to xmax covering ``fraction`` of each motif KDE."""
    xmax_vals: list[float] = []
    for tag in tags:
        _, s = points.get(tag, (np.array([]), np.array([])))
        x95 = _kde_x_at_cumulative_fraction(
            s, eval_xlim, fraction, n_pts=n_pts,
        )
        if x95 is not None:
            xmax_vals.append(x95)
    if not xmax_vals:
        return None
    return (eval_xlim[0], max(xmax_vals))


def _kde_curve(
    data: np.ndarray,
    n_pts: int = 200,
    *,
    xlim: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    data = np.asarray(data, dtype=float)
    if data.size < 2:
        return None
    if xlim is not None:
        x = np.linspace(xlim[0], xlim[1], n_pts)
    else:
        lo, hi = float(data.min()), float(data.max())
        pad = 1e-4 if hi <= lo else max((hi - lo) * 0.05, 1e-4)
        x = np.linspace(lo - pad, hi + pad, n_pts)
    return x, gaussian_kde(data)(x)


def _o_spin_kde_panel(
    ax,
    points: dict[str, tuple[np.ndarray, np.ndarray]],
    motif: dict,
    title: str,
    *,
    eval_xlim: tuple[float, float] | None = None,
    display_xlim: tuple[float, float] | None = None,
) -> None:
    palette = motif.get("colors", {})
    tags = _o_motif_tags(motif)
    ymax = 0.0
    plotted = 0

    for tag in tags:
        _, s = points.get(tag, (np.array([]), np.array([])))
        curve = _kde_curve(s, xlim=eval_xlim)
        if curve is None:
            continue
        x, y = curve
        ymax = max(ymax, float(y.max()))
        ax.plot(x, y, color=palette.get(tag, "#888888"), linewidth=1.2, label=tag)
        plotted += 1

    if plotted == 0:
        ax.text(0.5, 0.5, "No O Hirshfeld spin data", ha="center", va="center",
                fontsize=7, transform=ax.transAxes)
    ax.set_xlabel("|Hirshfeld spin| (e)", fontsize=8)
    ax.set_ylabel("Density", fontsize=8)
    ax.set_title(title, fontsize=8)
    ax.tick_params(axis="both", labelsize=7, colors="black")
    for spine in ax.spines.values():
        spine.set_color("black")
    ylim = (0.0, ymax * 1.05) if ymax > 0 else None
    _pin_limits(ax, xlim=display_xlim, ylim=ylim)
    if plotted:
        ax.legend(fontsize=5, frameon=False, ncol=2, loc="best")


def main() -> None:
    ensure_for_figures([9])
    ensure_dirs(OUTPUT_DIR)

    fig, axes = plt.subplots(
        2, 2, figsize=(FIG_W_MM / 25.4, FIG_H_MM / 25.4), dpi=DPI,
        sharex="row", sharey="row",
    )

    limits = _global_hirshfeld_limits(DATASETS)
    if limits is not None:
        xlim, ylim = limits
        spin_eval_xlim = ylim
    else:
        xlim, ylim, spin_eval_xlim = None, None, None

    kde_display_xlim = None
    if spin_eval_xlim is not None:
        kde_right_edges: list[float] = []
        for ds in DATASETS:
            if not os.path.exists(ds["traj"]) or not os.path.exists(ds["motif_json"]):
                continue
            points, motif = _hirshfeld_by_motif(ds["traj"], ds["motif_json"])
            tag_lim = _kde_spin_xlim_coverage(
                points, _o_motif_tags(motif), spin_eval_xlim,
            )
            if tag_lim is not None:
                kde_right_edges.append(tag_lim[1])
        if kde_right_edges:
            kde_display_xlim = (spin_eval_xlim[0], max(kde_right_edges))

    for col, ds in enumerate(DATASETS):
        ax_scatter = axes[0, col]
        ax_kde = axes[1, col]

        if not os.path.exists(ds["traj"]):
            for ax in (ax_scatter, ax_kde):
                ax.text(0.5, 0.5, "Trajectory missing", ha="center", va="center",
                        fontsize=7, transform=ax.transAxes)
                ax.set_axis_off()
            continue
        if not os.path.exists(ds["motif_json"]):
            for ax in (ax_scatter, ax_kde):
                ax.text(0.5, 0.5, "Motif JSON missing", ha="center", va="center",
                        fontsize=7, transform=ax.transAxes)
                ax.set_axis_off()
            continue

        points, motif = _hirshfeld_by_motif(ds["traj"], ds["motif_json"])

        _charge_spin_panel(
            ax_scatter, points, motif,
            f"{ds['label']} — Hirshfeld q vs |spin|",
            xlim=xlim,
            ylim=ylim,
        )
        _o_spin_kde_panel(
            ax_kde, points, motif,
            f"{ds['label']} — O |spin|",
            eval_xlim=spin_eval_xlim,
            display_xlim=kde_display_xlim,
        )

    fig.tight_layout(pad=6 / 25.4)

    if limits is not None:
        for col in range(2):
            _pin_limits(axes[0, col], xlim=xlim, ylim=ylim)
            kde_ylim = axes[1, col].get_ylim()
            _pin_limits(axes[1, col], xlim=kde_display_xlim, ylim=kde_ylim)

    save_fig(fig, OUTPUT_PATH, dpi=DPI)
    print(f"\n[fig09] {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
