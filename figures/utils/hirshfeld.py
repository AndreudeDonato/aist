"""Hirshfeld charge / spin analysis helpers for paper figures."""
from __future__ import annotations

import os

import numpy as np
from scipy.stats import gaussian_kde

from aist.ace.common import ace_atom_display_from_key
from aist.ace.io import load_motif_atoms_json

from figures.utils.io import load_traj

XLABEL = "Hirshfeld charge (e)"
YLABEL = "|Hirshfeld spin| (e)"


def motif_marker(motif_tag: str) -> str:
    return "s" if motif_tag.startswith("Ir") else "o"


def motif_order(motif: dict) -> list[str]:
    return list(motif.get("ir_motifs", [])) + list(motif.get("o_motifs", []))


def o_motif_tags(motif: dict) -> list[str]:
    return list(motif.get("o_motifs", []))


def hirshfeld_by_motif(
    traj_path: str,
    motif_json_path: str,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict]:
    motif = load_motif_atoms_json(motif_json_path)
    order = motif_order(motif)
    buckets: dict[str, list[tuple[float, float]]] = {tag: [] for tag in order}

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


def global_hirshfeld_limits(
    datasets: list[dict],
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Exact min/max Hirshfeld charge and |spin| across all atoms (all datasets)."""
    all_q: list[float] = []
    all_s: list[float] = []
    for ds in datasets:
        if not os.path.exists(ds["traj"]) or not os.path.exists(ds["motif_json"]):
            continue
        points, motif = hirshfeld_by_motif(ds["traj"], ds["motif_json"])
        for tag in motif_order(motif):
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


def pin_limits(
    ax,
    *,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
) -> None:
    ax.autoscale(enable=False)
    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])
    ax.margins(x=0, y=0)
    ax.set_xmargin(0)
    ax.set_ymargin(0)


def kde_curve(
    data: np.ndarray,
    n_pts: int = 200,
    *,
    xlim: tuple[float, float] | None = None,
    weights: np.ndarray | None = None,
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
    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        w_sum = float(weights.sum())
        if w_sum <= 0:
            return None
        weights = weights / w_sum
        return x, gaussian_kde(data, weights=weights)(x)
    return x, gaussian_kde(data)(x)


def charge_spin_panel(
    ax,
    points: dict[str, tuple[np.ndarray, np.ndarray]],
    motif: dict,
    title: str,
    *,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
) -> None:
    palette = motif.get("colors", {})
    order = motif_order(motif)
    n_total = sum(len(points[t][0]) for t in order)
    if n_total == 0:
        ax.text(0.5, 0.5, "No Hirshfeld / motif data", ha="center", va="center",
                fontsize=7, transform=ax.transAxes)
        ax.set_title(title, fontsize=8)
        return

    for tag in order:
        q, s = points.get(tag, (np.array([]), np.array([])))
        if q.size == 0:
            continue
        ax.scatter(
            q, s,
            c=palette.get(tag, "#888888"),
            marker=motif_marker(tag),
            s=6,
            alpha=0.35,
            edgecolors="none",
            label=ace_atom_display_from_key(tag),
            rasterized=True,
        )

    ax.set_xlabel(XLABEL, fontsize=8)
    ax.set_ylabel(YLABEL, fontsize=8)
    ax.set_title(title, fontsize=8)
    ax.tick_params(axis="both", labelsize=7, colors="black")
    for spine in ax.spines.values():
        spine.set_color("black")
    ax.legend(fontsize=5, frameon=False, ncol=2, loc="best", markerscale=2)
    pin_limits(ax, xlim=xlim, ylim=ylim)


def o_spin_kde_panel(
    ax,
    points: dict[str, tuple[np.ndarray, np.ndarray]],
    motif: dict,
    title: str,
    *,
    eval_xlim: tuple[float, float] | None = None,
    display_xlim: tuple[float, float] | None = None,
    show_xlabel: bool = True,
    show_ylabel: bool = True,
    corner_label: str | None = None,
) -> None:
    palette = motif.get("colors", {})
    tags = o_motif_tags(motif)
    # O-A2 omitted in paper figures (sparse motif); uncomment next line to restore
    tags = [t for t in tags if t != "O-A2"]
    ymax = 0.0
    plotted = 0

    for tag in tags:
        _, s = points.get(tag, (np.array([]), np.array([])))
        curve = kde_curve(s, xlim=eval_xlim)
        if curve is None:
            continue
        x, y = curve
        ymax = max(ymax, float(y.max()))
        ax.plot(x, y, color=palette.get(tag, "#888888"), linewidth=1.2,
                label=ace_atom_display_from_key(tag))
        plotted += 1

    if plotted == 0:
        ax.text(0.5, 0.5, "No O Hirshfeld spin data", ha="center", va="center",
                fontsize=7, transform=ax.transAxes)
    if show_xlabel:
        ax.set_xlabel("|Hirshfeld spin| (e)", fontsize=8)
    if show_ylabel:
        ax.set_ylabel("Density", fontsize=8)
    if title:
        ax.set_title(title, fontsize=8)
    ax.tick_params(axis="both", labelsize=7, colors="black")
    for spine in ax.spines.values():
        spine.set_color("black")
    ylim = (0.0, ymax * 1.05) if ymax > 0 else None
    pin_limits(ax, xlim=display_xlim, ylim=ylim)
    if plotted:
        ax.legend(
            fontsize=5, frameon=False,
            ncol=1 if corner_label else 2,
            loc="upper right",
        )
        if corner_label:
            ax.text(
                0.98, 0.58, corner_label,
                transform=ax.transAxes,
                ha="right", va="top", fontsize=7, color="black",
            )
