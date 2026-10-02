"""Supplementary – curated dataset composition per stoichiometry (one figure per functional).

Rows Ir_3 … Ir_9, columns n_O. Two tile maps per dataset:
  (a) number of retained structures per Ir_yO_x stoichiometry
  (b) energy range of those structures above the lowest one, per atom: max_i (E_i − E_min) / N_atoms
      (light-basis RE energies; compare with the 0.15 eV/atom GOFEE selection window of the Methods)

Inputs: ``data/IryOxPBE.traj``, ``data/IryOxPBE0.traj``.
Output: ``figures/outputs/supp_stoichiometry_tilemap_PBE.svg``, ``…_PBE0.svg``
(the paper's "Curated dataset overview", fig:si-tilemap-pbe / fig:si-tilemap-pbe0).
"""
from __future__ import annotations

import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from ase.io import read
from matplotlib.colors import BoundaryNorm

from figures.config import DPI, OUTPUT_DIR, TRAJ_PBE, TRAJ_PBE0
from figures.utils.plotting import AXIS_LABEL_FONTSIZE, TICK_FONTSIZE, TITLE_FONTSIZE, save_fig

FIG_W_MM = 180
FIG_H_MM = 118
TILE_TEXT_FONTSIZE = 5
SELECTION_WINDOW_EV_PER_ATOM = 0.15


def stoichiometry_stats(path: str) -> dict[tuple[int, int], dict]:
    """{(n_Ir, n_O): {"n": count, "span": max (E - E_min) / N_atoms}} over the structures of one traj."""
    energies: dict[tuple[int, int], list[float]] = defaultdict(list)
    for atoms in read(path, index=":"):
        s = atoms.get_chemical_symbols()
        energies[(s.count("Ir"), s.count("O"))].append(atoms.get_potential_energy())
    stats = {}
    for (y, x), es in energies.items():
        e = np.asarray(es)
        stats[(y, x)] = {"n": len(e), "span": float((e.max() - e.min()) / (y + x))}
    return stats


def _grid(stats: dict, key: str, ys: list[int], xs: list[int]) -> np.ndarray:
    g = np.full((len(ys), len(xs)), np.nan)
    for (y, x), v in stats.items():
        g[ys.index(y), xs.index(x)] = v[key]
    return np.ma.masked_invalid(g)


def _tile_panel(ax, grid, ys, xs, cmap, norm, fmt, title, cbar_label, fig, ticks=None):
    mesh = ax.pcolormesh(np.arange(len(xs) + 1), np.arange(len(ys) + 1), grid, cmap=cmap, norm=norm,
                         shading="flat", edgecolors="white", linewidth=0.4)
    for i in range(len(ys)):
        for j in range(len(xs)):
            v = grid[i, j]
            if np.ma.is_masked(v):
                continue
            r, g, b, _ = cmap(norm(v))
            ax.text(j + 0.5, i + 0.5, fmt(v), ha="center", va="center", fontsize=TILE_TEXT_FONTSIZE,
                    color="black" if 0.299 * r + 0.587 * g + 0.114 * b > 0.5 else "white")
    ax.set_xticks(np.arange(len(xs)) + 0.5)
    ax.set_xticklabels(xs)
    ax.set_yticks(np.arange(len(ys)) + 0.5)
    ax.set_yticklabels([f"Ir$_{{{y}}}$" for y in ys])
    ax.invert_yaxis()
    ax.set_xlim(0, len(xs))
    ax.set_xlabel("$n_\\mathrm{O}$", fontsize=AXIS_LABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, fontsize=TITLE_FONTSIZE, loc="left")
    cbar = fig.colorbar(mesh, ax=ax, ticks=ticks, fraction=0.025, pad=0.012)
    cbar.set_label(cbar_label, fontsize=AXIS_LABEL_FONTSIZE)
    cbar.ax.tick_params(labelsize=TICK_FONTSIZE)
    return mesh


def build(label: str, path: str) -> str:
    stats = stoichiometry_stats(path)
    ys = sorted({y for y, _ in stats})
    xs = list(range(max(x for _, x in stats) + 1))
    counts = _grid(stats, "n", ys, xs)
    spans = _grid(stats, "span", ys, xs)

    fig, (ax_n, ax_e) = plt.subplots(2, 1, figsize=(FIG_W_MM / 25.4, FIG_H_MM / 25.4), dpi=DPI)
    n_max = int(counts.max())
    cmap_n = plt.get_cmap("viridis", n_max).copy()
    norm_n = BoundaryNorm(np.arange(0.5, n_max + 1.5), ncolors=cmap_n.N)
    step = max(1, int(np.ceil(n_max / 10)))
    _tile_panel(ax_n, counts, ys, xs, cmap_n, norm_n, lambda v: f"{int(v)}",
                f"(a) {label}: structures per stoichiometry ({int(counts.sum())} in total)",
                "Number of structures", fig, ticks=np.arange(1, n_max + 1, step))
    cmap_e = plt.get_cmap("magma_r").copy()
    vmax = max(SELECTION_WINDOW_EV_PER_ATOM, float(spans.max()))
    norm_e = plt.Normalize(0.0, vmax)
    _tile_panel(ax_e, spans, ys, xs, cmap_e, norm_e, lambda v: f"{v:.2f}",
                f"(b) {label}: energy range above the lowest structure, per atom",
                "max $(E-E_\\mathrm{min})/N$ (eV/atom)", fig)
    fig.tight_layout(h_pad=1.2)
    out = os.path.join(OUTPUT_DIR, f"supp_stoichiometry_tilemap_{label}.svg")
    save_fig(fig, out, dpi=DPI)
    return out


def main() -> None:
    for label, path in (("PBE", TRAJ_PBE), ("PBE0", TRAJ_PBE0)):
        build(label, path)


if __name__ == "__main__":
    main()
