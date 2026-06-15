"""Figure 1 – PBE and PBE0 stoichiometry-minima tiles side by side.

Layout: 180 × 230 mm total; each dataset panel 88 × 230 mm with a 4 mm blank gap
        (88 + 4 + 88 = 180). No titles or other text.
        PBE and PBE0 share one ``half_span`` (uniform Å scale across both panels).

Also writes supplementary per-dataset 88 × 230 mm SVGs:
``supp_minima_PBE.svg``, ``supp_minima_PBE0.svg``.

Output: ``figures/outputs/fig01_minima.svg``
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from ase.io import read, write
from ase.visualize.plot import plot_atoms

from aist.core.minima import select_stoichiometry_minima
from figures.config import TRAJ_PBE, TRAJ_PBE0, OUTPUT_DIR, CACHE_DIR, MINIMA_TRAJ, DPI
from figures.utils.io import ensure_dirs
from figures.utils.cache import ensure_raw_data

# ---------------------------------------------------------------------------
# Layout (mm)
# ---------------------------------------------------------------------------
PANEL_W_MM = 88
PANEL_H_MM = 230
TOTAL_W_MM = 180
GAP_MM = TOTAL_W_MM - 2 * PANEL_W_MM  # 4 mm between panels
MM = 1 / 25.4

ROTATION = "0x,0y,0z"
OUTLINE_WIDTH = 0.05
TILE_BOX_SCALE = 2.2

DATASETS = [
    {"label": "PBE", "traj": TRAJ_PBE},
    {"label": "PBE0", "traj": TRAJ_PBE0},
]

OUTPUT_PATH = os.path.join(OUTPUT_DIR, "fig01_minima.svg")

plt.rcParams["svg.fonttype"] = "none"


def _rendered_span(atoms, rotation: str, outline_width: float) -> tuple[float, float]:
    fig, ax = plt.subplots(figsize=(2.0, 2.0), dpi=150)
    plot_atoms(atoms, ax=ax, rotation=rotation, radii=None, show_unit_cell=0)
    for col in ax.collections:
        try:
            col.set_linewidths(outline_width)
        except Exception:
            pass
    for patch in ax.patches:
        try:
            patch.set_linewidth(outline_width)
        except Exception:
            pass
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    plt.close(fig)
    return abs(float(x1 - x0)), abs(float(y1 - y0))


def _half_span(selected) -> float:
    half_span = 0.0
    for _, _, atoms in selected:
        sx, sy = _rendered_span(atoms, ROTATION, OUTLINE_WIDTH)
        half_span = max(half_span, 0.5 * max(sx, sy) * 1.05)
    return half_span


def _apply_tile_scale(axes, scale: float) -> None:
    if scale == 1.0:
        return
    for row in axes:
        for ax in row:
            pos = ax.get_position()
            cx = pos.x0 + 0.5 * pos.width
            cy = pos.y0 + 0.5 * pos.height
            nw = pos.width * scale
            nh = pos.height * scale
            ax.set_position([cx - 0.5 * nw, cy - 0.5 * nh, nw, nh])


def build_tile_into(
    fig,
    selected,
    *,
    half_span: float | None = None,
    panel_spec=None,
) -> np.ndarray:
    """Draw stoichiometry tiles (no labels).

    ``panel_spec``: optional ``GridSpec`` cell for the combined figure; if omitted,
    tiles fill the whole ``fig`` (standalone 88 mm panel).
    """
    if half_span is None:
        half_span = _half_span(selected)

    x_unique = sorted({x for x, _, _ in selected})
    y_unique = sorted({y for _, y, _ in selected})
    nrows = len(y_unique)
    ncols = len(x_unique)

    if panel_spec is None:
        axes = fig.subplots(nrows, ncols, squeeze=False)
        fig.subplots_adjust(
            left=0.01, right=0.99, bottom=0.01, top=0.99, wspace=0.0, hspace=0.0,
        )
    else:
        inner = panel_spec.subgridspec(nrows, ncols, wspace=0, hspace=0)
        axes = np.empty((nrows, ncols), dtype=object)
        for i in range(nrows):
            for j in range(ncols):
                ax = fig.add_subplot(inner[i, j])
                axes[i, j] = ax

    for row in axes:
        for ax in row:
            ax.set_axis_off()
            ax.set_facecolor((1.0, 1.0, 1.0, 0.0))
            ax.set_clip_on(False)

    xi = {v: i for i, v in enumerate(x_unique)}
    yi = {v: i for i, v in enumerate(y_unique)}

    for x_val, y_val, atoms in selected:
        ax = axes[yi[y_val]][xi[x_val]]
        plot_atoms(atoms, ax=ax, rotation=ROTATION, radii=None, show_unit_cell=0)
        for col in ax.collections:
            try:
                col.set_linewidths(OUTLINE_WIDTH)
            except Exception:
                pass
        for patch in ax.patches:
            try:
                patch.set_linewidth(OUTLINE_WIDTH)
            except Exception:
                pass
        xc = 0.5 * sum(ax.get_xlim())
        yc = 0.5 * sum(ax.get_ylim())
        ax.set_xlim(xc - half_span, xc + half_span)
        ax.set_ylim(yc - half_span, yc + half_span)
        ax.set_aspect("equal", adjustable="box")

    _apply_tile_scale(axes, TILE_BOX_SCALE)
    return axes


def _save_svg_exact(fig, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, format="svg", dpi=DPI, bbox_inches=None, pad_inches=0)
    plt.close(fig)
    print(f"[save] {path}")


def main() -> None:
    ensure_raw_data()
    ensure_dirs(OUTPUT_DIR, CACHE_DIR)

    selected_by_label: dict[str, list] = {}
    for ds in DATASETS:
        if not os.path.exists(ds["traj"]):
            print(f"[skip] {ds['label']}: trajectory not found at {ds['traj']}")
            continue
        print(f"[fig01] loading {ds['label']} ...")
        structures = list(read(ds["traj"], ":"))
        selected = select_stoichiometry_minima(structures)
        selected_by_label[ds["label"]] = selected

        write(MINIMA_TRAJ[ds["label"]], [a for _, _, a in selected])
        print(f"[fig01] {ds['label']}: {len(selected)} stoichiometries")

    if len(selected_by_label) < 2:
        raise SystemExit("[fig01] Need both PBE and PBE0 trajectories.")

    # One half_span from the largest structure in either dataset → same Å scale in both panels
    combined = [
        entry
        for selected in selected_by_label.values()
        for entry in selected
    ]
    shared_half_span = _half_span(combined)
    print(f"[fig01] shared half_span (PBE + PBE0): {shared_half_span:.4f}")

    for label, selected in selected_by_label.items():
        panel_path = os.path.join(OUTPUT_DIR, f"supp_minima_{label}.svg")
        panel_fig = plt.figure(
            figsize=(PANEL_W_MM * MM, PANEL_H_MM * MM), dpi=DPI,
        )
        build_tile_into(panel_fig, selected, half_span=shared_half_span)
        _save_svg_exact(panel_fig, panel_path)

    # Combined 180 × 230 mm: PBE | 4 mm gap | PBE0
    # GridSpec on the main figure (not SubFigure) so TILE_BOX_SCALE is not clipped.
    fig = plt.figure(figsize=(TOTAL_W_MM * MM, PANEL_H_MM * MM), dpi=DPI)
    outer = fig.add_gridspec(
        1, 3,
        width_ratios=[PANEL_W_MM, GAP_MM, PANEL_W_MM],
        wspace=0,
    )
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    build_tile_into(
        fig, selected_by_label["PBE"],
        half_span=shared_half_span, panel_spec=outer[0],
    )
    build_tile_into(
        fig, selected_by_label["PBE0"],
        half_span=shared_half_span, panel_spec=outer[2],
    )
    _save_svg_exact(fig, OUTPUT_PATH)
    print(f"\n[fig01] {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
