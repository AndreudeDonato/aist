#!/usr/bin/env python3
"""Visualize a trajectory file with motif colours (atom radii unchanged).

Run as::

    python3 visualize_motifs.py file.traj motif_atoms_PBE.json
    python3 visualize_motifs.py file.traj motif_atoms_PBE0.json
    python3 visualize_motifs.py file.traj motif_atoms_pooled.json

Opens an interactive matplotlib 3D viewer (mouse-drag rotation, scroll zoom),
with motif-coloured atoms, covalent-radius sized markers, and on-screen widgets
to toggle bonds / motif labels and step through frames.

CLI flags
---------
``--svg``
    Skip the interactive window and dump the multi-frame grid as
    ``<traj>_motifs.svg`` instead.
``--svg-out PATH``
    Override the SVG location (file path for one traj, directory for many).
``--legend [PATH]``
    Also emit a motif-colour legend SVG (defaults to ``motif_legend.svg``).

Interactive controls
--------------------
* Drag with the left mouse button : rotate.
* Scroll wheel : zoom.
* ``L`` (or the "Labels" button) : toggle motif labels.
* ``B`` (or the "Bonds" button)  : toggle bond rendering.
* Left/right arrows or the slider : step through frames.

Everything else (grid columns for the SVG mode, max frames in the grid,
sphere/label scaling, bond cutoff factor, transparency, etc.) is a module constant near
the top of this file - edit there, not on the command line.
"""

from __future__ import annotations

# ======================================================================
# Display defaults - tweak here instead of via CLI flags.
# ======================================================================
# Interactive 3D viewer
DEFAULT_RADIUS_SCALE = 1.1          # multiplies ASE covalent radii (visual only)
DEFAULT_SPHERE_SIZE_FACTOR = 50     # scatter size = (radius * factor)**2
DEFAULT_BONDS_SPHERE_SCALE = 0.5    # multiply sphere size by this when bonds are shown
DEFAULT_TRANSPARENT = False         # if True, atoms are drawn semi-transparent
DEFAULT_SPHERE_ALPHA = 0.55         # scatter alpha when DEFAULT_TRANSPARENT is True
DEFAULT_BOND_TOLERANCE = 1.00       # bond cutoff = (r_i + r_j) * tolerance
DEFAULT_BOND_COLOR = "black"
DEFAULT_BOND_LINEWIDTH = 2.4
DEFAULT_BOND_ALPHA = 1.0
DEFAULT_SHOW_BONDS = True           # initial bond visibility
DEFAULT_SHOW_LABELS = False         # initial motif-label visibility
DEFAULT_LABEL_FONTSIZE = 6
DEFAULT_VIEW_ELEV = 22              # initial camera elevation (deg)
DEFAULT_VIEW_AZIM = -55             # initial camera azimuth (deg)
DEFAULT_FIGSIZE = (8.5, 7.5)        # interactive viewer figure size (inches)

# Static --svg grid (only used when --svg / --svg-out is given)
DEFAULT_COLS = 6                    # subplot columns in the grid
DEFAULT_MAX_FRAMES = 24             # max frames rendered in the grid
DEFAULT_GRID_ROTATION = "0x,0y,0z"  # ASE rotation string for the 2D projection
DEFAULT_LEGEND_PATH = "motif_legend.svg"

import argparse
import json
import os
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from ase.data import covalent_radii
from ase.io import read
from ase.visualize.plot import plot_atoms
from matplotlib.widgets import Button, Slider
from mpl_toolkits.mplot3d.art3d import Line3DCollection  # noqa: F401 (registered side-effect)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hex_to_rgb(hex_string: str):
    h = hex_string.lstrip("#")
    return tuple(int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4))


def _load_motif_atoms(path: str):
    if not os.path.exists(path):
        raise SystemExit(
            f"[visualize_motifs] motif dict not found: {path!r}\n"
            "Run the clustering pipeline first, then pass the JSON path."
        )
    with open(path) as f:
        data = json.load(f)
    if "frames" not in data:
        raise SystemExit(
            f"[visualize_motifs] {path!r} is not a per-dataset motif dict "
            "(missing 'frames' key). Re-run motif_clustering.py to regenerate."
        )
    return data


def _resolve_colors(atoms, motif_atoms, frame_fallback_idx):
    """Return (rgb_list, summary_dict, motif_labels) for one Atoms object."""
    frames_dict = motif_atoms.get("frames", {})
    frame_idx = atoms.info.get("frame_index")
    if frame_idx is None:
        frame_idx = frame_fallback_idx
    motif_labels = frames_dict.get(str(int(frame_idx)))
    symbols = atoms.get_chemical_symbols()
    summary = {
        "frame_index": int(frame_idx),
        "n_atoms": len(symbols),
        "motif_lookup": "ok",
        "n_motif_coloured": 0,
        "n_fallback_species": 0,
        "n_unknown": 0,
    }
    if motif_labels is None:
        summary["motif_lookup"] = "missing-in-dict"
        motif_labels = ["_"] * len(symbols)
    elif len(motif_labels) != len(symbols):
        summary["motif_lookup"] = (
            f"length-mismatch (json={len(motif_labels)}, atoms={len(symbols)})"
        )
        motif_labels = ["_"] * len(symbols)

    colors_palette = motif_atoms.get("colors", {})
    fallback = motif_atoms.get("fallback_colors", {})
    rgbs = []
    for sym, mot in zip(symbols, motif_labels):
        if mot in colors_palette:
            rgbs.append(_hex_to_rgb(colors_palette[mot]))
            summary["n_motif_coloured"] += 1
        elif sym in fallback:
            rgbs.append(_hex_to_rgb(fallback[sym]))
            summary["n_fallback_species"] += 1
        else:
            rgbs.append(_hex_to_rgb(fallback.get("unassigned", "#888888")))
            summary["n_unknown"] += 1
    return rgbs, summary, motif_labels


def _print_summary(traj_path, summaries):
    if not summaries:
        print(f"[visualize_motifs] {traj_path}: empty trajectory")
        return
    total = sum(s["n_atoms"] for s in summaries)
    motif = sum(s["n_motif_coloured"] for s in summaries)
    fallback = sum(s["n_fallback_species"] for s in summaries)
    unknown = sum(s["n_unknown"] for s in summaries)
    bad_lookups = [s for s in summaries if s["motif_lookup"] != "ok"]
    pct = 100.0 * motif / max(total, 1)
    print(
        f"[visualize_motifs] {traj_path}: "
        f"{len(summaries)} frames, {motif}/{total} atoms ({pct:.1f}%) motif-coloured, "
        f"{fallback} species-fallback, {unknown} unknown"
    )
    if bad_lookups:
        print(f"  warning: {len(bad_lookups)} frame(s) without a valid motif lookup:")
        for s in bad_lookups[:5]:
            print(f"     frame_index={s['frame_index']} reason={s['motif_lookup']}")
        if len(bad_lookups) > 5:
            print(f"     ... ({len(bad_lookups) - 5} more)")


# ---------------------------------------------------------------------------
# Interactive 3D viewer (the default mode)
# ---------------------------------------------------------------------------


class _MotifViewer:
    """Single-window interactive 3D viewer with frame nav + label/bond toggles."""

    def __init__(self, traj_path, atoms_list, motif_atoms):
        self.traj_path = traj_path
        self.atoms_list = atoms_list
        self.motif_atoms = motif_atoms
        self.frame_idx = 0
        self.show_labels = DEFAULT_SHOW_LABELS
        self.show_bonds = DEFAULT_SHOW_BONDS
        self._scatter = None
        self._base_sizes = None
        self._ref_span = None
        self._build_figure()
        self._draw_frame(reset_view=True)

    # -- layout -----------------------------------------------------------

    def _build_figure(self):
        self.fig = plt.figure(figsize=DEFAULT_FIGSIZE)
        self.fig.canvas.manager.set_window_title(
            f"motifs: {os.path.basename(self.traj_path)}"
        )
        # 3D axes (rotation is mouse-driven by default)
        self.ax = self.fig.add_axes([0.04, 0.20, 0.92, 0.76], projection="3d")
        self.ax.set_proj_type("ortho")

        # Frame slider (only if multi-frame)
        multi = len(self.atoms_list) > 1
        if multi:
            slider_ax = self.fig.add_axes([0.10, 0.10, 0.55, 0.03])
            self.slider = Slider(
                slider_ax,
                "frame",
                0,
                len(self.atoms_list) - 1,
                valinit=0,
                valstep=1,
                valfmt="%d",
            )
            self.slider.on_changed(self._on_slider)
        else:
            self.slider = None

        # Prev/Next/Labels/Bonds buttons
        btn_w, btn_h = 0.07, 0.04
        prev_ax = self.fig.add_axes([0.10, 0.03, btn_w, btn_h])
        next_ax = self.fig.add_axes([0.18, 0.03, btn_w, btn_h])
        labels_ax = self.fig.add_axes([0.36, 0.03, 0.11, btn_h])
        bonds_ax = self.fig.add_axes([0.48, 0.03, 0.11, btn_h])
        self.btn_prev = Button(prev_ax, "< prev")
        self.btn_next = Button(next_ax, "next >")
        self.btn_labels = Button(labels_ax, "labels")
        self.btn_bonds = Button(bonds_ax, "bonds")
        self.btn_prev.on_clicked(self._on_prev)
        self.btn_next.on_clicked(self._on_next)
        self.btn_labels.on_clicked(self._on_toggle_labels)
        self.btn_bonds.on_clicked(self._on_toggle_bonds)
        if not multi:
            for b in (self.btn_prev, self.btn_next):
                b.ax.set_visible(False)

        # Help text and keyboard hooks.
        self.fig.text(
            0.62, 0.045,
            "drag = rotate | scroll = zoom | L = labels | B = bonds | arrows = frame",
            fontsize=7, color="0.4",
        )
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig.canvas.mpl_connect("draw_event", self._on_draw_resize)

    def _sphere_size_factor(self):
        factor = DEFAULT_SPHERE_SIZE_FACTOR
        if self.show_bonds:
            factor *= DEFAULT_BONDS_SPHERE_SCALE
        return factor

    def _view_span(self):
        """Largest axis span; shrinks on zoom-in and grows on zoom-out."""
        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()
        z0, z1 = self.ax.get_zlim()
        return max(x1 - x0, y1 - y0, z1 - z0)

    def _update_sphere_sizes(self):
        """Scale scatter markers with the current zoom (axis limits)."""
        if self._scatter is None or self._base_sizes is None or self._ref_span is None:
            return
        span = self._view_span()
        if span <= 0.0 or self._ref_span <= 0.0:
            return
        zoom = self._ref_span / span
        new_sizes = self._base_sizes * (zoom ** 2)
        current = self._scatter.get_sizes()
        if len(current) and np.allclose(
            np.sort(current), np.sort(new_sizes), rtol=1e-4, atol=0.5
        ):
            return
        self._scatter.set_sizes(new_sizes)

    def _on_draw_resize(self, event):
        if event.canvas is not self.fig.canvas:
            return
        self._update_sphere_sizes()

    def _style_view_axes(self):
        """Hide the 3D bounding box / pane edges; keep rotation working."""
        ax = self.ax
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_zlabel("")
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
            axis.pane.set_edgecolor((1.0, 1.0, 1.0, 0.0))
            axis.line.set_color((1.0, 1.0, 1.0, 0.0))
            axis.set_tick_params(
                bottom=False, labelbottom=False,
                top=False, labeltop=False,
                left=False, labelleft=False,
                right=False, labelright=False,
            )
        if hasattr(ax, "_axis3don"):
            ax._axis3don = False

    def _draw_bonds(self, pos, radii):
        """Draw bonds beneath atoms (black, thick); returns collection or None."""
        segments = []
        n = len(pos)
        for i in range(n):
            for j in range(i + 1, n):
                cutoff = (radii[i] + radii[j]) * DEFAULT_BOND_TOLERANCE
                if float(((pos[i] - pos[j]) ** 2).sum()) < cutoff * cutoff:
                    segments.append([pos[i], pos[j]])
        if not segments:
            return None
        lc = Line3DCollection(
            segments,
            colors=DEFAULT_BOND_COLOR,
            linewidths=DEFAULT_BOND_LINEWIDTH,
            alpha=DEFAULT_BOND_ALPHA,
            zorder=0,
        )
        self.ax.add_collection3d(lc)
        lc.set_zorder(0)
        return lc

    # -- per-frame rendering ---------------------------------------------

    def _draw_frame(self, reset_view=False):
        # Preserve the current camera orientation between redraws.
        if reset_view:
            elev, azim = DEFAULT_VIEW_ELEV, DEFAULT_VIEW_AZIM
        else:
            try:
                elev, azim = self.ax.elev, self.ax.azim
            except Exception:
                elev, azim = DEFAULT_VIEW_ELEV, DEFAULT_VIEW_AZIM

        self.ax.cla()
        atoms = self.atoms_list[self.frame_idx]
        pos = atoms.positions
        numbers = atoms.numbers
        radii = np.array(
            [DEFAULT_RADIUS_SCALE * covalent_radii[z] for z in numbers]
        )
        colors, summary, motif_labels = _resolve_colors(
            atoms, self.motif_atoms, self.frame_idx
        )

        # Equal box aspect on the cluster's bounding cube + a small margin.
        if len(pos) > 0:
            ranges = pos.max(axis=0) - pos.min(axis=0)
            mid = (pos.max(axis=0) + pos.min(axis=0)) / 2.0
            half = float(ranges.max()) / 2.0 + 1.5
        else:
            mid = np.zeros(3)
            half = 5.0
        self.ax.set_xlim(mid[0] - half, mid[0] + half)
        self.ax.set_ylim(mid[1] - half, mid[1] + half)
        self.ax.set_zlim(mid[2] - half, mid[2] + half)
        try:
            self.ax.set_box_aspect((1, 1, 1))
        except Exception:
            pass

        # Bonds first (lowest layer); atoms drawn on top and hide them unless
        # sphere alpha is reduced.
        self._bond_collection = None
        if self.show_bonds:
            self._bond_collection = self._draw_bonds(pos, radii)

        sizes = (radii * self._sphere_size_factor()) ** 2
        alpha = DEFAULT_SPHERE_ALPHA if DEFAULT_TRANSPARENT else 1.0
        self._base_sizes = sizes.copy()
        self._scatter = self.ax.scatter(
            pos[:, 0], pos[:, 1], pos[:, 2],
            c=colors, s=sizes,
            edgecolors="black", linewidths=0.35,
            depthshade=False, alpha=alpha,
            zorder=10,
        )
        self._scatter.set_zorder(10)

        # Optional motif labels.
        if self.show_labels:
            for p, m in zip(pos, motif_labels):
                if m == "_":
                    continue
                self.ax.text(
                    p[0], p[1], p[2], m,
                    fontsize=DEFAULT_LABEL_FONTSIZE,
                    ha="center", va="center",
                    color="black", zorder=11,
                )

        self._style_view_axes()

        cluster = atoms.info.get("cluster", "?")
        title = (
            f"{os.path.basename(self.traj_path)}   "
            f"frame {summary['frame_index']}  (idx {self.frame_idx + 1}/"
            f"{len(self.atoms_list)})   C{cluster}   "
            f"dataset={self.motif_atoms.get('label', '?')}"
        )
        if summary["motif_lookup"] != "ok":
            title += "   [no motif lookup]"
        elif summary["n_fallback_species"] or summary["n_unknown"]:
            title += (
                f"   [{summary['n_motif_coloured']}/{summary['n_atoms']} motif]"
            )
        self.ax.set_title(title, fontsize=9)

        self.ax.view_init(elev=elev, azim=azim)
        self._ref_span = self._view_span()
        self._update_sphere_sizes()
        self.fig.canvas.draw_idle()

    # -- event handlers ---------------------------------------------------

    def _on_prev(self, _event=None):
        if len(self.atoms_list) > 1:
            self.frame_idx = (self.frame_idx - 1) % len(self.atoms_list)
            self._sync_slider_silently()
            self._draw_frame()

    def _on_next(self, _event=None):
        if len(self.atoms_list) > 1:
            self.frame_idx = (self.frame_idx + 1) % len(self.atoms_list)
            self._sync_slider_silently()
            self._draw_frame()

    def _on_slider(self, val):
        new_idx = int(val)
        if new_idx != self.frame_idx:
            self.frame_idx = new_idx
            self._draw_frame()

    def _on_toggle_labels(self, _event=None):
        self.show_labels = not self.show_labels
        self._draw_frame()

    def _on_toggle_bonds(self, _event=None):
        self.show_bonds = not self.show_bonds
        self._draw_frame()

    def _on_key(self, event):
        key = (event.key or "").lower()
        if key in ("right", "n", " "):
            self._on_next()
        elif key in ("left", "p"):
            self._on_prev()
        elif key == "l":
            self._on_toggle_labels()
        elif key == "b":
            self._on_toggle_bonds()

    def _sync_slider_silently(self):
        if self.slider is None:
            return
        # Avoid recursive callbacks.
        self.slider.eventson = False
        self.slider.set_val(self.frame_idx)
        self.slider.eventson = True

    def show(self):
        plt.show()


def _interactive_view(traj_path, motif_atoms):
    atoms_list = read(traj_path, ":")
    if not atoms_list:
        print(f"[visualize_motifs] {traj_path}: empty trajectory, skipping")
        return
    # Per-frame summary printout (does no rendering work).
    summaries = []
    for i, atoms in enumerate(atoms_list):
        _, summary, _ = _resolve_colors(atoms, motif_atoms, i)
        summaries.append(summary)
    _print_summary(traj_path, summaries)
    viewer = _MotifViewer(traj_path, atoms_list, motif_atoms)
    viewer.show()


# ---------------------------------------------------------------------------
# Static SVG grid (only used when --svg / --svg-out is given)
# ---------------------------------------------------------------------------


def _render_traj_grid(traj_path, motif_atoms):
    atoms_list = read(traj_path, ":")[:DEFAULT_MAX_FRAMES]
    if not atoms_list:
        return None, []

    n = len(atoms_list)
    ncols = min(DEFAULT_COLS, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(2.2 * ncols, 2.2 * nrows),
        squeeze=False,
        dpi=140,
    )
    label = motif_atoms.get("label", "?")
    fig.suptitle(
        f"{os.path.relpath(traj_path)}  ({label})  - {n} frame(s)",
        fontsize=8, y=0.998,
    )

    summaries = []
    for i, atoms in enumerate(atoms_list):
        ax = axes[i // ncols][i % ncols]
        colors, summary, _ = _resolve_colors(atoms, motif_atoms, i)
        radii = [DEFAULT_RADIUS_SCALE * covalent_radii[a.number] for a in atoms]
        plot_atoms(
            atoms, ax=ax, colors=colors, radii=radii,
            rotation=DEFAULT_GRID_ROTATION, show_unit_cell=0,
        )
        cluster = atoms.info.get("cluster", "?")
        status = ""
        if summary["motif_lookup"] != "ok":
            status = "  [no motif]"
        elif summary["n_fallback_species"] > 0 or summary["n_unknown"] > 0:
            status = f"  [{summary['n_motif_coloured']}/{summary['n_atoms']} motif]"
        ax.set_title(
            f"frame {summary['frame_index']}  C{cluster}{status}",
            fontsize=6,
        )
        ax.set_axis_off()
        summaries.append(summary)

    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].set_axis_off()

    fig.tight_layout(pad=0.6)
    return fig, summaries


def _resolve_svg_output(svg_out, traj_path, multi: bool) -> str:
    if svg_out is None:
        stem, _ = os.path.splitext(traj_path)
        return f"{stem}_motifs.svg"
    if multi:
        os.makedirs(svg_out, exist_ok=True)
        return os.path.join(
            svg_out,
            os.path.splitext(os.path.basename(traj_path))[0] + "_motifs.svg",
        )
    return svg_out


# ---------------------------------------------------------------------------
# Legend
# ---------------------------------------------------------------------------


def _build_legend(motif_atoms, output_path: str):
    colors = motif_atoms.get("colors", {})
    ir_motifs = motif_atoms.get("ir_motifs", [])
    o_motifs = motif_atoms.get("o_motifs", [])
    rows = max(len(ir_motifs), len(o_motifs), 1)
    fig, ax = plt.subplots(figsize=(3.0, 0.35 * rows + 0.6), dpi=200)
    ax.set_axis_off()
    ax.text(0.05, 0.8, "Ir motifs", fontsize=8, fontweight="bold")
    ax.text(0.55, 0.8, "O motifs", fontsize=8, fontweight="bold")
    for i, m in enumerate(ir_motifs):
        ax.scatter(0.05, -i, s=120, c=colors.get(m, "#888888"),
                   edgecolors="black", linewidths=0.4)
        ax.text(0.12, -i, m, va="center", fontsize=7)
    for i, m in enumerate(o_motifs):
        ax.scatter(0.55, -i, s=120, c=colors.get(m, "#888888"),
                   edgecolors="black", linewidths=0.4)
        ax.text(0.62, -i, m, va="center", fontsize=7)
    ax.set_xlim(0, 1)
    ax.set_ylim(-rows, 1)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] legend -> {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Iterable[str] | None = None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("trajectories", nargs="+",
                        help="Trajectory files (.traj) to visualize.")
    parser.add_argument("json",
                        help="motif_atoms_PBE.json, motif_atoms_PBE0.json, "
                             "or motif_atoms_pooled.json")
    parser.add_argument("--svg", action="store_true",
                        help="Skip the interactive viewer and save a "
                             "multi-frame grid SVG instead.")
    parser.add_argument("--svg-out", default=None,
                        help="Explicit SVG output path (file for one traj, "
                             "directory for many).")
    parser.add_argument("--legend", nargs="?", const=DEFAULT_LEGEND_PATH, default=None,
                        help="Also emit a motif-colour legend SVG.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    motif_atoms = _load_motif_atoms(args.json)
    label = motif_atoms.get("label", "?")
    print(f"[visualize_motifs] {args.json}  (dataset={label})")

    if args.legend is not None:
        _build_legend(motif_atoms, args.legend)

    save_svg = bool(args.svg) or args.svg_out is not None
    multi = len(args.trajectories) > 1

    if save_svg:
        # Static grid for publication-ready figures.
        for traj in args.trajectories:
            fig, summaries = _render_traj_grid(traj, motif_atoms)
            _print_summary(traj, summaries)
            if fig is None:
                continue
            out = _resolve_svg_output(args.svg_out, traj, multi)
            fig.savefig(out, bbox_inches="tight")
            plt.close(fig)
            print(f"[save] {traj} -> {out}")
        return

    # Default: open the interactive 3D viewer, one window per traj.
    for traj in args.trajectories:
        _interactive_view(traj, motif_atoms)


if __name__ == "__main__":
    main()
