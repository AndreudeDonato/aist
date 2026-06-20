"""Nagai-triangle inertia-ratio scatter (I1/I3 vs I2/I3)."""
from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d.art3d import Line3DCollection


def inertia_ratios(moments: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``moments`` shape ``(n, 3)`` with I1 ≤ I2 ≤ I3 → (I1/I3, I2/I3)."""
    I1, I2, I3 = moments[:, 0], moments[:, 1], moments[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        x = I1 / I3
        y = I2 / I3
    return x, y


NAGAI_R23_MIN = 0.5  # λ₂/λ₃ floor for physical inertia tensors (see methodology doc §2.5)

# Achievable-region corners in raw (I₁/I₃, I₂/I₃) ratios
_NAGAI_CORNER_RAW = (
    (1.0, 1.0, "S"),  # spherical
    (0.0, 1.0, "L"),  # linear
    (0.5, NAGAI_R23_MIN, "P"),  # plane
)
NAGAI_SHAPE_CENTER = (
    float(np.mean([p[0] for p in _NAGAI_CORNER_RAW])),
    float(np.mean([p[1] for p in _NAGAI_CORNER_RAW])),
)
# Symmetric axis padding around ``NAGAI_SHAPE_CENTER`` (legacy framing)
NAGAI_VIEW_HALF_SPAN = 0.58
# Relative z-axis length in ``set_box_aspect`` (extent visual separation vs x/y)
NAGAI_EXTENT_BOX_SCALE = 0.90
NAGAI_XY_PAD = {"left": 0.0, "right": 0.03, "bottom": 0.02, "top": 0.03}

_NAGAI_CORNER_LABELS = _NAGAI_CORNER_RAW
# Label positions: vertices of a larger triangle (radial push from centroid through each corner)
NAGAI_LABEL_OUTER_SCALE = 1.14
_CORNER_TEXT_KW = {"fontsize": 8, "fontweight": "bold", "color": "black"}
_EXTENT_LABEL_KW = {"fontsize": 8, "fontweight": "bold", "color": "black"}
NAGAI_EXTENT_LABEL_CORNER = "S"  # pillar that carries the extent-axis ``E`` label


def _nagai_corner_xy(corner: str) -> tuple[float, float]:
    for x, y, label in _NAGAI_CORNER_LABELS:
        if label == corner:
            return x, y
    raise ValueError(f"unknown Nagai corner {corner!r}")


def _nagai_outer_label_xy(corner: str, *, scale: float = NAGAI_LABEL_OUTER_SCALE) -> tuple[float, float]:
    """Inertia-plane label position outside the triangle along centroid → vertex."""
    cx, cy = NAGAI_SHAPE_CENTER
    x, y = _nagai_corner_xy(corner)
    return cx + scale * (x - cx), cy + scale * (y - cy)


def _nagai_extent_label_3d(ax, z_lim: tuple[float, float], *, corner: str = NAGAI_EXTENT_LABEL_CORNER) -> None:
    """Bold ``E`` above the top of one corner pillar (extent axis)."""
    z0, z1 = float(z_lim[0]), float(z_lim[1])
    z_span = max(z1 - z0, 1e-9)
    x, y = _nagai_outer_label_xy(corner)
    ax.text(
        x, y, z1 + 0.04 * z_span, "E",
        ha="center", va="bottom", zorder=_NAGAI_ZORDER_LABEL + 1, **_EXTENT_LABEL_KW,
    )


def _nagai_floor_edge_segments() -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """S–L–P triangle edges on the base plane."""
    return [
        ((0.5, 0.5), (1.0, 1.0)),         # P → S
        ((0.0, 1.0), (1.0, 1.0)),         # L → S
        ((0.5, 0.5), (0.0, 1.0)),         # P → L
    ]


def _nagai_floor_segments_3d(z0: float) -> list:
    return [
        [(x0, y0, z0), (x1, y1, z0)]
        for (x0, y0), (x1, y1) in _nagai_floor_edge_segments()
    ]


def _nagai_tight_xy_limits() -> tuple[tuple[float, float], tuple[float, float]]:
    """Tight axis limits on triangle corners and outer label vertices."""
    xs: list[float] = []
    ys: list[float] = []
    for x, y, label in _NAGAI_CORNER_LABELS:
        xs.append(x)
        ys.append(y)
        lx, ly = _nagai_outer_label_xy(label)
        xs.append(lx)
        ys.append(ly)
    pad = NAGAI_XY_PAD
    return (
        (min(xs) - pad["left"], max(xs) + pad["right"]),
        (min(ys) - pad["bottom"], max(ys) + pad["top"]),
    )


def _nagai_view_xy_limits() -> tuple[tuple[float, float], tuple[float, float]]:
    """Axis limits with the Nagai triangle centroid at the panel centre."""
    cx, cy = NAGAI_SHAPE_CENTER
    span = NAGAI_VIEW_HALF_SPAN
    return (cx - span, cx + span), (cy - span, cy + span)


def _nagai_corner_text_2d(ax: Axes, zorder: int = 4) -> None:
    """Bold S / L / P labels outside the triangle along centroid → vertex."""
    for _, _, label in _NAGAI_CORNER_LABELS:
        x, y = _nagai_outer_label_xy(label)
        ax.text(
            x, y, label, ha="center", va="center", zorder=zorder,
            **_CORNER_TEXT_KW,
        )


_NAGAI_GUIDE_KW = {"color": "black", "linewidth": 0.9, "linestyle": "-", "alpha": 1.0}
_NAGAI_ZORDER_FLOOR = 0
_NAGAI_ZORDER_DATA = 2
NAGAI_ZORDER_DATA = _NAGAI_ZORDER_DATA
_NAGAI_ZORDER_PILLAR = _NAGAI_ZORDER_DATA
_NAGAI_ZORDER_MARKER = 4
_NAGAI_ZORDER_LABEL = 5


def _add_line3d_segments(
    ax,
    segments: list,
    *,
    sort_z: float,
    zorder: int,
) -> None:
    """Add black 3D guide line segments as a depth-sorted collection."""
    lc = Line3DCollection(
        segments,
        colors=_NAGAI_GUIDE_KW["color"],
        linewidths=_NAGAI_GUIDE_KW["linewidth"],
        alpha=_NAGAI_GUIDE_KW["alpha"],
    )
    lc.set_zorder(zorder)
    if hasattr(lc, "set_sort_zpos"):
        lc.set_sort_zpos(sort_z)
    ax.add_collection(lc)


def _draw_nagai_floor_3d(ax, z0: float) -> None:
    """Achievable Nagai band on the bottom extent plane."""
    _add_line3d_segments(
        ax, _nagai_floor_segments_3d(z0), sort_z=z0, zorder=_NAGAI_ZORDER_FLOOR,
    )


def _draw_nagai_corner_pillars_3d(ax, z0: float, z1: float, *, n_seg: int = 28) -> None:
    """Vertical corner guides; one collection per segment for depth sorting vs scatter."""
    z_vals = np.linspace(z0, z1, n_seg + 1)
    for x, y, _ in _NAGAI_CORNER_LABELS:
        for i in range(n_seg):
            z_a, z_b = float(z_vals[i]), float(z_vals[i + 1])
            _add_line3d_segments(
                ax,
                [[(x, y, z_a), (x, y, z_b)]],
                sort_z=z_a,
                zorder=_NAGAI_ZORDER_PILLAR,
            )


def _draw_nagai_corner_markers_3d(ax, z0: float, *, marker_scale: float) -> None:
    lim_s = 20.0 * marker_scale
    for x, y, label in _NAGAI_CORNER_LABELS:
        fc = "white" if label == "P" else "black"
        ec = "black" if label == "P" else "none"
        sc = ax.scatter(
            [x], [y], [z0], s=lim_s, c=fc, edgecolors=ec,
            linewidths=0.8 if label == "P" else 0.0,
            depthshade=False, zorder=_NAGAI_ZORDER_MARKER,
        )
        sc.set_sort_zpos(z0)


def overlay_nagai_corner_guides_3d(
    ax,
    z_lim: tuple[float, float],
    *,
    marker_scale: float = 0.10,
    pillars: bool = True,
    corner_markers: bool = True,
    labels: bool = True,
    extent_label: bool = False,
    extent_label_corner: str = NAGAI_EXTENT_LABEL_CORNER,
) -> None:
    """Corner pillars + base markers/labels; call after data scatter for depth sorting."""
    z0, z1 = float(z_lim[0]), float(z_lim[1])
    if pillars:
        _draw_nagai_corner_pillars_3d(ax, z0, z1)
    if corner_markers:
        _draw_nagai_corner_markers_3d(ax, z0, marker_scale=marker_scale)
    if labels:
        _nagai_corner_text_3d(ax, z0)
    if extent_label:
        _nagai_extent_label_3d(ax, z_lim, corner=extent_label_corner)


def _nagai_corner_text_3d(ax, z0: float) -> None:
    """Bold S / L / P labels outside the triangle along centroid → vertex."""
    for _, _, label in _NAGAI_CORNER_LABELS:
        x, y = _nagai_outer_label_xy(label)
        ax.text(
            x, y, z0, label,
            ha="center", va="center", zorder=_NAGAI_ZORDER_LABEL, **_CORNER_TEXT_KW,
        )


def _draw_nagai_guides_2d(ax: Axes, *, marker_scale: float = 0.10, zorder: int = 0) -> float:
    """Achievable Nagai-band guides, limit markers, and corner labels."""
    for (x0, y0), (x1, y1) in _nagai_floor_edge_segments():
        ax.plot(
            [x0, x1], [y0, y1], color="black", linewidth=0.9,
            linestyle="-", alpha=1.0, zorder=zorder,
        )

    lim_s = 20.0 * marker_scale
    for x, y, label in _NAGAI_CORNER_LABELS:
        fc = "white" if label == "P" else "black"
        ec = "black" if label == "P" else "none"
        ax.scatter(
            [x], [y], s=lim_s, c=fc, edgecolors=ec,
            linewidths=0.8 if label == "P" else 0.0, zorder=zorder + 3,
        )
    _nagai_corner_text_2d(ax, zorder=zorder + 4)
    return lim_s


def style_inertia_triangle(ax: Axes, *, marker_scale: float = 0.10) -> float:
    """Grey guides, limit markers, labels — match ``plot_inertia_comparison.py``."""
    xlim, ylim = _nagai_view_xy_limits()
    ax.set_xlabel(r"$I_x/I_z$")
    ax.set_ylabel(r"$I_y/I_z$")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xticks(np.linspace(xlim[0], xlim[1], 5))
    ax.set_yticks(np.linspace(ylim[0], ylim[1], 5))
    ax.set_aspect("equal", "box")
    return _draw_nagai_guides_2d(ax, marker_scale=marker_scale)


def _hide_3d_axes_ui(ax) -> None:
    """Hide ticks, labels, grid, and box panes (data + collections remain)."""
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    ax.grid(False)
    ax.xaxis.pane.set_visible(False)
    ax.yaxis.pane.set_visible(False)
    ax.zaxis.pane.set_visible(False)
    ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))


def style_inertia_shape_extent_3d(
    ax,
    z_lim: tuple[float, float],
    *,
    marker_scale: float = 0.10,
    elev: float = 28.0,
    azim: float = -67.0,
    bare: bool = False,
) -> None:
    """3D Nagai shape ratios + mass-normalized extent on the z-axis."""
    z0 = float(z_lim[0])
    xlim, ylim = _nagai_tight_xy_limits()
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(z_lim)
    ax.view_init(elev=elev, azim=azim)
    ax.computed_zorder = True

    if bare:
        _hide_3d_axes_ui(ax)
    else:
        ax.set_xlabel(r"$I_x/I_z$", fontsize=7, labelpad=0)
        ax.set_ylabel(r"$I_y/I_z$", fontsize=7, labelpad=0)
        ax.set_zlabel(r"$\log R_g - \frac{1}{3}\log M$", fontsize=7, labelpad=2)
        ax.set_xticks(np.linspace(xlim[0], xlim[1], 3))
        ax.set_yticks(np.linspace(ylim[0], ylim[1], 3))
        ax.tick_params(axis="both", labelsize=6)
        ax.tick_params(axis="z", labelsize=6)
        for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            pane.fill = False
            pane.set_edgecolor("0.85")
        ax.grid(True, linewidth=0.4, alpha=0.35)

    _draw_nagai_floor_3d(ax, z0)

    z_span = max(float(z_lim[1]) - z0, 1e-9)
    try:
        x_span = xlim[1] - xlim[0]
        y_span = ylim[1] - ylim[0]
        ax.set_box_aspect((x_span, y_span, NAGAI_EXTENT_BOX_SCALE * z_span))
    except AttributeError:
        pass


def scatter_discrete(
    ax: Axes,
    x: np.ndarray,
    y: np.ndarray,
    labels: np.ndarray,
    colors: list,
    *,
    sizes: np.ndarray | float = 12.0,
    alpha: float = 0.55,
    edgecolors: str = "none",
    linewidths: float = 0.0,
    zorder: int = 2,
) -> None:
    """Scatter each integer label group with its discrete colour."""
    labels = np.asarray(labels, dtype=int)
    for c in np.unique(labels):
        mask = labels == c
        if not mask.any():
            continue
        col = colors[int(c) % len(colors)]
        sz = sizes[mask] if isinstance(sizes, np.ndarray) else sizes
        ax.scatter(
            x[mask], y[mask], c=[col], s=sz, alpha=alpha,
            edgecolors=edgecolors, linewidths=linewidths, zorder=zorder,
        )


def scatter_discrete_3d(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    labels: np.ndarray,
    colors: list,
    *,
    sizes: np.ndarray | float = 12.0,
    alpha: float = 0.55,
    edgecolors: str = "none",
    linewidths: float = 0.0,
) -> None:
    """3D scatter coloured by discrete integer labels."""
    labels = np.asarray(labels, dtype=int)
    for c in np.unique(labels):
        mask = labels == c
        if not mask.any():
            continue
        col = colors[int(c) % len(colors)]
        sz = sizes[mask] if isinstance(sizes, np.ndarray) else sizes
        ax.scatter(
            x[mask], y[mask], z[mask], c=[col], s=sz, alpha=alpha,
            edgecolors=edgecolors, linewidths=linewidths, depthshade=False,
            zorder=_NAGAI_ZORDER_DATA,
        )
