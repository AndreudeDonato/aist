"""Thermodynamic conditions: temperature and one pressure or Δμ per reservoir, any of them an axis.

Every array-valued quantity becomes a grid axis and every scalar stays fixed. The default axis order is the
reservoirs (in the order they were given) and then T, which for one reservoir is the ``(p, T)`` layout of the
``compute_gc`` tensors.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Grid:
    names: tuple[str, ...]          # axis names, "T" or a reservoir name
    values: tuple[np.ndarray, ...]  # axis values
    shape: tuple[int, ...]
    T: np.ndarray                   # temperature on the full grid (K)
    T_values: np.ndarray            # 1D temperatures along the T axis, or a 0-d array when T is fixed
    T_axis: int | None
    modes: dict                     # reservoir name -> "p" (Pa) or "dmu" (eV, relative to its DFT energy)
    full: dict                      # reservoir name -> its pressure or Δμ on the full grid

    def along_T(self, values):
        """Reshape an array computed on ``T_values`` so that it broadcasts against the grid."""
        values = np.asarray(values)
        if self.T_axis is None or values.ndim == 0:
            return values
        shape = [1] * len(self.shape)
        shape[self.T_axis] = len(self.T_values)
        return values.reshape(shape)

    def index(self, **coordinates) -> tuple:
        """Nearest grid index for the given axis values (log scale for pressures, linear otherwise)."""
        unknown = set(coordinates) - set(self.names)
        if unknown:
            raise KeyError(f"{sorted(unknown)} are not axes of this grid (axes: {list(self.names)})")
        idx = []
        for name, values in zip(self.names, self.values):
            if name not in coordinates:
                raise KeyError(f"Give a value for every axis; missing {name!r}")
            target = coordinates[name]
            if name != "T" and self.modes.get(name) == "p":
                idx.append(int(np.argmin(np.abs(np.log(values) - np.log(target)))))
            else:
                idx.append(int(np.argmin(np.abs(values - target))))
        return tuple(idx)


def build_grid(T, p=None, dmu=None, *, reservoir_names, axes=None) -> Grid:
    """Grid of conditions: ``T`` (K) and, for every reservoir, exactly one of ``p[name]`` (Pa) or ``dmu[name]`` (eV)."""
    p, dmu = dict(p or {}), dict(dmu or {})
    unknown = (set(p) | set(dmu)) - set(reservoir_names)
    if unknown:
        raise KeyError(f"No reservoir named {sorted(unknown)} (reservoirs: {list(reservoir_names)})")
    entries = []  # (name, mode, value)
    for name in reservoir_names:
        if (name in p) == (name in dmu):
            raise ValueError(f"Reservoir {name!r}: give exactly one of p[{name!r}] (Pa) or dmu[{name!r}] (eV)")
        mode = "p" if name in p else "dmu"
        entries.append((name, mode, np.asarray(p[name] if mode == "p" else dmu[name], dtype=float)))
    entries.append(("T", "T", np.asarray(T, dtype=float)))
    for name, mode, value in entries:
        if value.ndim > 1:
            raise ValueError(f"{name}: give a scalar or a 1D array, got shape {value.shape}")
        if mode in ("T", "p") and np.any(value <= 0):
            raise ValueError(f"{name}: temperatures and pressures must be positive")

    arrays = [name for name, _, value in entries if value.ndim == 1]
    if axes is None:
        axes = arrays
    axes = list(axes)
    if sorted(axes) != sorted(arrays):
        raise ValueError(f"axes={axes} must list exactly the array-valued quantities {arrays}")
    by_name = {name: (mode, value) for name, mode, value in entries}
    axis_values = tuple(by_name[name][1] for name in axes)
    shape = tuple(len(v) for v in axis_values)
    meshes = dict(zip(axes, np.meshgrid(*axis_values, indexing="ij"))) if axes else {}

    def on_grid(name):
        return meshes[name] if name in meshes else np.full(shape, float(by_name[name][1]))

    T_axis = axes.index("T") if "T" in axes else None
    return Grid(
        names=tuple(axes),
        values=axis_values,
        shape=shape,
        T=on_grid("T"),
        T_values=by_name["T"][1],
        T_axis=T_axis,
        modes={name: by_name[name][0] for name in reservoir_names},
        full={name: on_grid(name) for name in reservoir_names},
    )
