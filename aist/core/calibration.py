"""Free-energy difference between two named rows of a GC tensor at an anchor (T, p)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import RegularGridInterpolator


@dataclass(frozen=True)
class CalibrationAnchor:
    T_K: float
    P_O2_Pa: float


def g_difference_at_anchor(
    g: np.ndarray, ps: np.ndarray, ts: np.ndarray, formulas, minuend: str, subtrahend: str, *,
    anchor: CalibrationAnchor,
) -> float:
    """``G[minuend] − G[subtrahend]`` interpolated at ``anchor`` from a ``(rows × P × T)`` tensor."""
    formulas = [str(f) for f in formulas]

    def _interp(formula: str) -> float:
        interp = RegularGridInterpolator((ps, ts), g[formulas.index(formula)], bounds_error=False, fill_value=None)
        return float(interp([[anchor.P_O2_Pa, anchor.T_K]])[0])

    return _interp(minuend) - _interp(subtrahend)
