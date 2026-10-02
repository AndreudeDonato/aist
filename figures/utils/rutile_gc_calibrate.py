"""G_rutile − G_IrO₃(g) at the experimental anchor (T, pO₂): the check stored with the rutile GC cache."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import RegularGridInterpolator


@dataclass(frozen=True)
class RutileCalibrationAnchor:
    T_K: float
    P_O2_Pa: float


def rutile_minus_iro3_at_anchor(
    g: np.ndarray,
    ps: np.ndarray,
    ts: np.ndarray,
    formulas,
    *,
    anchor: RutileCalibrationAnchor,
) -> float:
    """``G_rutile − G_IrO₃(g)`` (eV per Ir) interpolated at *anchor*.

    ``g`` is the rutile-GC ``G_tensor`` (rows × P × T) with the gas rows already carrying the same
    thermo patches as the plotted maps, so the check describes exactly what the figure shows.
    """
    formulas = [str(f) for f in formulas]

    def _interp(formula: str) -> float:
        interp = RegularGridInterpolator((ps, ts), g[formulas.index(formula)], bounds_error=False, fill_value=None)
        return float(interp([[anchor.P_O2_Pa, anchor.T_K]])[0])

    return _interp("Ir2O4") - _interp("IrO3")
