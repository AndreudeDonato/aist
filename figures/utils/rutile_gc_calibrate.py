"""Bisect rutile gas *P* so G_rutile = G_IrO₃(g) at the experimental anchor."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from figures.utils.gc_gas_oxides import _gas_reservoir_thermo_delta, _gas_rot_trans_delta


@dataclass(frozen=True)
class RutileCalibrationAnchor:
    T_K: float
    P_O2_Pa: float


def rutile_delta_g_rutile_minus_iro3(
    g_base: np.ndarray,
    ps: np.ndarray,
    ts: np.ndarray,
    gas_rows: list[tuple[int, object, float]],
    p_gas: float,
    *,
    anchor: RutileCalibrationAnchor,
    gas_use_vib: bool,
    rutile_idx: int = 0,
    iro3_idx: int = 2,
) -> float:
    t_anchor = np.array([[anchor.T_K]])

    def _interp(row: int) -> float:
        interp = RegularGridInterpolator(
            (ps, ts), g_base[row], bounds_error=False, fill_value=None,
        )
        return float(interp([[anchor.P_O2_Pa, anchor.T_K]])[0])

    g_rutile = _interp(rutile_idx)
    g_iro3 = _interp(iro3_idx)
    delta_fn = _gas_reservoir_thermo_delta if gas_use_vib else _gas_rot_trans_delta
    for _row, atoms, _ in gas_rows:
        if atoms.get_chemical_formula() == "IrO3":
            g_iro3 += float(
                delta_fn(
                    atoms, t_anchor, p_gas,
                    use_vib_zpe=gas_use_vib, include_e=True,
                )[0, 0]
            )
    return g_rutile - g_iro3


def calibrate_rutile_gas_p(
    g_base: np.ndarray,
    ps: np.ndarray,
    ts: np.ndarray,
    gas_rows: list[tuple[int, object, float]],
    *,
    anchor: RutileCalibrationAnchor,
    gas_use_vib: bool,
    p_min: float = 1.0e-5,
    p_max: float = 1.0e12,
    tol_eV: float = 1.0e-3,
    max_iter: int = 80,
) -> tuple[float, float]:
    """Return ``(P_gas_Pa, delta_G_eV)`` with ``G_rutile ≈ G_IrO₃(g)`` at *anchor*."""
    lo, hi = float(p_min), float(p_max)
    dg_lo = rutile_delta_g_rutile_minus_iro3(
        g_base, ps, ts, gas_rows, lo, anchor=anchor, gas_use_vib=gas_use_vib,
    )
    dg_hi = rutile_delta_g_rutile_minus_iro3(
        g_base, ps, ts, gas_rows, hi, anchor=anchor, gas_use_vib=gas_use_vib,
    )
    if dg_lo * dg_hi > 0:
        raise RuntimeError(
            f"Cannot bracket rutile/IrO₃ at T={anchor.T_K} K, p_O2={anchor.P_O2_Pa:.0f} Pa: "
            f"ΔG({p_min:.1e})={dg_lo:.4f} eV, ΔG({p_max:.1e})={dg_hi:.4f} eV"
        )

    for _ in range(max_iter):
        mid = np.sqrt(lo * hi)
        dg_mid = rutile_delta_g_rutile_minus_iro3(
            g_base, ps, ts, gas_rows, mid, anchor=anchor, gas_use_vib=gas_use_vib,
        )
        if abs(dg_mid) < tol_eV:
            return mid, dg_mid
        if dg_lo * dg_mid <= 0:
            hi, dg_hi = mid, dg_mid
        else:
            lo, dg_lo = mid, dg_mid

    mid = np.sqrt(lo * hi)
    dg_mid = rutile_delta_g_rutile_minus_iro3(
        g_base, ps, ts, gas_rows, mid, anchor=anchor, gas_use_vib=gas_use_vib,
    )
    return mid, dg_mid
