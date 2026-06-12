"""Vibrational thermodynamics helpers (ZPE, Tchar, H/S at fixed T)."""
from __future__ import annotations

from typing import Any

import numpy as np
from ase import Atoms
from ase.io import read

from aist.composedthermo import process_frequencies
from aist.thermo import ThermoFromAtoms, ThermoPartitionFunctions


_TP = ThermoPartitionFunctions()


def prepare_vibrational_modes(atoms: Atoms) -> np.ndarray:
    """Return vibrational frequencies (cm-1) with trans/rot modes stripped."""
    freqs = atoms.info.get("frequencies", None)
    if freqs is None:
        return np.array([], dtype=float)
    freqs = np.asarray(freqs, dtype=float)
    if freqs.size == 0:
        return np.array([], dtype=float)
    thermo = ThermoFromAtoms(atoms)
    return process_frequencies(thermo, freqs)


def compute_zpe_energy(atoms: Atoms, temperature_k: float = 1.0) -> float:
    """ZPE-only vibrational energy (eV) at low T with zpe=True."""
    vib_modes = prepare_vibrational_modes(atoms)
    if vib_modes.size == 0:
        return np.nan
    thermo = ThermoFromAtoms(atoms)
    e_vib, _, _ = thermo.get_vibrational(
        vib_modes, np.array([float(temperature_k)]), zpe=True
    )
    return float(np.asarray(e_vib).reshape(-1)[0])


def tchar_mean(atoms: Atoms) -> float:
    """Mean Einstein characteristic temperature (K)."""
    freqs = prepare_vibrational_modes(atoms)
    if freqs.size == 0:
        return np.nan
    vib_temp = freqs * 100.0 * _TP.c * _TP.h_planck / _TP.kb
    return float(np.mean(vib_temp))


def tchar_min(atoms: Atoms) -> float:
    freqs = prepare_vibrational_modes(atoms)
    if freqs.size == 0:
        return np.nan
    vib_temp = freqs * 100.0 * _TP.c * _TP.h_planck / _TP.kb
    return float(np.min(vib_temp))


def mode_characteristic_temperatures(atoms: Atoms) -> np.ndarray:
    """Characteristic vibrational temperatures Θ_i (K) for each cluster mode."""
    freqs = prepare_vibrational_modes(atoms)
    if freqs.size == 0:
        return np.array([], dtype=float)
    return freqs * 100.0 * _TP.c * _TP.h_planck / _TP.kb


def _thermal_vib_energy_per_mode(theta: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Thermal vibrational energy (J/mol) per mode; ``theta`` (M,), ``T`` (nT,)."""
    theta = np.asarray(theta, dtype=float).reshape(-1, 1)
    T = np.asarray(T, dtype=float).reshape(1, -1)
    ratio = theta / T
    with np.errstate(over="ignore", invalid="ignore"):
        e_th = _TP.r_cons * theta / (np.exp(ratio) - 1.0)
    large = ratio > 500.0
    if np.any(large):
        e_th = np.where(large, _TP.r_cons * T * np.exp(-ratio), e_th)
    return e_th


def tchar_energy_weighted_at_Ts(atoms: Atoms, Ts: np.ndarray) -> np.ndarray:
    """Energy-weighted mean Θ at each temperature (cluster thermal modes only)."""
    theta = mode_characteristic_temperatures(atoms)
    if theta.size == 0:
        return np.full(np.asarray(Ts).size, np.nan, dtype=float)

    e_th = _thermal_vib_energy_per_mode(theta, Ts)
    e_tot = e_th.sum(axis=0)
    weights = np.zeros_like(e_th)
    valid = e_tot > 0.0
    if np.any(valid):
        weights[:, valid] = e_th[:, valid] / e_tot[valid]
    theta_eff = np.sum(weights * theta.reshape(-1, 1), axis=0)
    theta_eff = np.where(valid, theta_eff, np.nan)
    return theta_eff.astype(float)


def tchar_energy_weighted(atoms: Atoms, temperature_k: float) -> float:
    """Energy-weighted mean Θ (K) from cluster modes at one temperature."""
    val = tchar_energy_weighted_at_Ts(atoms, np.array([float(temperature_k)]))
    return float(val[0])


def hvib_svib_at_T(atoms: Atoms, T: float) -> tuple[float, float]:
    """Return (H_vib incl. ZPE, S_vib) in eV and eV/K at temperature T."""
    freqs = prepare_vibrational_modes(atoms)
    if freqs.size == 0:
        return np.nan, np.nan
    thermo = ThermoFromAtoms(atoms)
    ev, sv, _ = thermo.get_vibrational(freqs, np.array([float(T)]))
    zpe = float(atoms.info.get("zpe", 0.0))
    return float(np.asarray(ev).ravel()[0]) + zpe, float(np.asarray(sv).ravel()[0])


def compute_vib_table(structures: list[Atoms], T_ref: float = 273.0) -> dict[str, np.ndarray]:
    """Per-structure vibrational descriptors."""
    n = len(structures)
    tc_mean = np.full(n, np.nan)
    tc_min = np.full(n, np.nan)
    H_vib = np.full(n, np.nan)
    S_vib = np.full(n, np.nan)

    for i, atoms in enumerate(structures):
        tc_mean[i] = tchar_mean(atoms)
        tc_min[i] = tchar_min(atoms)
        h, s = hvib_svib_at_T(atoms, T_ref)
        H_vib[i], S_vib[i] = h, s

    return {
        "tchar_mean": tc_mean,
        "tchar_min": tc_min,
        "H_vib": H_vib,
        "S_vib": S_vib,
        "mTS_vib": -T_ref * S_vib,
        "G_vib": H_vib - T_ref * S_vib,
    }


def tchar_ensemble_map(
    structures: list[Atoms],
    Pgi: np.ndarray,
    *,
    use: str = "mean",
) -> np.ndarray:
    """Ensemble-weighted characteristic T on (nP, nT) grid."""
    fn = tchar_mean if use == "mean" else tchar_min
    tc = np.array([fn(a) for a in structures], dtype=float)
    valid = np.isfinite(tc)
    if not valid.any():
        return np.full(Pgi.shape[1:], np.nan)

    n = min(len(structures), Pgi.shape[0])
    if len(structures) != Pgi.shape[0]:
        tc = tc[:n]
        valid = valid[:n]
        Pgi = Pgi[:n]

    Pgi_valid = Pgi[valid]
    tc_valid = tc[valid]
    weight_sum = Pgi_valid.sum(axis=0)
    tc_map = np.einsum("i,ijk->jk", tc_valid, Pgi_valid)
    with np.errstate(invalid="ignore", divide="ignore"):
        tc_map = np.where(weight_sum > 0, tc_map / weight_sum, np.nan)
    return tc_map


def tchar_energy_weighted_ensemble_map(
    structures: list[Atoms],
    Pgi: np.ndarray,
    Ts: np.ndarray,
) -> np.ndarray:
    """Ensemble map of energy-weighted cluster Θ_eff(T) on the GC (nP, nT) grid.

    At each grid temperature ``T_j``, per-structure values use thermal
    energy weights from cluster modes only (no O₂ / Ir reference). The map
    field is ``Pgi``-weighted over isomers at each ``(p, T_j)``.
    """
    Pgi = np.asarray(Pgi, dtype=float)
    Ts = np.asarray(Ts, dtype=float)
    n_s = min(len(structures), Pgi.shape[0])
    if n_s == 0 or Ts.size == 0:
        return np.full(Pgi.shape[1:], np.nan)

    tc_at_T = np.full((n_s, Ts.size), np.nan, dtype=float)
    for i, atoms in enumerate(structures[:n_s]):
        row = tchar_energy_weighted_at_Ts(atoms, Ts)
        if np.isfinite(row).any():
            tc_at_T[i] = row

    valid = np.isfinite(tc_at_T).any(axis=1)
    if not np.any(valid):
        return np.full(Pgi.shape[1:], np.nan)

    Pgi = Pgi[:n_s]
    tc_at_T = tc_at_T[valid]
    Pgi = Pgi[valid]

    # tc_map[p, j] = sum_i Pgi[i,p,j] * tc_at_T[i,j] / sum_i Pgi[i,p,j]
    weight_sum = Pgi.sum(axis=0)
    tc_map = np.einsum("ij,ikj->kj", tc_at_T, Pgi)
    with np.errstate(invalid="ignore", divide="ignore"):
        tc_map = np.where(weight_sum > 0, tc_map / weight_sum, np.nan)
    return tc_map


def zpe_by_ir_content(
    structures: list[Atoms] | str,
    o2_atoms: Atoms | str,
    *,
    ir_symbol: str = "Ir",
    o_symbol: str = "O",
    zpe_temperature_k: float = 1.0,
) -> dict[int, list[dict[str, Any]]]:
    """Group ZPE metrics by Ir count (logic from analysis/zpe_by_ir_content.py)."""
    if isinstance(structures, str):
        structures = list(read(structures, ":"))
    if isinstance(o2_atoms, str):
        o2_atoms = read(o2_atoms)

    zpe_o2 = compute_zpe_energy(o2_atoms, zpe_temperature_k)
    if not np.isfinite(zpe_o2):
        raise ValueError("Could not compute finite O2 ZPE")
    zpe_o_per_atom = 0.5 * float(zpe_o2)

    grouped: dict[int, list[dict[str, Any]]] = {}
    for atoms in structures:
        syms = atoms.get_chemical_symbols()
        n_ir = sum(1 for s in syms if s == ir_symbol)
        n_o = sum(1 for s in syms if s == o_symbol)
        if n_ir <= 0:
            continue
        zpe = compute_zpe_energy(atoms, zpe_temperature_k)
        if not np.isfinite(zpe):
            continue
        grouped.setdefault(n_ir, []).append({"n_o": n_o, "zpe": zpe})

    for n_ir, rows in grouped.items():
        zpe_ref = next((r["zpe"] for r in rows if int(r["n_o"]) == 0), None)
        if zpe_ref is None:
            zpe_ref = 0.0
        for r in rows:
            r["zpe_diff"] = float(r["zpe"]) - float(zpe_ref)
            r["zpe_minus_o2_ref"] = float(r["zpe"]) - float(r["n_o"]) * zpe_o_per_atom

    return grouped


def zpe_global_minima_by_ir_content(
    structures: list[Atoms] | str,
    *,
    ir_symbol: str = "Ir",
    o_symbol: str = "O",
    zpe_temperature_k: float = 1.0,
) -> dict[int, list[dict[str, Any]]]:
    """ZPE of lowest-energy isomer per stoichiometry, minus Ir_n with zero O.

    For each Ir count, plots use only the global minimum structure at each
  ``(n_Ir, n_O)`` and subtract the ZPE of the ``n_O = 0`` minimum for that
    Ir count (not an O₂ molecular reference).
    """
    from aist.core.minima import select_stoichiometry_minima

    if isinstance(structures, str):
        structures = list(read(structures, ":"))

    minima = select_stoichiometry_minima(
        structures,
        x_group_formula=ir_symbol,
        y_group_formula=o_symbol,
        orient=False,
    )

    grouped: dict[int, list[dict[str, Any]]] = {}
    for n_ir, n_o, atoms in minima:
        if n_ir <= 0:
            continue
        zpe = compute_zpe_energy(atoms, zpe_temperature_k)
        if not np.isfinite(zpe):
            continue
        grouped.setdefault(n_ir, []).append({"n_o": n_o, "zpe": zpe})

    out: dict[int, list[dict[str, Any]]] = {}
    for n_ir, rows in grouped.items():
        zpe_ref = next((r["zpe"] for r in rows if int(r["n_o"]) == 0), None)
        if zpe_ref is None:
            continue
        zpe_ref = float(zpe_ref)
        enriched = []
        for r in rows:
            enriched.append({
                "n_o": int(r["n_o"]),
                "zpe": float(r["zpe"]),
                "zpe_diff": float(r["zpe"]) - zpe_ref,
            })
        out[n_ir] = enriched

    return out
