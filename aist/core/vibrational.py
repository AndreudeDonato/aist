"""Vibrational thermodynamics helpers (ZPE, Tchar, H/S at fixed T)."""
from __future__ import annotations

from typing import Any

import numpy as np
from ase import Atoms
from ase.io import read

from aist.contributions import internal_frequencies, vibrational, zero_point_energy
from aist.thermo import ThermoFromAtoms, ThermoPartitionFunctions


_TP = ThermoPartitionFunctions()
KB_EV = _TP.kb * _TP.J_to_eV


def prepare_vibrational_modes(atoms: Atoms) -> np.ndarray:
    """Return vibrational frequencies (cm-1) with trans/rot modes stripped."""
    return internal_frequencies(atoms)


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
    if len(structures) != Pgi.shape[0]:
        raise ValueError(f"{len(structures)} structures but {Pgi.shape[0]} Pgi rows")
    n_s = len(structures)
    if n_s == 0 or Ts.size == 0:
        return np.full(Pgi.shape[1:], np.nan)

    tc_at_T = np.full((n_s, Ts.size), np.nan, dtype=float)
    for i, atoms in enumerate(structures):
        row = tchar_energy_weighted_at_Ts(atoms, Ts)
        if np.isfinite(row).any():
            tc_at_T[i] = row

    valid = np.isfinite(tc_at_T).any(axis=1)
    if not np.any(valid):
        return np.full(Pgi.shape[1:], np.nan)

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


def compute_vib_gibbs_contribution(atoms: Atoms, temperature_k: float) -> float:
    """Vibrational Gibbs term G_vib = E_vib + ZPE − T S_vib (eV), cluster modes only.

    The structure's own part of ``EnsembleResult.contributions()["vibrational"]``, without the reference and
    reservoir terms.
    """
    if prepare_vibrational_modes(atoms).size == 0:
        return np.nan
    T = float(temperature_k)
    ev, sv = vibrational(atoms, np.array([T]))
    return float((ev + zero_point_energy(atoms) - T * sv)[0])


def _structure_stoichiometry_groups(
    structures: list[Atoms],
    *,
    ir_symbol: str = "Ir",
    o_symbol: str = "O",
) -> dict[tuple[int, int], list[int]]:
    groups: dict[tuple[int, int], list[int]] = {}
    for i, atoms in enumerate(structures):
        syms = atoms.get_chemical_symbols()
        n_ir = sum(1 for s in syms if s == ir_symbol)
        n_o = sum(1 for s in syms if s == o_symbol)
        if n_ir <= 0:
            continue
        groups.setdefault((n_ir, n_o), []).append(i)
    return groups


def stoichiometry_weights(
    gc: dict,
    indices: list[int],
    p_idx: int,
    t_idx: int,
) -> np.ndarray:
    """Boltzmann weights over the isomers ``indices`` (one stoichiometry) at grid point ``(p_idx, t_idx)``.

    Same rule as ``GrandCanonicalEnsemble`` (``g_i exp(-beta ΔG_i)`` with the stored ``G_tensor`` and the
    chirality degeneracies), taken relative to the lowest isomer of the group, so the weights stay exact
    where the global populations ``Pgi`` underflow (stoichiometries far from stable).
    """
    idx = np.asarray(indices, dtype=int)
    if idx.size == 0:
        return np.zeros(0)
    for key in ("G_tensor", "degeneracy"):
        if key not in gc:
            raise KeyError(f"GC data has no {key!r}; rebuild the cache (build_all.py --rebuild-cache)")
    g = np.asarray(gc["G_tensor"][idx, p_idx, t_idx], dtype=float)
    beta = 1.0 / (KB_EV * float(np.asarray(gc["Ts"])[t_idx]))
    w = np.asarray(gc["degeneracy"], dtype=float)[idx] * np.exp(-beta * (g - g.min()))
    return w / w.sum()


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    mask = np.isfinite(values)
    if not np.any(mask):
        return np.nan
    v = values[mask]
    w = weights[mask]
    s = float(w.sum())
    if s <= 0.0:
        return float(np.mean(v))
    return float(np.dot(w / s, v))


def boltzmann_weighted_vib_stoichiometry_by_ir_content(
    structures: list[Atoms],
    gc: dict,
    *,
    temperature_k: float,
    pressure_pa: float,
    zpe_temperature_k: float = 1.0,
    ir_symbol: str = "Ir",
    o_symbol: str = "O",
) -> dict[int, list[dict[str, Any]]]:
    """Boltzmann-weighted ZPE and G_vib/n_Ir vs O content, by Ir count.

    At fixed ``(temperature_k, pressure_pa)``, weights within each
    ``(n_Ir, n_O)`` bucket are the GC Boltzmann weights restricted to the isomers of that
    stoichiometry (``stoichiometry_weights``; ``gc`` needs ``Ts``, ``Ps``, ``G_tensor``,
    ``degeneracy`` in the row order of ``structures``). ZPE uses ``zpe_temperature_k``; ``G_vib`` uses
    ``temperature_k``. Differences are referenced to the Boltzmann average at
    ``n_O = 0`` for each ``n_Ir``.
    """
    if len(structures) != np.shape(gc["G_tensor"])[0]:
        raise ValueError(f"{len(structures)} structures but {np.shape(gc['G_tensor'])[0]} GC rows")
    p_idx = int(np.argmin(np.abs(np.log(np.asarray(gc["Ps"], dtype=float)) - np.log(pressure_pa))))
    t_idx = int(np.argmin(np.abs(np.asarray(gc["Ts"], dtype=float) - temperature_k)))

    zpe_all = np.array(
        [compute_zpe_energy(s, zpe_temperature_k) for s in structures],
        dtype=float,
    )
    g_vib_all = np.array(
        [compute_vib_gibbs_contribution(s, temperature_k) for s in structures],
        dtype=float,
    )
    n_vib_modes_all = np.array(
        [float(prepare_vibrational_modes(s).size) for s in structures],
        dtype=float,
    )

    groups = _structure_stoichiometry_groups(
        structures, ir_symbol=ir_symbol, o_symbol=o_symbol,
    )

    avg: dict[tuple[int, int], dict[str, float]] = {}
    for (n_ir, n_o), idxs in groups.items():
        w = stoichiometry_weights(gc, idxs, p_idx, t_idx)
        avg[(n_ir, n_o)] = {
            "zpe": _weighted_mean(zpe_all[idxs], w),
            "g_vib": _weighted_mean(g_vib_all[idxs], w),
            "g_vib_per_ir": _weighted_mean(g_vib_all[idxs] / float(n_ir), w),
            "n_vib_modes": _weighted_mean(n_vib_modes_all[idxs], w),
        }

    by_ir: dict[int, list[dict[str, Any]]] = {}
    for (n_ir, n_o), metrics in avg.items():
        by_ir.setdefault(n_ir, []).append({"n_o": n_o, **metrics})

    out: dict[int, list[dict[str, Any]]] = {}
    for n_ir, rows in by_ir.items():
        ref = next((r for r in rows if int(r["n_o"]) == 0), None)
        if ref is None or not np.isfinite(ref["zpe"]):
            continue
        zpe_ref = float(ref["zpe"])
        g_ref = float(ref["g_vib"])
        g_ref_per_ir = float(ref["g_vib_per_ir"])
        enriched = []
        for r in rows:
            zpe = float(r["zpe"])
            g_vib = float(r["g_vib"])
            g_per_ir = float(r["g_vib_per_ir"])
            enriched.append({
                "n_o": int(r["n_o"]),
                "n_vib_modes": float(r["n_vib_modes"]),
                "zpe": zpe,
                "zpe_diff": zpe - zpe_ref,
                "g_vib": g_vib,
                "g_vib_diff": (
                    g_vib - g_ref
                    if np.isfinite(g_vib) and np.isfinite(g_ref)
                    else np.nan
                ),
                "g_vib_per_ir": g_per_ir,
                "g_vib_per_ir_diff": (
                    g_per_ir - g_ref_per_ir
                    if np.isfinite(g_per_ir) and np.isfinite(g_ref_per_ir)
                    else np.nan
                ),
            })
        out[n_ir] = enriched

    return out
