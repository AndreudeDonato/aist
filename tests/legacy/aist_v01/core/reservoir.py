"""Gas-phase reservoir species appended to a grand-canonical structure pool.

Volatile molecules (e.g. IrO₂(g), IrO₃(g)) can compete with the clusters as sinks for the reference atom.
``merge_gas_oxides`` appends them to the structure file that ``compute_gc`` reads; the ``patch_*`` functions
then give those rows the gas-phase thermodynamics (translation, rotation, optionally vibrations) at a fixed
trace pressure that ``compute_gc`` does not apply to cluster rows, and re-ensemble the NPZ.
"""
from __future__ import annotations

import json
import os

import numpy as np
from ase import Atoms
from ase.io import read, write

from aist_v01.composedthermo import compute_thermo_contributions
from aist_v01.core.ensemble import reensemble_from_g, refresh_ensemble


def elec_states(atoms: Atoms):
    """``[(0.0, J)]`` from ``atoms.info['total_spin']`` (dict or ``_JSON`` string), else ``None``."""
    spin = atoms.info.get("total_spin", {})
    if isinstance(spin, str):
        spin = json.loads(spin.replace("_JSON ", "", 1).strip())
    j = spin.get("J") if isinstance(spin, dict) else None
    return [(0.0, float(j))] if j is not None else None


def merge_gas_oxides(
    structures_file: str,
    gas_specs: list[tuple[str, float]],
    merged_path: str,
) -> tuple[str, list[tuple[int, Atoms, float]]]:
    """Append existing gas-species files; return the path to read and ``(row, atoms, P_gas)`` per species."""
    structures = read(structures_file, index=":")
    gas_mols: list[tuple[Atoms, float]] = []
    for path, p_gas in gas_specs:
        if not os.path.exists(path):
            continue
        mol = read(path)
        if not isinstance(mol, Atoms):
            mol = mol[0]
        gas_mols.append((mol, p_gas))
    if not gas_mols:
        return structures_file, []
    write(merged_path, structures + [mol for mol, _ in gas_mols])
    base = len(structures)
    return merged_path, [(base + i, mol, p) for i, (mol, p) in enumerate(gas_mols)]


def gas_vib_zpe_delta(atoms: Atoms, tg: np.ndarray, p_gas: float, *, include_e: bool) -> np.ndarray:
    """Vib/ZPE contribution relative to the electronic-only baseline."""
    elec = elec_states(atoms) if include_e else None
    pgas = np.full_like(tg, p_gas)
    g_vib, _ = compute_thermo_contributions(
        atoms, tg, pgas, elec_states=elec,
        include_v=True, include_zpe=True, include_r=False, include_t=False,
    )
    g_elec, _ = compute_thermo_contributions(
        atoms, tg, pgas, elec_states=elec,
        include_v=False, include_zpe=False, include_r=False, include_t=False,
    )
    return np.asarray(g_vib - g_elec, dtype=np.float32)


def gas_rot_trans_delta(
    atoms: Atoms,
    tg: np.ndarray,
    p_gas: float,
    *,
    use_vib_zpe: bool | None = None,
    include_v: bool | None = None,
    include_zpe: bool | None = None,
    include_e: bool,
) -> np.ndarray:
    """Rotational + translational G at ``p_gas`` relative to the same species without them."""
    if use_vib_zpe is not None:
        include_v = bool(use_vib_zpe)
        include_zpe = bool(use_vib_zpe)
    if include_v is None:
        include_v = False
    if include_zpe is None:
        include_zpe = False
    elec = elec_states(atoms) if include_e else None
    pgas = np.full_like(tg, p_gas)
    g_rt, _ = compute_thermo_contributions(
        atoms, tg, pgas, elec_states=elec,
        include_v=include_v, include_zpe=include_zpe, include_r=True, include_t=True,
    )
    g_base, _ = compute_thermo_contributions(
        atoms, tg, pgas, elec_states=elec,
        include_v=include_v, include_zpe=include_zpe, include_r=False, include_t=False,
    )
    return np.asarray(g_rt - g_base, dtype=np.float32)


def gas_reservoir_thermo_delta(
    atoms: Atoms, tg: np.ndarray, p_gas: float, *, use_vib_zpe: bool, include_e: bool,
) -> np.ndarray:
    """Full gas thermo minus the novib electronic baseline stored by ``run_gc``."""
    elec = elec_states(atoms) if include_e else None
    pgas = np.full_like(tg, p_gas)
    g_full, _ = compute_thermo_contributions(
        atoms, tg, pgas, elec_states=elec,
        include_v=use_vib_zpe, include_zpe=use_vib_zpe, include_r=True, include_t=True,
    )
    g_elec, _ = compute_thermo_contributions(
        atoms, tg, pgas, elec_states=elec,
        include_v=False, include_zpe=False, include_r=False, include_t=False,
    )
    return np.asarray(g_full - g_elec, dtype=np.float32)


def _load_g(npz_path: str) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    data = dict(np.load(npz_path, allow_pickle=True))
    g = np.asarray(data["G_tensor"], dtype=float)
    tg = np.asarray(data["Tg"], dtype=float)
    stoich = np.asarray(data["stoich"], dtype=float)
    return data, g, tg, stoich


def _save_g(npz_path: str, data: dict, g: np.ndarray, tg: np.ndarray, stoich: np.ndarray) -> None:
    data["G_tensor"] = g.astype(np.float32)
    refresh_ensemble(data, g, tg, stoich)
    np.savez(npz_path, **data)


def patch_gas_strip_vib_zpe(
    npz_path: str, gas_rows: list[tuple[int, Atoms, float]], *, include_e: bool = True,
) -> None:
    """Remove vib/ZPE from gas rows (undo what ``run_gc`` may have included)."""
    if not gas_rows:
        return
    data, g, tg, stoich = _load_g(npz_path)
    for row, atoms, p_gas in gas_rows:
        g[row] -= gas_vib_zpe_delta(atoms, tg, p_gas, include_e=include_e)
    _save_g(npz_path, data, g, tg, stoich)


def patch_gas_rot_trans(
    npz_path: str, gas_rows: list[tuple[int, Atoms, float]], *, use_vib_zpe: bool, include_e: bool = True,
) -> None:
    """Add rotational + translational G corrections at fixed trace pressure."""
    if not gas_rows:
        return
    data, g, tg, stoich = _load_g(npz_path)
    for row, atoms, p_gas in gas_rows:
        g[row] += gas_rot_trans_delta(atoms, tg, p_gas, use_vib_zpe=use_vib_zpe, include_e=include_e)
    _save_g(npz_path, data, g, tg, stoich)


def patch_gas_reservoir_thermo(
    npz_path: str, gas_rows: list[tuple[int, Atoms, float]], *, use_vib_zpe: bool, include_e: bool = True,
) -> None:
    """Patch gas rows onto a novib ``run_gc`` NPZ (vib/ZPE + rot/trans for gas only)."""
    if not gas_rows:
        return
    data, g, tg, stoich = _load_g(npz_path)
    for row, atoms, p_gas in gas_rows:
        g[row] += gas_reservoir_thermo_delta(atoms, tg, p_gas, use_vib_zpe=use_vib_zpe, include_e=include_e)
    _save_g(npz_path, data, g, tg, stoich)


def _patch_gc_variants_branch(
    g, tg, stoich, gas_rows, *, use_vib_zpe=None, include_v=None, include_zpe=None,
    include_e: bool, include_rot_trans: bool, strip_vib_zpe: bool, degeneracy,
) -> dict:
    if use_vib_zpe is not None:
        include_v = bool(use_vib_zpe)
        include_zpe = bool(use_vib_zpe)
    if include_v is None:
        include_v = False
    if include_zpe is None:
        include_zpe = False
    for row, atoms, p_gas in gas_rows:
        if strip_vib_zpe:
            g[row] -= gas_vib_zpe_delta(atoms, tg, p_gas, include_e=include_e)
        if include_rot_trans:
            g[row] += gas_rot_trans_delta(
                atoms, tg, p_gas, include_v=include_v, include_zpe=include_zpe, include_e=include_e,
            )
    return reensemble_from_g(g, tg, stoich, degeneracy)


VARIANT_BRANCHES = (
    ("novib", False, False),
    ("full", True, True),
    ("vib_nozpe", True, False),
    ("zpe_only", False, True),
)


def patch_gc_variants_gas(
    npz_path: str,
    gas_rows: list[tuple[int, Atoms, float]],
    *,
    include_e: bool = True,
    include_rot_trans: bool = True,
    strip_vib_zpe: bool = False,
) -> None:
    """Gas thermo patches on the novib / full / vib_nozpe / zpe_only branches of a ``run_gc_variants`` NPZ."""
    if not gas_rows:
        return
    data = dict(np.load(npz_path, allow_pickle=True))
    ts = np.asarray(data["Ts"], dtype=float)
    ps = np.asarray(data["Ps"], dtype=float)
    tg, _ = np.meshgrid(ts, ps, indexing="xy")
    stoich = np.asarray(data["stoich"], dtype=float)
    for prefix, include_v, include_zpe in VARIANT_BRANCHES:
        g_key = f"{prefix}_G_tensor"
        if g_key not in data:
            continue
        g = np.asarray(data[g_key], dtype=float)
        ens = _patch_gc_variants_branch(
            g, tg, stoich, gas_rows, include_v=include_v, include_zpe=include_zpe, include_e=include_e,
            include_rot_trans=include_rot_trans, strip_vib_zpe=strip_vib_zpe, degeneracy=data["degeneracy"],
        )
        data[g_key] = g.astype(np.float32)
        data[f"{prefix}_Pgi"] = ens["Pgi"]
        data[f"{prefix}_phase_index"] = ens["phase_index"]
        data[f"{prefix}_x_per_non_o"] = ens["x_per_non_o"]
        data[f"{prefix}_stoich_map"] = stoich[ens["phase_index"]]
    np.savez(npz_path, **data)
