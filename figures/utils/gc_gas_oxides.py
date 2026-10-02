"""Append volatile IrO₂/IrO₃ gas molecules to the GC structure pool (figures layer only)."""
from __future__ import annotations

import json
import os

import numpy as np
from ase import Atoms
from ase.io import read, write
from scipy.constants import Boltzmann, electron_volt

from aist.composedthermo import compute_thermo_contributions


def _elec_states(atoms: Atoms):
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
    """Append existing gas-oxide files; return path and ``(row, atoms, P_gas)``."""
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


def _gas_vib_zpe_delta(
    atoms: Atoms,
    tg: np.ndarray,
    p_gas: float,
    *,
    include_e: bool,
) -> np.ndarray:
    """Vib/ZPE contribution relative to the electronic-only baseline."""
    elec = _elec_states(atoms) if include_e else None
    pgas = np.full_like(tg, p_gas)
    g_vib, _ = compute_thermo_contributions(
        atoms, tg, pgas, elec_states=elec,
        include_v=True, include_zpe=True,
        include_r=False, include_t=False,
    )
    g_elec, _ = compute_thermo_contributions(
        atoms, tg, pgas, elec_states=elec,
        include_v=False, include_zpe=False,
        include_r=False, include_t=False,
    )
    return np.asarray(g_vib - g_elec, dtype=np.float32)


def _gas_rot_trans_delta(
    atoms: Atoms,
    tg: np.ndarray,
    p_gas: float,
    *,
    use_vib_zpe: bool | None = None,
    include_v: bool | None = None,
    include_zpe: bool | None = None,
    include_e: bool,
) -> np.ndarray:
    if use_vib_zpe is not None:
        include_v = bool(use_vib_zpe)
        include_zpe = bool(use_vib_zpe)
    if include_v is None:
        include_v = False
    if include_zpe is None:
        include_zpe = False
    elec = _elec_states(atoms) if include_e else None
    pgas = np.full_like(tg, p_gas)
    g_rt, _ = compute_thermo_contributions(
        atoms, tg, pgas, elec_states=elec,
        include_v=include_v, include_zpe=include_zpe,
        include_r=True, include_t=True,
    )
    g_base, _ = compute_thermo_contributions(
        atoms, tg, pgas, elec_states=elec,
        include_v=include_v, include_zpe=include_zpe,
        include_r=False, include_t=False,
    )
    return np.asarray(g_rt - g_base, dtype=np.float32)


def _gas_reservoir_thermo_delta(
    atoms: Atoms,
    tg: np.ndarray,
    p_gas: float,
    *,
    use_vib_zpe: bool,
    include_e: bool,
) -> np.ndarray:
    """Full gas thermo minus the novib electronic baseline stored by ``run_gc``."""
    elec = _elec_states(atoms) if include_e else None
    pgas = np.full_like(tg, p_gas)
    g_full, _ = compute_thermo_contributions(
        atoms, tg, pgas, elec_states=elec,
        include_v=use_vib_zpe, include_zpe=use_vib_zpe,
        include_r=True, include_t=True,
    )
    g_elec, _ = compute_thermo_contributions(
        atoms, tg, pgas, elec_states=elec,
        include_v=False, include_zpe=False,
        include_r=False, include_t=False,
    )
    return np.asarray(g_full - g_elec, dtype=np.float32)


def _reensemble_from_g(g: np.ndarray, tg: np.ndarray, stoich: np.ndarray, degeneracy: np.ndarray) -> dict:
    """Populations from a patched ``G_tensor``, with the same chirality degeneracies as ``compute_gc``."""
    kB = Boltzmann / electron_volt
    beta = 1.0 / (kB * tg)
    weights = np.exp(-beta * (g - np.min(g, axis=0, keepdims=True)))
    weights *= np.asarray(degeneracy, dtype=float)[:, None, None]
    xi = np.sum(weights, axis=0)
    pgi = (weights / xi).astype(np.float32)
    phase_index = np.argmin(g, axis=0)
    return {
        "Pgi": pgi,
        "phase_index": phase_index,
        "x_per_non_o": np.sum(pgi * stoich[:, None, None], axis=0),
        "G_avg": np.sum(pgi * g, axis=0),
        "S_conf": -kB * np.sum(pgi * np.log(pgi + 1e-20), axis=0),
        "pop_max": np.max(pgi, axis=(1, 2)),
    }


def patch_gas_strip_vib_zpe(
    npz_path: str,
    gas_rows: list[tuple[int, Atoms, float]],
    *,
    include_e: bool = True,
) -> None:
    """Remove vib/ZPE from gas rows (undo what ``run_gc`` may have included)."""
    if not gas_rows:
        return
    data = dict(np.load(npz_path, allow_pickle=True))
    g = np.asarray(data["G_tensor"], dtype=float)
    tg = np.asarray(data["Tg"], dtype=float)
    stoich = np.asarray(data["stoich"], dtype=float)

    for row, atoms, p_gas in gas_rows:
        g[row] -= _gas_vib_zpe_delta(atoms, tg, p_gas, include_e=include_e)

    data["G_tensor"] = g.astype(np.float32)
    data.update(_reensemble_from_g(g, tg, stoich, data["degeneracy"]))
    np.savez(npz_path, **data)


def patch_gas_rot_trans(
    npz_path: str,
    gas_rows: list[tuple[int, Atoms, float]],
    *,
    use_vib_zpe: bool,
    include_e: bool = True,
) -> None:
    """Add rotational + translational G corrections at fixed trace pressure."""
    if not gas_rows:
        return
    data = dict(np.load(npz_path, allow_pickle=True))
    g = np.asarray(data["G_tensor"], dtype=float)
    tg = np.asarray(data["Tg"], dtype=float)
    stoich = np.asarray(data["stoich"], dtype=float)

    for row, atoms, p_gas in gas_rows:
        g[row] += _gas_rot_trans_delta(
            atoms, tg, p_gas, use_vib_zpe=use_vib_zpe, include_e=include_e,
        )

    data["G_tensor"] = g.astype(np.float32)
    data.update(_reensemble_from_g(g, tg, stoich, data["degeneracy"]))
    np.savez(npz_path, **data)


def patch_gas_reservoir_thermo(
    npz_path: str,
    gas_rows: list[tuple[int, Atoms, float]],
    *,
    use_vib_zpe: bool,
    include_e: bool = True,
) -> None:
    """Patch gas rows onto a novib ``run_gc`` NPZ (vib/ZPE + rot/trans for gas only)."""
    if not gas_rows:
        return
    data = dict(np.load(npz_path, allow_pickle=True))
    g = np.asarray(data["G_tensor"], dtype=float)
    tg = np.asarray(data["Tg"], dtype=float)
    stoich = np.asarray(data["stoich"], dtype=float)

    for row, atoms, p_gas in gas_rows:
        g[row] += _gas_reservoir_thermo_delta(
            atoms, tg, p_gas, use_vib_zpe=use_vib_zpe, include_e=include_e,
        )

    data["G_tensor"] = g.astype(np.float32)
    data.update(_reensemble_from_g(g, tg, stoich, data["degeneracy"]))
    np.savez(npz_path, **data)


def _patch_gc_variants_branch(
    g: np.ndarray,
    tg: np.ndarray,
    stoich: np.ndarray,
    gas_rows: list[tuple[int, Atoms, float]],
    *,
    use_vib_zpe: bool | None = None,
    include_v: bool | None = None,
    include_zpe: bool | None = None,
    include_e: bool,
    include_rot_trans: bool,
    strip_vib_zpe: bool,
    degeneracy: np.ndarray,
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
            g[row] -= _gas_vib_zpe_delta(atoms, tg, p_gas, include_e=include_e)
        if include_rot_trans:
            g[row] += _gas_rot_trans_delta(
                atoms, tg, p_gas,
                include_v=include_v, include_zpe=include_zpe,
                include_e=include_e,
            )
    return _reensemble_from_g(g, tg, stoich, degeneracy)


def patch_gc_variants_gas(
    npz_path: str,
    gas_rows: list[tuple[int, Atoms, float]],
    *,
    include_e: bool = True,
    include_rot_trans: bool = True,
    strip_vib_zpe: bool = False,
) -> None:
    """Gas thermo patches on novib / full / vib_nozpe branches in ``gc_variants.npz``."""
    if not gas_rows:
        return
    data = dict(np.load(npz_path, allow_pickle=True))
    ts = np.asarray(data["Ts"], dtype=float)
    ps = np.asarray(data["Ps"], dtype=float)
    tg, _ = np.meshgrid(ts, ps, indexing="xy")
    stoich = np.asarray(data["stoich"], dtype=float)

    branch_flags = (
        ("novib", False, False),
        ("full", True, True),
        ("vib_nozpe", True, False),
        ("zpe_only", False, True),
    )
    for prefix, include_v, include_zpe in branch_flags:
        g_key = f"{prefix}_G_tensor"
        if g_key not in data:
            continue
        g = np.asarray(data[g_key], dtype=float)
        ens = _patch_gc_variants_branch(
            g, tg, stoich, gas_rows,
            include_v=include_v, include_zpe=include_zpe,
            include_e=include_e,
            include_rot_trans=include_rot_trans, strip_vib_zpe=strip_vib_zpe,
            degeneracy=data["degeneracy"],
        )
        data[g_key] = g.astype(np.float32)
        data[f"{prefix}_Pgi"] = ens["Pgi"]
        data[f"{prefix}_phase_index"] = ens["phase_index"]
        data[f"{prefix}_x_per_non_o"] = ens["x_per_non_o"]
        data[f"{prefix}_stoich_map"] = stoich[ens["phase_index"]]

    np.savez(npz_path, **data)
