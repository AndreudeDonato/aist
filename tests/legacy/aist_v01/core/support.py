"""Support-stabilisation proxy for free clusters standing in for supported ones.

Every cluster row of a ``run_gc`` NPZ is shifted by a flat adhesion-energy proxy, ``-(ev_per_ir * n_M +
ev_per_o * n_O)`` per cluster, i.e. ``-(ev_per_ir + ev_per_o * n_O/n_M)`` on the per-metal-atom G tensor
(``n_M`` = non-O atoms, as in ``compute_gc``); gas-reservoir rows are left untouched and the populations are
re-ensembled.
"""
from __future__ import annotations

import numpy as np
from ase.io import read

from aist_v01.core.ensemble import reensemble_from_g, refresh_ensemble
from aist_v01.core.reservoir import VARIANT_BRANCHES


def support_corrections_per_ir(
    structures_file: str, gas_rows_idx, *, ev_per_ir: float, ev_per_o: float,
) -> np.ndarray:
    """Per-row stabilisation (eV per metal atom, positive); zero for gas rows."""
    structures = read(structures_file, index=":")
    gas = {int(i) for i in gas_rows_idx}
    corr = np.zeros(len(structures), dtype=float)
    for i, atoms in enumerate(structures):
        if i in gas:
            continue
        symbols = atoms.get_chemical_symbols()
        n_ir = sum(1 for s in symbols if s != "O")
        n_o = symbols.count("O")
        corr[i] = ev_per_ir + ev_per_o * (n_o / n_ir)
    return corr


def patch_support_stabilization(
    npz_path: str, structures_file: str, gas_rows_idx, *, ev_per_ir: float, ev_per_o: float,
) -> None:
    """Stabilise the cluster rows of a main ``run_gc`` NPZ and re-ensemble."""
    corr = support_corrections_per_ir(structures_file, gas_rows_idx, ev_per_ir=ev_per_ir, ev_per_o=ev_per_o)
    data = dict(np.load(npz_path, allow_pickle=True))
    g = np.asarray(data["G_tensor"], dtype=float)
    tg = np.asarray(data["Tg"], dtype=float)
    stoich = np.asarray(data["stoich"], dtype=float)
    g -= corr[:, None, None]
    data["G_tensor"] = g.astype(np.float32)
    refresh_ensemble(data, g, tg, stoich)
    data["gc_support_ev_per_ir"] = np.float32(ev_per_ir)
    data["gc_support_ev_per_o"] = np.float32(ev_per_o)
    np.savez(npz_path, **data)


def patch_gc_variants_support(
    npz_path: str, structures_file: str, gas_rows_idx, *, ev_per_ir: float, ev_per_o: float,
) -> None:
    """Stabilise the cluster rows on every thermo branch of a ``run_gc_variants`` NPZ."""
    corr = support_corrections_per_ir(structures_file, gas_rows_idx, ev_per_ir=ev_per_ir, ev_per_o=ev_per_o)
    data = dict(np.load(npz_path, allow_pickle=True))
    ts = np.asarray(data["Ts"], dtype=float)
    ps = np.asarray(data["Ps"], dtype=float)
    tg, _ = np.meshgrid(ts, ps, indexing="xy")
    stoich = np.asarray(data["stoich"], dtype=float)
    for branch, _, _ in VARIANT_BRANCHES:
        key = f"{branch}_G_tensor"
        if key not in data:
            continue
        g = np.asarray(data[key], dtype=float)
        g -= corr[:, None, None]
        ens = reensemble_from_g(g, tg, stoich, data["degeneracy"])
        data[key] = g.astype(np.float32)
        data[f"{branch}_Pgi"] = ens["Pgi"]
        data[f"{branch}_phase_index"] = ens["phase_index"]
        data[f"{branch}_x_per_non_o"] = ens["x_per_non_o"]
        data[f"{branch}_stoich_map"] = stoich[ens["phase_index"]]
    data["gc_support_ev_per_ir"] = np.float32(ev_per_ir)
    data["gc_support_ev_per_o"] = np.float32(ev_per_o)
    np.savez(npz_path, **data)
