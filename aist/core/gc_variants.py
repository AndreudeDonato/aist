"""Full / novib / vib-no-ZPE / ZPE-only grand-canonical variants (0.1 file-based API over ``aist.gce``)."""
from __future__ import annotations

import os

import numpy as np
from ase.io import read

from aist.core.grandcanonical import GCConfig, ensemble_from_config


def _branch_out(prefix: str, result: dict, stoich: np.ndarray) -> dict:
    return {
        f"{prefix}_phase_index": result["phase_index"],
        f"{prefix}_x_per_non_o": result["x_per_non_o"],
        f"{prefix}_stoich_map": stoich[result["phase_index"]],
        f"{prefix}_Pgi": result["Pgi"],
        f"{prefix}_G_tensor": result["G_tensor"],
    }


def run_gc_variants(
    structures_file: str,
    pressure_file: str,
    output_npz: str,
    *,
    grid_config: GCConfig | None = None,
) -> None:
    """Compute and save full / novib / vib-no-ZPE / ZPE-only GC maps with ``Pgi`` tensors.

    Only the grid and the reference file are taken from ``grid_config``; every other setting is the
    ``GCConfig`` default, as in aist 0.1. The variants share one ensemble, so the partition functions of each
    structure are computed once.
    """
    os.makedirs(os.path.dirname(output_npz) or ".", exist_ok=True)

    base = grid_config or GCConfig()
    cfg = GCConfig(
        T_min=base.T_min,
        T_max=base.T_max,
        n_T=base.n_T,
        P_min=base.P_min,
        P_max=base.P_max,
        n_P=base.n_P,
        reference_file=base.reference_file,
        include_decomposition=False,
    )
    ens = ensemble_from_config(read(structures_file, index=":"), read(pressure_file), cfg)
    reservoir = ens.reservoirs[0]
    x_element = next(iter(reservoir.composition))
    Ts = np.linspace(cfg.T_min, cfg.T_max, cfg.n_T)
    Ps = np.logspace(np.log10(cfg.P_min), np.log10(cfg.P_max), cfg.n_P)
    stoich = np.array(ens.count(x_element), dtype=float) / np.array(ens.normalization, dtype=float)

    branches = (
        ("full", True, True, "full (vib + ZPE + ensemble)"),
        ("novib", False, False, "novib (no vib, no ZPE)"),
        ("vib_nozpe", True, False, "vib only (no ZPE)"),
        ("zpe_only", False, True, "ZPE only (no thermal vib)"),
    )

    out: dict = {"Ts": Ts, "Ps": Ps, "stoich": stoich, "degeneracy": np.array(ens.degeneracy, dtype=float)}
    for prefix, use_v, use_zpe, label in branches:
        print(f"[gc_variants] {label}")
        res = ens.variant(vibrations=use_v, zpe=use_zpe).evaluate(T=Ts, p={reservoir.name: Ps})
        result = {
            "phase_index": res.phase_index,
            "x_per_non_o": np.sum(res.populations * stoich[:, None, None], axis=0),
            "Pgi": res.populations,
            "G_tensor": res.G,
        }
        out.update(_branch_out(prefix, result, stoich))
        del res, result

    np.savez(output_npz, **out)
    print(f"[gc_variants] saved: {output_npz}")
