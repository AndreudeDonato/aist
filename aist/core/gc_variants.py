"""Full / novib / vib-no-ZPE grand-canonical variants."""
from __future__ import annotations

import os

import numpy as np

from aist.core.grandcanonical import GCConfig, compute_gc


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
    """Compute and save full / novib / vib-no-ZPE GC maps with ``Pgi`` tensors."""
    os.makedirs(os.path.dirname(output_npz) or ".", exist_ok=True)

    base = grid_config or GCConfig()
    common = {
        "T_min": base.T_min,
        "T_max": base.T_max,
        "n_T": base.n_T,
        "P_min": base.P_min,
        "P_max": base.P_max,
        "n_P": base.n_P,
        "reference_file": base.reference_file,
    }
    branches = (
        ("full", True, True, "full (vib + ZPE + ensemble)"),
        ("novib", False, False, "novib (no vib, no ZPE)"),
        ("vib_nozpe", True, False, "vib only (no ZPE)"),
        ("zpe_only", False, True, "ZPE only (no thermal vib)"),
    )

    out: dict = {}
    stoich = None
    for prefix, use_v, use_zpe, label in branches:
        cfg = GCConfig(
            **common,
            include_v_product=use_v,
            include_v_mu=use_v,
            include_zpe_product=use_zpe,
            include_zpe_mu=use_zpe,
            include_decomposition=False,
        )
        print(f"[gc_variants] {label}")
        result = compute_gc(
            structures_file,
            pressure_file,
            config=cfg,
            save_outputs=False,
        )
        if stoich is None:
            stoich = result["stoich"]
            out["Ts"] = result["Ts"]
            out["Ps"] = result["Ps"]
            out["stoich"] = stoich
            out["degeneracy"] = result["degeneracy"]
        out.update(_branch_out(prefix, result, stoich))
        del result

    np.savez(output_npz, **out)
    print(f"[gc_variants] saved: {output_npz}")
