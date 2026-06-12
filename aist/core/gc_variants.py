"""Full vs no-vib grand-canonical variants for Figures 5 and 8."""
from __future__ import annotations

import os

import numpy as np

from aist.core.grandcanonical import GCConfig, compute_gc


def run_gc_variants(
    structures_file: str,
    pressure_file: str,
    output_npz: str,
    *,
    grid_config: GCConfig | None = None,
) -> None:
    """Compute and save full/novib GC maps with full ``Pgi`` tensors."""
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
    cfg_full = GCConfig(
        **common,
        include_v_product=True,
        include_v_mu=True,
        include_zpe_product=True,
        include_zpe_mu=True,
        include_decomposition=False,
    )
    cfg_novib = GCConfig(
        **common,
        include_v_product=False,
        include_v_mu=False,
        include_zpe_product=False,
        include_zpe_mu=False,
        include_decomposition=False,
    )

    print("[gc_variants] full (vib + ZPE + ensemble)")
    full = compute_gc(
        structures_file,
        pressure_file,
        config=cfg_full,
        save_outputs=False,
    )

    print("[gc_variants] novib (no vib, no ZPE)")
    novib = compute_gc(
        structures_file,
        pressure_file,
        config=cfg_novib,
        save_outputs=False,
    )

    stoich = full["stoich"]
    np.savez(
        output_npz,
        Ts=full["Ts"],
        Ps=full["Ps"],
        stoich=stoich,
        full_phase_index=full["phase_index"],
        full_x_per_non_o=full["x_per_non_o"],
        full_stoich_map=stoich[full["phase_index"]],
        full_Pgi=full["Pgi"],
        novib_phase_index=novib["phase_index"],
        novib_x_per_non_o=novib["x_per_non_o"],
        novib_stoich_map=stoich[novib["phase_index"]],
        novib_Pgi=novib["Pgi"],
    )
    print(f"[gc_variants] saved: {output_npz}")
