"""Fig. 6 cache: PBE0 grand-canonical with IrO₂(g) / IrO₃(g), no support baked in.

Dedicated T–p grid (100–1400 K). Support variants are applied in memory when loading.
"""
from __future__ import annotations

import os

import numpy as np

from aist.core.grandcanonical import run_gc

from figures.config import (
    FIG06_CACHE_DIR,
    FIG06_GC_NPZ_PBE0,
    FIG06_N_P,
    FIG06_N_T,
    FIG06_T_MAX,
    FIG06_T_MIN,
    GC_GAS_INCLUDE_ROT_TRANS,
    GC_GAS_INCLUDE_VIB,
    GC_GAS_OXIDES,
    GC_GAS_P,
    GC_USE_VIB_ZPE_BY_DATASET,
    IR_REF_PBE0,
    O2_REF_PBE0,
    TRAJ_PBE0,
    gc_fig06_cache_config,
)
from figures.utils.cache import (
    _any_newer_than,
    _gc_frames_stale,
    _gc_input_structures,
    _gc_kept_frames,
    _write_gc_frames,
)
from figures.utils.gc_gas_oxides import (
    _reensemble_from_g,
    merge_gas_oxides,
    patch_gas_rot_trans,
    patch_gas_strip_vib_zpe,
)
from figures.utils.gc_support import support_corrections_per_ir
from figures.utils.io import load_npz


def _gas_specs() -> list[tuple[str, float]]:
    return list(GC_GAS_OXIDES["PBE0"])


def _fig06_npz_grid_mismatch(path: str) -> bool:
    if not os.path.exists(path):
        return True
    with np.load(path) as data:
        ts, ps = data["Ts"], data["Ps"]
        if int(ts.size) != FIG06_N_T or int(ps.size) != FIG06_N_P:
            return True
        if not np.isclose(float(ts[0]), FIG06_T_MIN) or not np.isclose(float(ts[-1]), FIG06_T_MAX):
            return True
    return False


def _fig06_npz_stale(path: str) -> bool:
    if _fig06_npz_grid_mismatch(path):
        return True
    if not os.path.exists(path):
        return True
    with np.load(path) as data:
        if "fig06_base" not in data.files or not bool(data["fig06_base"]):
            return True
        if "gc_support_ev_per_ir" not in data.files:
            return True
        if not np.isclose(float(data["gc_support_ev_per_ir"]), 0.0, atol=1e-6):
            return True
        if not np.isclose(float(data["gc_support_ev_per_o"]), 0.0, atol=1e-6):
            return True
        if "gc_gas_P_Pa" not in data.files:
            return True
        if not np.isclose(float(data["gc_gas_P_Pa"]), GC_GAS_P["PBE0"], rtol=1e-6):
            return True
    return False


def _write_fig06_metadata(npz_path: str, *, p_gas: float) -> None:
    data = dict(np.load(npz_path, allow_pickle=True))
    data["gc_gas_P_Pa"] = np.float32(p_gas)
    data["gc_dataset"] = np.array("PBE0")
    data["gc_gas_include_vib"] = np.bool_(GC_GAS_INCLUDE_VIB)
    data["gc_gas_include_rot_trans"] = np.bool_(GC_GAS_INCLUDE_ROT_TRANS)
    data["gc_support_ev_per_ir"] = np.float32(0.0)
    data["gc_support_ev_per_o"] = np.float32(0.0)
    data["fig06_base"] = np.bool_(True)
    data["fig06_T_max_K"] = np.float32(FIG06_T_MAX)
    np.savez(npz_path, **data)


def ensure_fig06_gc(*, force: bool = False) -> None:
    """Build the no-support PBE0 GC NPZ with gas oxides for Fig. 6."""
    merged_path = os.path.join(FIG06_CACHE_DIR, "gc_structures_PBE0_with_gas.traj")
    gas_specs = _gas_specs()
    os.makedirs(FIG06_CACHE_DIR, exist_ok=True)
    gc_structures, gas_rows = merge_gas_oxides(_gc_input_structures("PBE0", TRAJ_PBE0), gas_specs, merged_path)
    sources = [TRAJ_PBE0, O2_REF_PBE0, IR_REF_PBE0] + [p for p, _ in gas_specs]
    output_npz = FIG06_GC_NPZ_PBE0
    kept = _gc_kept_frames("PBE0", TRAJ_PBE0)
    stale = _fig06_npz_stale(output_npz) or _gc_frames_stale(output_npz, kept, len(gas_rows))
    p_gas = float(GC_GAS_P["PBE0"])

    if not force and not stale and os.path.exists(output_npz) and not _any_newer_than(
        sources, [output_npz]
    ):
        print(
            f"[cache] fig06 GC PBE0 up to date "
            f"({FIG06_N_T}×{FIG06_N_P}, T={FIG06_T_MIN:.0f}–{FIG06_T_MAX:.0f} K, "
            f"P_gas={p_gas:.3g} Pa, no support)"
        )
        return

    print(
        f"[cache] fig06 GC PBE0, grid {FIG06_N_T}×{FIG06_N_P} "
        f"(T={FIG06_T_MIN:.0f}–{FIG06_T_MAX:.0f} K), "
        f"P_gas={p_gas:.3g} Pa -> {output_npz}"
    )
    use_vib = GC_USE_VIB_ZPE_BY_DATASET.get("PBE0", False)
    run_gc(
        structures_file=gc_structures,
        pressure_file=O2_REF_PBE0,
        output_npz=output_npz,
        output_threshold_dir=os.path.join(FIG06_CACHE_DIR, "structures_PBE0"),
        output_minima_traj=os.path.join(FIG06_CACHE_DIR, "phase_minima_PBE0.traj"),
        config=gc_fig06_cache_config(reference_file=IR_REF_PBE0, use_vib_zpe=use_vib),
    )
    _write_gc_frames(output_npz, kept, len(gas_rows))
    if gas_rows:
        if not GC_GAS_INCLUDE_VIB:
            patch_gas_strip_vib_zpe(output_npz, gas_rows)
        if GC_GAS_INCLUDE_ROT_TRANS:
            patch_gas_rot_trans(output_npz, gas_rows, use_vib_zpe=use_vib)
    _write_fig06_metadata(output_npz, p_gas=p_gas)


def load_fig06_gc_view(
    *,
    ev_per_ir: float,
    ev_per_o: float,
) -> dict:
    """GC view keyed like ``load_gc_npz``, with the requested support correction."""
    ensure_fig06_gc()
    data = load_npz(
        FIG06_GC_NPZ_PBE0,
        keys=("G_tensor", "Tg", "Ts", "Ps", "stoich", "formulas", "degeneracy"),
    )
    g = np.asarray(data["G_tensor"], dtype=float).copy()
    tg = np.asarray(data["Tg"], dtype=float)
    stoich = np.asarray(data["stoich"], dtype=float)

    merged_path = os.path.join(FIG06_CACHE_DIR, "gc_structures_PBE0_with_gas.traj")
    _, gas_rows = merge_gas_oxides(_gc_input_structures("PBE0", TRAJ_PBE0), _gas_specs(), merged_path)
    gas_idx = [row for row, _, _ in gas_rows]
    corr = support_corrections_per_ir(
        merged_path, gas_idx, ev_per_ir=ev_per_ir, ev_per_o=ev_per_o,
    )
    g -= corr[:, None, None]
    ens = _reensemble_from_g(g, tg, stoich, data["degeneracy"])
    return {
        "Ts": data["Ts"],
        "Ps": data["Ps"],
        "Pgi": ens["Pgi"],
        "phase_index": ens["phase_index"],
        "x_per_non_o": ens["x_per_non_o"],
        "formulas": data["formulas"],
        "degeneracy": data["degeneracy"],
        "source_frames": np.load(FIG06_GC_NPZ_PBE0[:-4] + "_frames.npy"),
    }
