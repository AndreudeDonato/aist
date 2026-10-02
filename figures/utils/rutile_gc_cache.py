"""Dedicated GC cache: IrO₂ rutile + IrO₂(g) + IrO₃(g) only."""
from __future__ import annotations

import os

import numpy as np
from ase.io import read, write

from aist.core.grandcanonical import run_gc

from figures.config import (
    AMBIENT_PO2_PA,
    IR_REF_PBE,
    IR_REF_PBE0,
    IRO2_RUTILE_PBE,
    IRO2_RUTILE_PBE0,
    O2_RUTILE_PBE,
    O2_RUTILE_PBE0,
    RUTILE_GC_CACHE_DIR,
    RUTILE_GC_CALIBRATION_T_K,
    RUTILE_GC_GAS_P,
    RUTILE_GC_GAS_USE_VIB,
    RUTILE_GC_GAS_INCLUDE_ROT_TRANS,
    RUTILE_GC_N_P,
    RUTILE_GC_N_T,
    RUTILE_GC_NPZ,
    RUTILE_GC_P_MAX,
    RUTILE_GC_P_MIN,
    RUTILE_GC_STRUCTURES,
    RUTILE_GC_T_MAX,
    RUTILE_GC_T_MIN,
    gc_rutile_cache_config,
    gc_rutile_gas_specs,
)
from figures.utils.cache import _any_newer_than
from figures.utils.gc_gas_oxides import merge_gas_oxides, patch_gas_reservoir_thermo, patch_gas_rot_trans
from figures.utils.cache_stamp import gc_config_settings, make_stamp, npz_stamp, write_npz_stamp
from figures.utils.rutile_gc_calibrate import RutileCalibrationAnchor, rutile_minus_iro3_at_anchor


def _rutile_dataset_specs(label: str) -> dict:
    if label == "PBE":
        return {
            "rutile": IRO2_RUTILE_PBE,
            "ir_ref": IR_REF_PBE,
            "o2_traj": O2_RUTILE_PBE,
            "output_npz": RUTILE_GC_NPZ["PBE"],
            "structures_traj": RUTILE_GC_STRUCTURES["PBE"],
        }
    return {
        "rutile": IRO2_RUTILE_PBE0,
        "ir_ref": IR_REF_PBE0,
        "o2_traj": O2_RUTILE_PBE0,
        "output_npz": RUTILE_GC_NPZ["PBE0"],
        "structures_traj": RUTILE_GC_STRUCTURES["PBE0"],
    }


def _write_rutile_base_traj(rutile_path: str, output_path: str) -> None:
    rutile = read(rutile_path)
    if not isinstance(rutile, list):
        rutile = [rutile]
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    write(output_path, rutile)


def _stored_rutile_gas_p(path: str) -> float | None:
    if not os.path.exists(path):
        return None
    with np.load(path) as data:
        if "rutile_gas_P_Pa" in data.files:
            return float(data["rutile_gas_P_Pa"])
        if "rutile_gas_P_IRO3_Pa" in data.files:
            return float(data["rutile_gas_P_IRO3_Pa"])
    return None


def _stored_rutile_gas_use_vib(path: str) -> bool | None:
    if not os.path.exists(path):
        return None
    with np.load(path) as data:
        if "rutile_gas_use_vib" not in data.files:
            return None
        return bool(data["rutile_gas_use_vib"])
    return None


def _stored_rutile_gas_include_rot_trans(path: str) -> bool | None:
    if not os.path.exists(path):
        return None
    with np.load(path) as data:
        if "rutile_gas_include_rot_trans" not in data.files:
            return None
        return bool(data["rutile_gas_include_rot_trans"])
    return None


def _rutile_npz_stale(path: str) -> bool:
    if not os.path.exists(path):
        return True
    with np.load(path) as data:
        ts, ps = data["Ts"], data["Ps"]
        if int(ts.size) != RUTILE_GC_N_T or int(ps.size) != RUTILE_GC_N_P:
            return True
        if not np.isclose(float(ts[0]), RUTILE_GC_T_MIN) or not np.isclose(float(ts[-1]), RUTILE_GC_T_MAX):
            return True
        if not np.isclose(float(ps[0]), RUTILE_GC_P_MIN, rtol=1e-3) or not np.isclose(
            float(ps[-1]), RUTILE_GC_P_MAX, rtol=1e-3
        ):
            return True
        if "rutile_gas_P_Pa" not in data.files:
            return True
        if not np.isclose(float(data["rutile_gas_P_Pa"]), RUTILE_GC_GAS_P, rtol=1e-6):
            return True
        if "rutile_calibration_dG_eV" not in data.files:
            return True
    vib_stored = _stored_rutile_gas_use_vib(path)
    if vib_stored is None or vib_stored != RUTILE_GC_GAS_USE_VIB:
        return True
    rt_stored = _stored_rutile_gas_include_rot_trans(path)
    if rt_stored is None or rt_stored != RUTILE_GC_GAS_INCLUDE_ROT_TRANS:
        return True
    return False


def _gas_rows_with_pressure(
    gas_rows: list[tuple[int, object, float]],
    p_gas: float,
) -> list[tuple[int, object, float]]:
    return [(row, atoms, p_gas) for row, atoms, _ in gas_rows]


def _anchor_delta_g(output_npz: str) -> float:
    """G_rutile − G_IrO₃(g) at the reference anchor, from the patched rutile GC (what the map shows)."""
    with np.load(output_npz, allow_pickle=True) as data:
        g = np.asarray(data["G_tensor"], dtype=float)
        ts = np.asarray(data["Ts"], dtype=float)
        ps = np.asarray(data["Ps"], dtype=float)
        formulas = data["formulas"]
    anchor = RutileCalibrationAnchor(T_K=RUTILE_GC_CALIBRATION_T_K, P_O2_Pa=AMBIENT_PO2_PA)
    return rutile_minus_iro3_at_anchor(g, ps, ts, formulas, anchor=anchor)


def _write_rutile_gas_metadata(
    npz_path: str,
    *,
    label: str,
    p_gas: float,
    gas_use_vib: bool,
    calibration_dg_eV: float,
) -> None:
    data = dict(np.load(npz_path, allow_pickle=True))
    data["rutile_gas_P_Pa"] = np.float32(p_gas)
    data["rutile_gas_P_IRO2_Pa"] = np.float32(p_gas)
    data["rutile_gas_P_IRO3_Pa"] = np.float32(p_gas)
    data["rutile_gas_use_vib"] = np.bool_(gas_use_vib)
    data["rutile_gas_include_rot_trans"] = np.bool_(RUTILE_GC_GAS_INCLUDE_ROT_TRANS)
    data["rutile_dataset"] = np.array(label)
    data["rutile_anchor_T_K"] = np.float32(RUTILE_GC_CALIBRATION_T_K)
    data["rutile_anchor_pO2_Pa"] = np.float32(AMBIENT_PO2_PA)
    data["rutile_calibration_dG_eV"] = np.float32(calibration_dg_eV)
    np.savez(npz_path, **data)


def _ensure_rutile_dataset(*, label: str, force: bool) -> None:
    spec = _rutile_dataset_specs(label)
    rutile_path = spec["rutile"]
    if not os.path.exists(rutile_path):
        raise FileNotFoundError(f"Missing rutile structure: {rutile_path}")

    base_traj = spec["structures_traj"]
    merged_path = os.path.join(RUTILE_GC_CACHE_DIR, f"structures_{label}_with_gas.traj")
    _write_rutile_base_traj(rutile_path, base_traj)

    gas_specs = gc_rutile_gas_specs(label)
    gc_structures, gas_rows = merge_gas_oxides(base_traj, gas_specs, merged_path)

    sources = [rutile_path, spec["o2_traj"], spec["ir_ref"]]
    sources += [p for p, _ in gas_specs]
    output_npz = spec["output_npz"]
    gas_use_vib = RUTILE_GC_GAS_USE_VIB
    cfg = gc_rutile_cache_config(reference_file=spec["ir_ref"], use_vib_zpe=False)
    stamp = make_stamp("rutile", inputs=sources, label=label, config=gc_config_settings(cfg),
                       gas_p=float(RUTILE_GC_GAS_P), gas_vib=gas_use_vib, gas_rot_trans=RUTILE_GC_GAS_INCLUDE_ROT_TRANS)
    stale = npz_stamp(output_npz) != stamp or _rutile_npz_stale(output_npz)
    p_stored = _stored_rutile_gas_p(output_npz)

    if not force and not stale and os.path.exists(output_npz) and not _any_newer_than(
        sources, [output_npz]
    ):
        vib_tag = "+vib gas" if gas_use_vib else "novib gas"
        print(
            f"[cache] rutile GC {label} up to date "
            f"({RUTILE_GC_N_T}×{RUTILE_GC_N_P}, "
            f"T={RUTILE_GC_T_MIN:.0f}–{RUTILE_GC_T_MAX:.0f} K, "
            f"P_gas={p_stored:.3g} Pa, {vib_tag})"
        )
        return

    os.makedirs(RUTILE_GC_CACHE_DIR, exist_ok=True)
    vib_tag = "+vib gas" if gas_use_vib else "novib gas"
    print(
        f"[cache] rutile GC {label}, grid {RUTILE_GC_N_T}×{RUTILE_GC_N_P} "
        f"(T={RUTILE_GC_T_MIN:.0f}–{RUTILE_GC_T_MAX:.0f} K, {vib_tag}) -> {output_npz}"
    )
    run_gc(
        structures_file=gc_structures,
        pressure_file=spec["o2_traj"],
        output_npz=output_npz,
        output_threshold_dir=os.path.join(RUTILE_GC_CACHE_DIR, f"structures_{label}"),
        output_minima_traj=os.path.join(RUTILE_GC_CACHE_DIR, f"phase_minima_rutile_{label}.traj"),
        config=cfg,
    )
    if gas_rows:
        p_gas = float(RUTILE_GC_GAS_P)
        rows = _gas_rows_with_pressure(gas_rows, p_gas)
        if gas_use_vib:
            patch_gas_reservoir_thermo(output_npz, rows, use_vib_zpe=True)
        elif RUTILE_GC_GAS_INCLUDE_ROT_TRANS:
            patch_gas_rot_trans(output_npz, rows, use_vib_zpe=False)
        delta_g = _anchor_delta_g(output_npz)
        print(
            f"[cache] rutile GC {label}: P_gas={p_gas:.6g} Pa "
            f"(ΔG={delta_g:.4f} eV at T={RUTILE_GC_CALIBRATION_T_K:.0f} K, "
            f"p_O2={AMBIENT_PO2_PA:.3g} Pa)"
        )
        _write_rutile_gas_metadata(
            output_npz,
            label=label,
            p_gas=p_gas,
            gas_use_vib=gas_use_vib,
            calibration_dg_eV=delta_g,
        )
    write_npz_stamp(output_npz, stamp)


def ensure_rutile_gc(*, force: bool = False) -> None:
    """Build GC NPZ for IrO₂ rutile + gas oxides (PBE and PBE0)."""
    for label in ("PBE", "PBE0"):
        _ensure_rutile_dataset(label=label, force=force)
