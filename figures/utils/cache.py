"""Build and validate derived artifacts from ``data/`` only."""
from __future__ import annotations

import os
from typing import Iterable

import numpy as np

from figures.config import (
    ACE_ATOM_IR_NPZ,
    ACE_ATOM_O_NPZ,
    ACE_FEATURES,
    ACE_ROOT,
    ACE_STRUCT_NPZ,
    CACHE_DIR,
    GC_NPZ,
    GC_NPZ_PBE0,
    GC_N_P,
    GC_N_T,
    GC_STRUCTURES,
    GC_USE_VIB_ZPE_BY_DATASET,
    GC_VARIANTS_NPZ,
    MOTIF_JSON,
    IR_REF_PBE,
    IR_REF_PBE0,
    O2_TRAJ_PBE,
    O2_TRAJ_PBE0,
    RAW_DATA_DIR,
    REQUIRED_RAW_FILES,
    TRAJ_PBE,
    TRAJ_PBE0,
    gc_cache_config,
    gc_grid_config,
)
from figures.utils.validation import validate_cache_for_figures

from aist.ace.pipeline import main as run_ace_pipeline
from aist.core.grandcanonical import run_gc
from aist.core.gc_variants import run_gc_variants


def _mtime(path: str) -> float:
    return os.path.getmtime(path) if os.path.exists(path) else 0.0


def _any_newer_than(sources: Iterable[str], outputs: Iterable[str]) -> bool:
    if not all(os.path.exists(p) for p in outputs):
        return True
    t_out = min(_mtime(p) for p in outputs)
    return any(_mtime(s) > t_out for s in sources)


def _gc_npz_grid_mismatch(path: str) -> bool:
    if not os.path.exists(path):
        return True
    data = np.load(path)
    return int(data["Ts"].size) != GC_N_T or int(data["Ps"].size) != GC_N_P


def ensure_raw_data() -> None:
    missing = [p for p in REQUIRED_RAW_FILES if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            f"Missing raw data under {RAW_DATA_DIR}:\n  " + "\n  ".join(missing)
        )
    print(f"[cache] raw data OK ({len(REQUIRED_RAW_FILES)} files in data/)")


def ensure_ace(*, force: bool = False) -> None:
    """ACE clustering on IryOx PBE + PBE0 (figures 2–3; PBE CSV also used for GC joins)."""
    outputs = [
        ACE_FEATURES["PBE"],
        ACE_FEATURES["PBE0"],
        ACE_ATOM_IR_NPZ,
        ACE_ATOM_O_NPZ,
        ACE_STRUCT_NPZ,
        MOTIF_JSON["pooled"],
    ]
    sources = [TRAJ_PBE, TRAJ_PBE0]
    if not force and not _any_newer_than(sources, outputs):
        print("[cache] ACE clustering up to date")
        return

    os.makedirs(ACE_ROOT, exist_ok=True)
    print(f"[cache] ACE clustering (IryOx PBE+PBE0) -> {ACE_ROOT}")
    run_ace_pipeline(
        results_dir=ACE_ROOT,
        data_dir=RAW_DATA_DIR,
        write_plots=False,
    )


def _ensure_gc_dataset(
    *,
    label: str,
    structures_file: str,
    o2_traj: str,
    ir_ref: str,
    output_npz: str,
    force: bool,
) -> None:
    sources = [structures_file, o2_traj, ir_ref]
    stale = _gc_npz_grid_mismatch(output_npz)
    if not force and not stale and os.path.exists(output_npz) and not _any_newer_than(
        sources, [output_npz]
    ):
        print(f"[cache] grand-canonical {label} up to date ({GC_N_T}×{GC_N_P} grid)")
        return

    os.makedirs(CACHE_DIR, exist_ok=True)
    print(f"[cache] grand-canonical {label}, grid {GC_N_T}×{GC_N_P} -> {output_npz}")
    use_vib = GC_USE_VIB_ZPE_BY_DATASET.get(label, False)
    run_gc(
        structures_file=structures_file,
        pressure_file=o2_traj,
        output_npz=output_npz,
        output_threshold_dir=os.path.join(CACHE_DIR, f"gc_structures_{label}"),
        output_minima_traj=os.path.join(CACHE_DIR, f"phase_minima_{label}.traj"),
        config=gc_cache_config(reference_file=ir_ref, use_vib_zpe=use_vib),
    )


def ensure_gc(*, force: bool = False) -> None:
    _ensure_gc_dataset(
        label="PBE",
        structures_file=GC_STRUCTURES,
        o2_traj=O2_TRAJ_PBE,
        ir_ref=IR_REF_PBE,
        output_npz=GC_NPZ,
        force=force,
    )
    _ensure_gc_dataset(
        label="PBE0",
        structures_file=TRAJ_PBE0,
        o2_traj=O2_TRAJ_PBE0,
        ir_ref=IR_REF_PBE0,
        output_npz=GC_NPZ_PBE0,
        force=force,
    )


def ensure_gc_variants(*, force: bool = False) -> None:
    sources = [GC_STRUCTURES, O2_TRAJ_PBE, IR_REF_PBE]
    stale = _gc_npz_grid_mismatch(GC_VARIANTS_NPZ)
    if not force and not stale and os.path.exists(GC_VARIANTS_NPZ) and not _any_newer_than(
        sources, [GC_VARIANTS_NPZ]
    ):
        print(f"[cache] GC variants up to date ({GC_N_T}×{GC_N_P} grid)")
        return

    print(f"[cache] GC variants PBE, grid {GC_N_T}×{GC_N_P} -> {GC_VARIANTS_NPZ}")
    run_gc_variants(
        GC_STRUCTURES,
        O2_TRAJ_PBE,
        GC_VARIANTS_NPZ,
        grid_config=gc_grid_config(reference_file=IR_REF_PBE),
    )


_FIG_NEEDS_ACE = {2, 3, 4, 5, 6, 9, 10}
_FIG_NEEDS_GC = {5, 6, 7, 8, 10, 11}
_FIG_NEEDS_GC_VARIANTS = {5, 7, 10}


def ensure_for_figures(fig_nums: Iterable[int], *, force: bool = False) -> None:
    ensure_raw_data()
    nums = set(fig_nums)
    if nums & _FIG_NEEDS_ACE:
        ensure_ace(force=force)
    if nums & _FIG_NEEDS_GC:
        ensure_gc(force=force)
    if nums & _FIG_NEEDS_GC_VARIANTS:
        ensure_gc_variants(force=force)
    validate_cache_for_figures(nums)
