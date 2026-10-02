"""Input validation for paper figure scripts."""
from __future__ import annotations

import os

import numpy as np


def check_file(path: str, label: str = "") -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required file missing{' (' + label + ')' if label else ''}: {path}")


def _n_gc_expected(npz_path: str, traj_path: str, n_gas: int) -> int:
    """GC rows expected: from the frame sidecar when the ensemble is a subset of the trajectory."""
    side = npz_path[:-4] + "_frames.npy"
    if os.path.exists(side):
        return int(np.load(side).size)
    from figures.utils.io import load_traj
    return len(list(load_traj(traj_path))) + n_gas


def check_gc_ace_alignment(gc_data: dict, ace_rows: list, dataset: str) -> None:
    from aist.ace.join import align_ace_rows_to_gc
    ace_rows = align_ace_rows_to_gc(gc_data, ace_rows)
    n_gc = gc_data["Pgi"].shape[0]
    n_ace = len(ace_rows)
    if n_gc != n_ace:
        raise ValueError(
            f"[{dataset}] GC has {n_gc} structures, ACE features has {n_ace} rows. "
            "Re-run: python3 figures/build_all.py --rebuild-cache"
        )


def check_gc_grid(path: str, n_t: int, n_p: int) -> None:
    data = np.load(path)
    if int(data["Ts"].size) != n_t or int(data["Ps"].size) != n_p:
        raise ValueError(
            f"{path} grid is {data['Ts'].size}×{data['Ps'].size}, "
            f"expected {n_t}×{n_p}. Rebuild with --rebuild-cache."
        )


def check_gc_variants_npz(path: str) -> None:
    """Ensure variant NPZ includes full ``Pgi`` tensors for GC variant figures."""
    check_file(path, "gc_variants")
    data = np.load(path)
    required = (
        "Ts", "Ps", "full_Pgi", "novib_Pgi", "vib_nozpe_Pgi", "zpe_only_Pgi",
        "full_phase_index", "novib_phase_index", "vib_nozpe_phase_index",
        "zpe_only_phase_index",
        "novib_G_tensor", "full_G_tensor", "vib_nozpe_G_tensor", "zpe_only_G_tensor",
    )
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(
            f"gc_variants.npz is missing keys {missing}. "
            "Rebuild: python3 figures/build_all.py --rebuild-cache"
        )


def validate_cache_for_figures(fig_nums: set[int]) -> None:
    """Post-build checks for the main figures: artifact presence, GC/ACE alignment, variant schema.

    Which caches a figure needs comes from ``figures.utils.cache`` (the sets the cache builder uses).
    """
    from figures.config import (
        ACE_FEATURES,
        GC_NPZ_BY_DATASET,
        GC_N_P,
        GC_N_T,
        GC_VARIANTS_NPZ_BY_DATASET,
        TRAJ_PBE,
        TRAJ_PBE0,
    )
    from figures.utils.cache import _FIG_NEEDS_ACE, _FIG_NEEDS_GC, _FIG_NEEDS_GC_VARIANTS
    from figures.utils.comparison import count_gas_structures
    from figures.utils.io import load_ace_features, load_gc_npz

    nums = set(fig_nums)
    trajs = {"PBE": TRAJ_PBE, "PBE0": TRAJ_PBE0}

    if nums & _FIG_NEEDS_ACE:
        from figures.config import ACE_ATOM_IR_NPZ, ACE_ATOM_O_NPZ, ACE_STRUCT_NPZ, MOTIF_JSON

        for label, path in ACE_FEATURES.items():
            check_file(path, f"ACE features ({label})")
        for path, label in [
            (ACE_ATOM_IR_NPZ, "ACE Ir atom embeddings"),
            (ACE_ATOM_O_NPZ, "ACE O atom embeddings"),
            (ACE_STRUCT_NPZ, "ACE structure embeddings"),
            (MOTIF_JSON["pooled"], "ACE motif_atoms_pooled.json"),
        ]:
            check_file(path, label)

    for label in ("PBE", "PBE0"):
        n_gas = count_gas_structures(label)
        n_expected = _n_gc_expected(GC_NPZ_BY_DATASET[label], trajs[label], n_gas)
        if nums & _FIG_NEEDS_GC:
            path = GC_NPZ_BY_DATASET[label]
            check_file(path, f"grand-canonical {label}")
            check_gc_grid(path, GC_N_T, GC_N_P)
            gc = load_gc_npz(path)
            if gc["Pgi"].shape[0] != n_expected:
                raise ValueError(
                    f"{label} GC NPZ has {gc['Pgi'].shape[0]} structures, expected {n_expected}. Rebuild cache."
                )
            if n_gas == 0 and nums & _FIG_NEEDS_ACE:
                check_gc_ace_alignment(gc, load_ace_features(ACE_FEATURES[label]), label)
        if nums & _FIG_NEEDS_GC_VARIANTS:
            path = GC_VARIANTS_NPZ_BY_DATASET[label]
            check_gc_variants_npz(path)
            check_gc_grid(path, GC_N_T, GC_N_P)
            vd = np.load(path)
            if int(vd["novib_Pgi"].shape[0]) != n_expected:
                raise ValueError(
                    f"{label} gc_variants novib has {vd['novib_Pgi'].shape[0]} structures, "
                    f"expected {n_expected}. Rebuild cache."
                )


def check_vib_availability(structures, dataset: str, required: bool = True) -> int:
    """Return count of structures with frequency data; warn or raise if 0."""
    has_freq = sum(1 for a in structures if a.info.get("frequencies") is not None)
    if has_freq == 0:
        msg = f"[{dataset}] No structures have vibrational frequencies in atoms.info."
        if required:
            raise RuntimeError(msg + " Cannot produce vib-dependent figure.")
        else:
            print(f"[warn] {msg} Vib quantities will be NaN.")
    else:
        print(f"[check] {dataset}: {has_freq}/{len(structures)} structures have frequencies.")
    return has_freq
