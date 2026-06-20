"""Input validation for paper figure scripts."""
from __future__ import annotations

import os

import numpy as np


def check_file(path: str, label: str = "") -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required file missing{' (' + label + ')' if label else ''}: {path}")


def check_gc_ace_alignment(gc_data: dict, ace_rows: list, dataset: str) -> None:
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
    """Ensure variant NPZ includes full ``Pgi`` tensors for Figures 3 and 8."""
    check_file(path, "gc_variants")
    data = np.load(path)
    required = (
        "Ts", "Ps", "full_Pgi", "novib_Pgi", "full_phase_index",
        "novib_phase_index", "novib_G_tensor",
    )
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(
            f"gc_variants.npz is missing keys {missing}. "
            "Rebuild: python3 figures/build_all.py --rebuild-cache"
        )


def validate_cache_for_figures(fig_nums: set[int]) -> None:
    """Post-build checks: artifact presence, GC/ACE alignment, variant schema."""
    from figures.config import (
        ACE_FEATURES,
        GC_ACE_FEATURES,
        GC_NPZ,
        GC_NPZ_PBE0,
        GC_N_P,
        GC_N_T,
        GC_STRUCTURES,
        GC_VARIANTS_NPZ,
        TRAJ_PBE0,
    )
    from figures.utils.io import load_ace_features, load_gc_npz, load_traj

    needs_ace = bool(fig_nums & {2, 3, 6})
    needs_gc = bool(fig_nums & {3, 6, 7, 8, 9, 10, 12})
    needs_variants = bool(fig_nums & {3, 7, 9, 10, 12})
    needs_gc_join = bool(fig_nums & {9})
    needs_pbe0_gc = bool(fig_nums & {3, 6, 10})

    if needs_ace:
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

    if needs_gc:
        check_file(GC_NPZ, "grand-canonical")
        check_gc_grid(GC_NPZ, GC_N_T, GC_N_P)
        gc = load_gc_npz(GC_NPZ)
        n_traj = len(list(load_traj(GC_STRUCTURES)))
        if gc["Pgi"].shape[0] != n_traj:
            raise ValueError(
                f"GC NPZ has {gc['Pgi'].shape[0]} structures but "
                f"{GC_STRUCTURES} has {n_traj}. Rebuild cache."
            )
        if needs_gc_join:
            rows = load_ace_features(GC_ACE_FEATURES)
            check_gc_ace_alignment(gc, rows, "PBE")

    if needs_pbe0_gc:
        check_file(GC_NPZ_PBE0, "grand-canonical PBE0")
        check_gc_grid(GC_NPZ_PBE0, GC_N_T, GC_N_P)
        gc0 = load_gc_npz(GC_NPZ_PBE0)
        n_traj0 = len(list(load_traj(TRAJ_PBE0)))
        if gc0["Pgi"].shape[0] != n_traj0:
            raise ValueError(
                f"PBE0 GC NPZ has {gc0['Pgi'].shape[0]} structures but "
                f"{TRAJ_PBE0} has {n_traj0}. Rebuild cache."
            )
        rows_pbe = load_ace_features(ACE_FEATURES["PBE"])
        rows_pbe0 = load_ace_features(ACE_FEATURES["PBE0"])
        check_gc_ace_alignment(gc, rows_pbe, "PBE")
        check_gc_ace_alignment(gc0, rows_pbe0, "PBE0")

    if needs_variants:
        check_gc_variants_npz(GC_VARIANTS_NPZ)
        check_gc_grid(GC_VARIANTS_NPZ, GC_N_T, GC_N_P)


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
