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
    from figures.utils.comparison import count_gas_structures
    from figures.utils.io import load_ace_features, load_gc_npz, load_traj

    needs_ace = bool(fig_nums & {2, 3, 6})
    needs_gc = bool(fig_nums & {3, 4, 5, 6, 7, 8, 9})
    needs_variants = bool(fig_nums & {3, 4, 5, 7, 8})
    needs_gc_join = False
    needs_pbe0_gc = bool(fig_nums & {3, 4, 5, 6})

    n_gas_pbe = count_gas_structures("PBE")
    n_gas_pbe0 = count_gas_structures("PBE0")
    n_expected_pbe = len(list(load_traj(GC_STRUCTURES))) + n_gas_pbe
    n_expected_pbe0 = len(list(load_traj(TRAJ_PBE0))) + n_gas_pbe0

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
        if gc["Pgi"].shape[0] != n_expected_pbe:
            raise ValueError(
                f"GC NPZ has {gc['Pgi'].shape[0]} structures, expected {n_expected_pbe}. "
                "Rebuild cache."
            )
        if needs_gc_join and n_gas_pbe == 0:
            rows = load_ace_features(GC_ACE_FEATURES)
            check_gc_ace_alignment(gc, rows, "PBE")

    if needs_pbe0_gc:
        check_file(GC_NPZ_PBE0, "grand-canonical PBE0")
        check_gc_grid(GC_NPZ_PBE0, GC_N_T, GC_N_P)
        gc0 = load_gc_npz(GC_NPZ_PBE0)
        if gc0["Pgi"].shape[0] != n_expected_pbe0:
            raise ValueError(
                f"PBE0 GC NPZ has {gc0['Pgi'].shape[0]} structures, "
                f"expected {n_expected_pbe0}. Rebuild cache."
            )
        if n_gas_pbe == 0:
            check_gc_ace_alignment(
                load_gc_npz(GC_NPZ), load_ace_features(ACE_FEATURES["PBE"]), "PBE",
            )
        if n_gas_pbe0 == 0:
            check_gc_ace_alignment(gc0, load_ace_features(ACE_FEATURES["PBE0"]), "PBE0")

    if needs_variants:
        check_gc_variants_npz(GC_VARIANTS_NPZ)
        check_gc_grid(GC_VARIANTS_NPZ, GC_N_T, GC_N_P)
        vd = np.load(GC_VARIANTS_NPZ)
        if int(vd["novib_Pgi"].shape[0]) != n_expected_pbe:
            raise ValueError(
                f"gc_variants novib has {vd['novib_Pgi'].shape[0]} structures, "
                f"expected {n_expected_pbe}. Rebuild cache."
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
