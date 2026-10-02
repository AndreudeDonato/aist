"""Build and validate derived artifacts from ``data/`` only."""
from __future__ import annotations

import os
from typing import Iterable

import numpy as np
from ase.io import read, write

from figures.config import (
    ACE_ATOM_IR_NPZ,
    ACE_ATOM_O_NPZ,
    ACE_FEATURES,
    ACE_ROOT,
    ACE_STRUCT_NPZ,
    CACHE_DIR,
    GC_APPLY_SUPPORT_STAB,
    GC_GAS_INCLUDE_ROT_TRANS,
    GC_GAS_INCLUDE_VIB,
    GC_GAS_P,
    GC_NPZ,
    GC_SUPPORT_STAB_EV_PER_IR,
    GC_SUPPORT_STAB_EV_PER_O,
    GC_NPZ_PBE0,
    GC_N_P,
    GC_N_T,
    GC_T_MAX,
    GC_T_MIN,
    GC_STRUCTURES,
    GC_VARIANTS_NPZ_PBE0,
    GC_IR_COUNTS,
    GC_USE_VIB_ZPE_BY_DATASET,
    GC_VARIANTS_NPZ,
    MOTIF_JSON,
    IR_REF_PBE,
    IR_REF_PBE0,
    O2_REF_PBE,
    O2_REF_PBE0,
    RAW_DATA_DIR,
    REQUIRED_RAW_FILES,
    TRAJ_PBE,
    TRAJ_PBE0,
    gc_cache_config,
    gc_gas_reservoir_specs,
    gc_grid_config,
)
from figures.utils.gc_gas_oxides import (
    merge_gas_oxides,
    patch_gas_rot_trans,
    patch_gas_strip_vib_zpe,
    patch_gc_variants_gas,
)
from figures.utils.gc_support import (
    patch_gc_variants_support,
    patch_support_stabilization,
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
    if int(data["Ts"].size) != GC_N_T or int(data["Ps"].size) != GC_N_P:
        return True
    ts = np.asarray(data["Ts"], dtype=float)
    return not (
        np.isclose(float(ts[0]), GC_T_MIN, rtol=0, atol=0.5)
        and np.isclose(float(ts[-1]), GC_T_MAX, rtol=0, atol=0.5)
    )


def _gc_gas_p_stale(path: str, *, label: str) -> bool:
    if not os.path.exists(path):
        return True
    with np.load(path) as data:
        if "gc_gas_P_Pa" not in data.files:
            return True
        return not np.isclose(float(data["gc_gas_P_Pa"]), GC_GAS_P[label], rtol=1e-6)


def _gc_support_stale(path: str) -> bool:
    if not os.path.exists(path):
        return True
    with np.load(path) as data:
        if "gc_support_ev_per_ir" not in data.files or "gc_support_ev_per_o" not in data.files:
            return True
        ev_ir = 0.0 if not GC_APPLY_SUPPORT_STAB else GC_SUPPORT_STAB_EV_PER_IR
        ev_o = 0.0 if not GC_APPLY_SUPPORT_STAB else GC_SUPPORT_STAB_EV_PER_O
        return not (
            np.isclose(float(data["gc_support_ev_per_ir"]), ev_ir, atol=1e-6)
            and np.isclose(float(data["gc_support_ev_per_o"]), ev_o, atol=1e-6)
        )


def _gc_gas_thermo_stale(path: str) -> bool:
    if not os.path.exists(path):
        return True
    with np.load(path) as data:
        for key, expected in (
            ("gc_gas_include_vib", GC_GAS_INCLUDE_VIB),
            ("gc_gas_include_rot_trans", GC_GAS_INCLUDE_ROT_TRANS),
        ):
            if key not in data.files:
                return True
            if bool(data[key]) != expected:
                return True
    return False


def _write_gc_gas_metadata(npz_path: str, *, label: str, p_gas: float) -> None:
    data = dict(np.load(npz_path, allow_pickle=True))
    data["gc_gas_P_Pa"] = np.float32(p_gas)
    data["gc_dataset"] = np.array(label)
    data["gc_gas_include_vib"] = np.bool_(GC_GAS_INCLUDE_VIB)
    data["gc_gas_include_rot_trans"] = np.bool_(GC_GAS_INCLUDE_ROT_TRANS)
    np.savez(npz_path, **data)


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


def _write_gc_frames(output_npz: str, frames: list[int], n_gas: int) -> None:
    """Sidecar ``<npz>_frames.npy``: for every GC row, its frame index in the data/ trajectory (-1 = gas oxide).
    Loaders attach it as ``source_frames`` so joins with per-structure data (ACE rows, inertia, gaps)
    pick structures by frame instead of by position."""
    np.save(output_npz[:-4] + "_frames.npy", np.asarray(list(frames) + [-1] * n_gas, dtype=int))


def _has_frequencies(atoms) -> bool:
    freqs = atoms.info.get("frequencies")
    return freqs is not None and len(freqs) > 0


def _gc_selection(label: str, frames) -> list[int]:
    """Frames that enter ``label``'s grand-canonical ensembles: Ir sizes in ``GC_IR_COUNTS`` and, with
    vibrations on for ``label``, only structures that carry frequencies."""
    vib = GC_USE_VIB_ZPE_BY_DATASET.get(label, False)
    return [
        i for i, a in enumerate(frames)
        if a.get_chemical_symbols().count("Ir") in GC_IR_COUNTS and (not vib or _has_frequencies(a))
    ]


def _gc_kept_frames(label: str, structures_file: str) -> list[int]:
    return _gc_selection(label, read(structures_file, index=":"))


def _gc_frames_stale(output_npz: str, kept: list[int], n_gas: int) -> bool:
    """True when the cache was built from another selection of frames (or has no sidecar)."""
    path = output_npz[:-4] + "_frames.npy"
    if not os.path.exists(path):
        return True
    return np.load(path).tolist() != list(kept) + [-1] * n_gas


def _gc_input_structures(label: str, structures_file: str) -> str:
    """Trajectory of the structures that enter ``label``'s grand-canonical ensembles (see ``_gc_selection``).

    Structures of those Ir sizes left out for lack of frequencies are listed in
    ``cache/gc_excluded_<label>.tsv`` (aist would otherwise add them without vibrational and rotational terms).
    """
    frames = read(structures_file, index=":")
    keep = _gc_selection(label, frames)
    if len(keep) == len(frames):
        return structures_file
    out = os.path.join(CACHE_DIR, f"gc_input_{label}.traj")
    listing = os.path.join(CACHE_DIR, f"gc_excluded_{label}.tsv")
    in_size = [a for a in frames if a.get_chemical_symbols().count("Ir") in GC_IR_COUNTS]
    no_freq = [a for a in in_size if not _has_frequencies(a)] if GC_USE_VIB_ZPE_BY_DATASET.get(label, False) else []
    os.makedirs(CACHE_DIR, exist_ok=True)
    write(out, [frames[i] for i in keep])
    with open(listing, "w") as fh:
        fh.write("id\tsource_id\tformula\tenergy_eV\tvib_status\n")
        for a in no_freq:
            fh.write(f"{a.info.get('id', '')}\t{a.info.get('source_id', '')}\t{a.get_chemical_formula()}\t"
                     f"{a.get_potential_energy():.6f}\t{a.info.get('vib_status', '')}\n")
    sizes = "".join(f"Ir{n}" if i == 0 else f"/{n}" for i, n in enumerate(sorted(GC_IR_COUNTS)))
    print(f"[cache] {label}: {len(keep)} of {len(frames)} structures enter the grand-canonical ensembles "
          f"({sizes}: {len(in_size)} structures, {len(no_freq)} of them without frequencies, listed in "
          f"{os.path.relpath(listing, CACHE_DIR)})")
    return out


def _ensure_gc_dataset(
    *,
    label: str,
    structures_file: str,
    o2_traj: str,
    ir_ref: str,
    output_npz: str,
    force: bool,
) -> None:
    merged_path = os.path.join(CACHE_DIR, f"gc_structures_{label}_with_gas.traj")
    gc_structures, gas_rows = merge_gas_oxides(
        _gc_input_structures(label, structures_file), gc_gas_reservoir_specs(label), merged_path
    )
    sources = [structures_file, o2_traj, ir_ref] + [p for p, _ in gc_gas_reservoir_specs(label)]
    kept = _gc_kept_frames(label, structures_file)
    stale = (
        _gc_npz_grid_mismatch(output_npz)
        or _gc_gas_p_stale(output_npz, label=label)
        or _gc_support_stale(output_npz)
        or _gc_gas_thermo_stale(output_npz)
        or _gc_frames_stale(output_npz, kept, len(gas_rows))
    )
    p_gas = float(GC_GAS_P[label])
    if not force and not stale and os.path.exists(output_npz) and not _any_newer_than(
        sources, [output_npz]
    ):
        print(f"[cache] grand-canonical {label} up to date ({GC_N_T}×{GC_N_P} grid, P_gas={p_gas:.3g} Pa)")
        return

    os.makedirs(CACHE_DIR, exist_ok=True)
    print(f"[cache] grand-canonical {label}, grid {GC_N_T}×{GC_N_P}, P_gas={p_gas:.3g} Pa -> {output_npz}")
    use_vib = GC_USE_VIB_ZPE_BY_DATASET.get(label, False)
    run_gc(
        structures_file=gc_structures,
        pressure_file=o2_traj,
        output_npz=output_npz,
        output_threshold_dir=os.path.join(CACHE_DIR, f"gc_structures_{label}"),
        output_minima_traj=os.path.join(CACHE_DIR, f"phase_minima_{label}.traj"),
        config=gc_cache_config(reference_file=ir_ref, use_vib_zpe=use_vib),
    )
    _write_gc_frames(output_npz, kept, len(gas_rows))
    if gas_rows:
        if not GC_GAS_INCLUDE_VIB:
            patch_gas_strip_vib_zpe(output_npz, gas_rows)
        if GC_GAS_INCLUDE_ROT_TRANS:
            patch_gas_rot_trans(output_npz, gas_rows, use_vib_zpe=use_vib)
    # Written even without gas rows: the staleness checks above need these keys, and without them
    # the cache was rebuilt on every call when the gas reservoir is off.
    _write_gc_gas_metadata(output_npz, label=label, p_gas=p_gas)
    if GC_APPLY_SUPPORT_STAB:
        patch_support_stabilization(
            output_npz,
            gc_structures,
            [row for row, _, _ in gas_rows],
            ev_per_ir=GC_SUPPORT_STAB_EV_PER_IR,
            ev_per_o=GC_SUPPORT_STAB_EV_PER_O,
        )
    else:
        patch_support_stabilization(
            output_npz,
            gc_structures,
            [row for row, _, _ in gas_rows],
            ev_per_ir=0.0,
            ev_per_o=0.0,
        )


def ensure_gc(*, force: bool = False) -> None:
    _ensure_gc_dataset(
        label="PBE",
        structures_file=GC_STRUCTURES,
        o2_traj=O2_REF_PBE,
        ir_ref=IR_REF_PBE,
        output_npz=GC_NPZ,
        force=force,
    )
    _ensure_gc_dataset(
        label="PBE0",
        structures_file=TRAJ_PBE0,
        o2_traj=O2_REF_PBE0,
        ir_ref=IR_REF_PBE0,
        output_npz=GC_NPZ_PBE0,
        force=force,
    )


def _gc_variants_stale(path: str, label: str = "PBE") -> bool:
    if _gc_npz_grid_mismatch(path):
        return True
    if not os.path.exists(path):
        return True
    with np.load(path) as data:
        if "novib_G_tensor" not in data.files or "full_G_tensor" not in data.files:
            return True
        if "vib_nozpe_G_tensor" not in data.files:
            return True
        if "zpe_only_G_tensor" not in data.files:
            return True
        if "gc_gas_P_Pa" not in data.files:
            return True
        if not np.isclose(float(data["gc_gas_P_Pa"]), GC_GAS_P[label], rtol=1e-6):
            return True
    return _gc_support_stale(path) or _gc_gas_thermo_stale(path)


def _ensure_gc_variants_dataset(*, label: str, structures_file: str, o2_ref: str, ir_ref: str,
                                output_npz: str, force: bool) -> None:
    merged_path = os.path.join(CACHE_DIR, f"gc_structures_{label}_with_gas.traj")
    gc_structures, gas_rows = merge_gas_oxides(
        _gc_input_structures(label, structures_file), gc_gas_reservoir_specs(label), merged_path
    )
    sources = [structures_file, o2_ref, ir_ref] + [p for p, _ in gc_gas_reservoir_specs(label)]
    kept = _gc_kept_frames(label, structures_file)
    stale = _gc_variants_stale(output_npz, label) or _gc_frames_stale(output_npz, kept, len(gas_rows))
    if not force and not stale and os.path.exists(output_npz) and not _any_newer_than(sources, [output_npz]):
        print(f"[cache] GC variants {label} up to date ({GC_N_T}×{GC_N_P} grid)")
        return

    print(f"[cache] GC variants {label}, grid {GC_N_T}×{GC_N_P} -> {output_npz}")
    run_gc_variants(
        gc_structures,
        o2_ref,
        output_npz,
        grid_config=gc_grid_config(reference_file=ir_ref),
    )
    _write_gc_frames(output_npz, kept, len(gas_rows))
    patch_gc_variants_gas(
        output_npz,
        gas_rows,
        include_rot_trans=GC_GAS_INCLUDE_ROT_TRANS,
        strip_vib_zpe=not GC_GAS_INCLUDE_VIB,
    )
    _write_gc_gas_metadata(output_npz, label=label, p_gas=float(GC_GAS_P[label]))
    patch_gc_variants_support(
        output_npz,
        gc_structures,
        [row for row, _, _ in gas_rows],
        ev_per_ir=GC_SUPPORT_STAB_EV_PER_IR if GC_APPLY_SUPPORT_STAB else 0.0,
        ev_per_o=GC_SUPPORT_STAB_EV_PER_O if GC_APPLY_SUPPORT_STAB else 0.0,
    )


def ensure_gc_variants(*, force: bool = False) -> None:
    """Vib / no-vib / ZPE-only variants for both functionals (the vibrational comparison needs both)."""
    _ensure_gc_variants_dataset(label="PBE", structures_file=GC_STRUCTURES, o2_ref=O2_REF_PBE,
                                ir_ref=IR_REF_PBE, output_npz=GC_VARIANTS_NPZ, force=force)
    _ensure_gc_variants_dataset(label="PBE0", structures_file=TRAJ_PBE0, o2_ref=O2_REF_PBE0,
                                ir_ref=IR_REF_PBE0, output_npz=GC_VARIANTS_NPZ_PBE0, force=force)


_FIG_NEEDS_ACE = {2, 3, 4, 6, 9}
_FIG_NEEDS_GC = {3, 4, 5, 7, 8, 9}
_FIG_NEEDS_GC_VARIANTS = {3, 4, 5, 8, 9}
_FIG_NEEDS_RUTILE_GC = {6}
_FIG_NEEDS_FIG06_GC = {6}
_SUPPLEMENTARY_NEEDS_ACE = {
    "supp_atom_hirshfeld",
    "supp_moments_of_inertia",
    "supp_pooled_cluster_pop_inertia",
    "supp_ir_size_gc_maps",
    "supp_ir_size_o_motif_ratio_maps",
    "supp_ir_size_vib_vs_novib_maps",
    "supp_ir_size_oxidation_delta_maps",
    "supp_ir_size_tchar_vs_gvib_maps",
    "supp_gc_clustering_maps",
    "supp_gc_clustering_pressure_bars",
    "supp_vib_contrib_tchar",
    "supp_min_pop_fixedpoints",
    "supp_ranked_pop_by_ir",
    "supp_ir_size_oxidation_at_po2",
    "supp_vib_vs_novib_quadrants",
    "supp_vib_vs_novib_bars",
    "supp_oxidation_delta_by_ir",
    "supp_oxidation_delta_by_ir_low_po2",
}
_SUPPLEMENTARY_NEEDS_GC = {
    "supp_ir_size_gc_maps",
    "supp_ir_size_o_motif_ratio_maps",
    "supp_ir_size_vib_vs_novib_maps",
    "supp_ir_size_oxidation_delta_maps",
    "supp_ir_size_tchar_vs_gvib_maps",
    "supp_gc_clustering_maps",
    "supp_gc_clustering_pressure_bars",
    "supp_vib_contrib_tchar",
    "supp_min_pop_fixedpoints",
    "supp_ranked_pop_by_ir",
    "supp_ir_size_oxidation_at_po2",
    "supp_vib_dof_gvib_contrib",
    "supp_vib_vs_novib_quadrants",
    "supp_vib_vs_novib_bars",
    "supp_oxidation_delta_by_ir",
    "supp_oxidation_delta_by_ir_low_po2",
}
_SUPPLEMENTARY_NEEDS_GC_VARIANTS = {
    "supp_gc_g_vs_mu",
    "supp_ir_size_gc_maps",
    "supp_ir_size_o_motif_ratio_maps",
    "supp_ir_size_vib_vs_novib_maps",
    "supp_ir_size_oxidation_delta_maps",
    "supp_ir_size_tchar_vs_gvib_maps",
    "supp_gc_clustering_maps",
    "supp_gc_clustering_pressure_bars",
    "supp_min_pop_fixedpoints",
    "supp_ranked_pop_by_ir",
    "supp_ir_size_oxidation_at_po2",
    "supp_vib_vs_novib_quadrants",
    "supp_vib_vs_novib_bars",
    "supp_oxidation_delta_by_ir",
    "supp_oxidation_delta_by_ir_low_po2",
}


_FORCED: set[str] = set()


def _force_once(name: str, force: bool) -> bool:
    """Honour ``force`` only the first time a cache is requested in this process (build_all asks for the
    figure caches and then for the supplementary ones; a forced rebuild should run once)."""
    if not force or name in _FORCED:
        return False
    _FORCED.add(name)
    return True


def ensure_for_supplementary(names: Iterable[str], *, force: bool = False) -> None:
    ensure_raw_data()
    names = set(names)
    if names & _SUPPLEMENTARY_NEEDS_ACE:
        ensure_ace(force=_force_once("ace", force))
    if names & _SUPPLEMENTARY_NEEDS_GC:
        ensure_gc(force=_force_once("gc", force))
    if names & _SUPPLEMENTARY_NEEDS_GC_VARIANTS:
        ensure_gc_variants(force=_force_once("gc_variants", force))
    if "supp_rutile_gc_phase_maps" in names:
        from figures.utils.rutile_gc_cache import ensure_rutile_gc
        ensure_rutile_gc(force=_force_once("rutile_gc", force))


def ensure_for_figures(fig_nums: Iterable[int], *, force: bool = False) -> None:
    ensure_raw_data()
    nums = set(fig_nums)
    if nums & _FIG_NEEDS_ACE:
        ensure_ace(force=_force_once("ace", force))
    if nums & _FIG_NEEDS_GC:
        ensure_gc(force=_force_once("gc", force))
    if nums & _FIG_NEEDS_GC_VARIANTS:
        ensure_gc_variants(force=_force_once("gc_variants", force))
    if nums & _FIG_NEEDS_RUTILE_GC:
        from figures.utils.rutile_gc_cache import ensure_rutile_gc
        ensure_rutile_gc(force=_force_once("rutile_gc", force))
    if nums & _FIG_NEEDS_FIG06_GC:
        from figures.utils.fig06_gc_cache import ensure_fig06_gc
        ensure_fig06_gc(force=_force_once("fig06_gc", force))
    validate_cache_for_figures(nums)
