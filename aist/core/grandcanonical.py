"""Grand-canonical ensembles in the 0.1 file-based API (``compute_gc`` / ``run_gc`` NPZ dictionaries).

These functions keep the names, arguments and outputs of aist 0.1 and are thin layers over the species-agnostic
core in ``aist.ensembles`` (``GrandCanonicalEnsemble``): one reference element read from ``GCConfig.reference_file``
and one single-element reservoir molecule read from ``pressure_file``. The NPZ keys keep their IryOx names
(``x_per_non_o``, ``q_ir_mean``, ...); here "O" means the reservoir element and "Ir" the reference element.
"""
from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
from ase import Atoms
from ase.io import read, write
from scipy.constants import Boltzmann, electron_volt

from aist.inertia import (
    get_hirshfeld_volumes,
    principal_moments_sorted_hirshfeld,
    principal_moments_sorted_mass,
)
from aist.thermo import ThermoPartitionFunctions
from aist.utils import ensure_output_dir, get_energy


def load_reference(path: str) -> Atoms:
    """Load a single-element reference (e.g. one metal atom) from a structure file."""
    ref = read(path)
    if not isinstance(ref, Atoms):
        ref = ref[0]
    if len(set(ref.get_chemical_symbols())) != 1:
        raise ValueError(f"Reference {path!r} must contain one element, got {ref.get_chemical_formula()}")
    return ref


def load_ir_reference(path: str) -> Atoms:
    """Load a single-atom Ir reference from an extxyz file (e.g. ``data/IrPBE.xyz``)."""
    ref = read(path)
    if not isinstance(ref, Atoms):
        ref = ref[0]
    if ref.get_chemical_symbols() != ["Ir"]:
        raise ValueError(
            f"Reference {path!r} must contain one Ir atom, "
            f"got formula {ref.get_chemical_formula()}"
        )
    return ref


@dataclass
class GCConfig:
    """Thermodynamic grid and grand-canonical assembly settings."""

    T_min: float = 100.0
    T_max: float = 1000.0
    n_T: int = 500
    P_min: float = 1e-10
    P_max: float = 1e10
    n_P: int = 500
    include_v_product: bool = True
    include_v_mu: bool = True
    include_zpe_product: bool = True
    include_zpe_mu: bool = True
    include_e_product: bool = True
    include_e_mu: bool = True
    include_r_product: bool = False
    include_r_mu: bool = True
    include_t_product: bool = False
    include_t_mu: bool = True
    P_dilute: float = 1e-2
    P_beam: float = 1e-2
    reference_file: str | None = None
    include_decomposition: bool = True
    population_threshold: float = 0.05
    selection_ranges: list | None = None

    def reference_structure(self) -> Atoms:
        if not self.reference_file:
            raise ValueError(
                "GCConfig.reference_file must be set (e.g. data/IrPBE.xyz or data/IrPBE0.xyz)"
            )
        return load_reference(self.reference_file)


def mean_hirshfeld_charge(atoms, element: str) -> float:
    """Mean Hirshfeld charge of the atoms of ``element`` (``atoms.info['hirshfeld']['charges']``)."""
    h = atoms.info.get("hirshfeld", None)
    if not (isinstance(h, dict) and "charges" in h):
        raise ValueError("Missing atoms.info['hirshfeld']['charges']")
    charges = np.asarray(h["charges"], dtype=float)
    mask = np.asarray(atoms.get_chemical_symbols()) == element
    if not np.any(mask):
        raise ValueError(f"No {element} atoms found for Hirshfeld charge average")
    return float(charges[mask].mean())


def mean_ir_hirshfeld_charge(atoms) -> float:
    return mean_hirshfeld_charge(atoms, "Ir")


def reservoir_from_file(structure: Atoms, cfg: GCConfig):
    """The reservoir molecule of a 0.1-style computation: multiplicity from ``info`` (O2 files without it: triplet)."""
    from aist.contributions import ThermoTerms, _spin_multiplicity
    from aist.ensembles import Reservoir

    terms = ThermoTerms(vibrations=cfg.include_v_mu, zpe=cfg.include_zpe_mu, electronic=cfg.include_e_mu,
                        rotation=cfg.include_r_mu, translation=cfg.include_t_mu)
    mult = _spin_multiplicity(structure)
    if mult is None and structure.get_chemical_formula() == "O2":
        mult = 3  # the 0.1 API's O2 triplet, for O2 files without spin data
    return Reservoir(structure, multiplicity=mult, terms=terms)


def ensemble_from_config(structures, structure_pressure: Atoms, cfg: GCConfig):
    """The ``GrandCanonicalEnsemble`` that ``compute_gc`` evaluates for ``cfg``."""
    from aist.contributions import ThermoTerms
    from aist.ensembles import GrandCanonicalEnsemble, Reference

    reservoir = reservoir_from_file(structure_pressure, cfg)
    if len(reservoir.composition) != 1:
        raise ValueError("compute_gc takes a single-element reservoir molecule (O2, H2, N2, ...); use "
                         "aist.GrandCanonicalEnsemble for others")
    terms = ThermoTerms(vibrations=cfg.include_v_product, zpe=cfg.include_zpe_product,
                        electronic=cfg.include_e_product, rotation=cfg.include_r_product,
                        translation=cfg.include_t_product, pressure=cfg.P_dilute)
    return GrandCanonicalEnsemble(structures, references=[Reference(cfg.reference_structure())],
                                  reservoirs=[reservoir], terms=terms)


def compute_gc(
    structures_file: str,
    pressure_file: str,
    config: GCConfig | None = None,
    *,
    save_outputs: bool = True,
    output_npz: str | None = None,
    output_threshold_dir: str | None = None,
    output_minima_traj: str | None = None,
) -> dict[str, Any]:
    """Compute grand-canonical tensors; optionally write NPZ and auxiliary trajs."""
    cfg = config or GCConfig()
    if cfg.selection_ranges is None:
        cfg.selection_ranges = [
            {
                "name": "range1",
                "T_min": cfg.T_min,
                "T_max": cfg.T_max,
                "P_min": cfg.P_min,
                "P_max": cfg.P_max,
            },
            {"name": "point1", "T": 300, "P": 21227},
            {"name": "point2", "T": 300, "P": 1e-5},
        ]

    structure_pressure = read(pressure_file)
    structures_main = read(structures_file, index=":")
    ens = ensemble_from_config(structures_main, structure_pressure, cfg)
    reservoir = ens.reservoirs[0]
    x_element = next(iter(reservoir.composition))
    ref_element = ens.references[0].element

    Ts = np.linspace(cfg.T_min, cfg.T_max, cfg.n_T)
    Ps = np.logspace(np.log10(cfg.P_min), np.log10(cfg.P_max), cfg.n_P)
    Tg, Pg = np.meshgrid(Ts, Ps, indexing="xy")
    res = ens.evaluate(T=Ts, p={reservoir.name: Ps})
    G_tensor, Pgi, phase_index = res.G, res.populations, res.phase_index

    if save_outputs and output_minima_traj:
        ensure_output_dir(output_minima_traj)
        unique_minima_indices = np.unique(phase_index)
        write(
            output_minima_traj,
            [structures_main[i] for i in unique_minima_indices],
        )

    if save_outputs and output_threshold_dir:

        def _range_indices(values, vmin=None, vmax=None, vpoint=None):
            if vpoint is not None:
                idx = int(np.argmin(np.abs(values - vpoint)))
                return np.array([idx], dtype=int)
            if vmin is None:
                vmin = float(np.min(values))
            if vmax is None:
                vmax = float(np.max(values))
            return np.where((values >= vmin) & (values <= vmax))[0]

        os.makedirs(output_threshold_dir, exist_ok=True)
        for entry in cfg.selection_ranges:
            name = entry.get("name", "selection")
            T_point = entry.get("T", None)
            P_point = entry.get("P", None)
            sel_T_min = entry.get("T_min", None)
            sel_T_max = entry.get("T_max", None)
            sel_P_min = entry.get("P_min", None)
            sel_P_max = entry.get("P_max", None)

            T_idx = _range_indices(Ts, vmin=sel_T_min, vmax=sel_T_max, vpoint=T_point)
            P_idx = _range_indices(Ps, vmin=sel_P_min, vmax=sel_P_max, vpoint=P_point)
            if T_idx.size == 0 or P_idx.size == 0:
                continue

            sub_pop = Pgi[:, P_idx[:, None], T_idx[None, :]]
            mask_struct = np.any(sub_pop >= cfg.population_threshold, axis=(1, 2))
            selected_indices = np.where(mask_struct)[0]
            if selected_indices.size == 0:
                continue

            out_path = os.path.join(
                output_threshold_dir, f"threshold_structures_{name}.traj"
            )
            write(out_path, [structures_main[i] for i in selected_indices])

    tp = ThermoPartitionFunctions()
    q_ir_mean, moments, moments_vol, E0_dft_structures, Eform_dft_structures = [], [], [], [], []
    warned_missing_hirshfeld = warned_missing_charges = False
    E_ref = ens.references[0].energy_per_atom
    for i, structure in enumerate(structures_main):
        try:
            q_ir_mean.append(mean_hirshfeld_charge(structure, ref_element))
        except ValueError:
            if not warned_missing_charges:
                warnings.warn(f"No Hirshfeld charges on {ref_element} for some structures; q_ir_mean is NaN there")
                warned_missing_charges = True
            q_ir_mean.append(np.nan)
        moments.append(principal_moments_sorted_mass(structure, tp))
        volumes = get_hirshfeld_volumes(structure)
        if volumes is None:
            if not warned_missing_hirshfeld:
                print("No Hirshfeld volumes found; using mass-weighted moments of inertia.")
                warned_missing_hirshfeld = True
            moments_vol.append((np.nan, np.nan, np.nan))
        else:
            moments_vol.append(principal_moments_sorted_hirshfeld(structure, volumes))
        E_struct = get_energy(structure)
        E0_dft_structures.append(E_struct)
        Eform_dft_structures.append(ens._p[i] * E_struct + ens._c[i, 0] * E_ref + ens._c[i, 1] * reservoir.energy)

    o_counts = np.array(ens.count(x_element), dtype=float)
    non_o_counts = np.array(ens.normalization, dtype=float)
    ratio_o_non_o = o_counts / non_o_counts
    x_per_non_o = np.sum(Pgi * ratio_o_non_o[:, None, None], axis=0)
    stoich = ratio_o_non_o

    kB = Boltzmann / electron_volt
    G_avg = np.sum(Pgi * G_tensor, axis=0)
    S_conf = -kB * np.sum(Pgi * np.log(Pgi + 1e-20), axis=0)

    q_ir_mean = np.array(q_ir_mean, dtype=float)
    q_ir_avg = np.sum(Pgi * q_ir_mean[:, None, None], axis=0)

    moments = np.array(moments, dtype=float)
    I1, I2, I3 = moments[:, 0], moments[:, 1], moments[:, 2]

    moments_vol = np.array(moments_vol, dtype=float)
    I1_vol, I2_vol, I3_vol = moments_vol[:, 0], moments_vol[:, 1], moments_vol[:, 2]

    data: dict[str, Any] = {
        "Ts": Ts,
        "Ps": Ps,
        "Tg": Tg,
        "Pg": Pg,
        "G_tensor": G_tensor,
        "phase_index": phase_index,
        "Pgi": Pgi,
        "x_values": o_counts,
        "non_o_counts": non_o_counts,
        "x_per_non_o": x_per_non_o,
        "stoich": stoich,
        "formulas": np.array([s.get_chemical_formula() for s in structures_main], dtype=object),
        "q_ir_mean": q_ir_mean,
        "q_ir_avg": q_ir_avg,
        "E0_dft_structures": np.array(E0_dft_structures, dtype=float),
        "Eform_dft_structures": np.array(Eform_dft_structures, dtype=float),
        "I1": I1,
        "I2": I2,
        "I3": I3,
        "I1_I3": I1 / I3,
        "I2_I3": I2 / I3,
        "I1_vol": I1_vol,
        "I2_vol": I2_vol,
        "I3_vol": I3_vol,
        "I1_I3_vol": I1_vol / I3_vol,
        "I2_I3_vol": I2_vol / I3_vol,
        "pop_max": np.max(Pgi, axis=(1, 2)),
        "degeneracy": np.array(ens.degeneracy, dtype=float),
        "G_avg": G_avg,
        "S_conf": S_conf,
        "include_decomposition": np.array([int(cfg.include_decomposition)], dtype=int),
    }

    if cfg.include_decomposition:
        data.update(res.decomposition())

    if save_outputs and output_npz:
        ensure_output_dir(output_npz)
        np.savez(output_npz, **data)
        print(f"Saved results to: {output_npz}")

    return data


def run_gc(
    structures_file: str,
    pressure_file: str,
    output_npz: str,
    *,
    output_threshold_dir: str | None = None,
    output_minima_traj: str | None = None,
    config: GCConfig | None = None,
) -> dict[str, Any]:
    """Run grand-canonical computation and save NPZ (+ optional auxiliary outputs)."""
    return compute_gc(
        structures_file,
        pressure_file,
        config=config,
        save_outputs=True,
        output_npz=output_npz,
        output_threshold_dir=output_threshold_dir,
        output_minima_traj=output_minima_traj,
    )


# Labels and NPZ keys for ensemble-averaged ΔG contribution maps (``include_decomposition``).
GC_CONTRIBUTION_SERIES = (
    ("Bare DFT", "E_dft_avg"),
    ("Electronic", "E_elec_avg"),
    ("Vibrational", "E_vib_avg"),
    ("Rotational", "E_rot_avg"),
    ("Translational", "E_trans_avg"),
)


def temperature_grid_index(Ts: np.ndarray, temperature_k: float) -> tuple[int, float]:
    """Nearest temperature-grid index and the grid value used."""
    Ts = np.asarray(Ts, dtype=float)
    t_idx = int(np.argmin(np.abs(Ts - temperature_k)))
    return t_idx, float(Ts[t_idx])


def pressure_grid_index(Ps: np.ndarray, pressure_pa: float) -> tuple[int, float]:
    """Nearest pressure-grid index (in log p, the grid is logarithmic) and the grid value used."""
    Ps = np.asarray(Ps, dtype=float)
    p_idx = int(np.argmin(np.abs(np.log(Ps) - np.log(pressure_pa))))
    return p_idx, float(Ps[p_idx])


def oxidation_slice_at_pressure(gc_data: dict, pressure_pa: float) -> dict[str, Any]:
    """Ensemble and stable-phase ``n_O/n_Ir`` vs ``T`` at fixed ``pO₂``.

    Matches the left axis of ``plot_grandcanonical`` ``O_per_Ir_and_isomer_pop_vs_T``.
    """
    Ts = np.asarray(gc_data["Ts"], dtype=float)
    Ps = np.asarray(gc_data["Ps"], dtype=float)
    p_idx, pressure_used = pressure_grid_index(Ps, pressure_pa)
    stoich = np.asarray(gc_data["stoich"], dtype=float)
    phase_index = np.asarray(gc_data["phase_index"], dtype=int)
    return {
        "Ts": Ts,
        "pressure_pa": pressure_used,
        "p_idx": p_idx,
        "x_avg": np.asarray(gc_data["x_per_non_o"][p_idx], dtype=float),
        "x_phase": stoich[phase_index[p_idx]],
    }


def isomer_population_stack_at_pressure(
    gc_data: dict,
    pressure_pa: float,
    *,
    population_threshold: float = 0.001,
    color_values: np.ndarray | None = None,
) -> dict[str, Any]:
    """Isomer population stack (%) vs ``T`` at fixed ``pO₂``.

    Returns ``Y`` with shape ``(k, n_T)`` for structures that exceed
    ``population_threshold`` at any temperature, plus per-structure colours from
    ``color_values`` (default: ``stoich``). Same logic as
    ``plot_grandcanonical.build_isomer_stack``.
    """
    Ts = np.asarray(gc_data["Ts"], dtype=float)
    Ps = np.asarray(gc_data["Ps"], dtype=float)
    Pgi = np.asarray(gc_data["Pgi"], dtype=float)
    p_idx, pressure_used = pressure_grid_index(Ps, pressure_pa)

    if color_values is None:
        color_values = np.asarray(gc_data["stoich"], dtype=float)
    else:
        color_values = np.asarray(color_values, dtype=float)

    global_order: list[int] = []
    active_at_T: list[set[int]] = []
    n_T = len(Ts)
    for j in range(n_T):
        pops = Pgi[:, p_idx, j]
        active = {int(i) for i in np.where(pops >= population_threshold)[0]}
        active_at_T.append(active)
        for idx in sorted(active):
            if idx not in global_order:
                global_order.append(idx)

    if not global_order:
        return {
            "Ts": Ts,
            "pressure_pa": pressure_used,
            "p_idx": p_idx,
            "structure_indices": np.array([], dtype=int),
            "Y": None,
            "color_values": None,
        }

    k = len(global_order)
    Y = np.zeros((k, n_T), dtype=float)
    for ki, idx in enumerate(global_order):
        for j in range(n_T):
            if idx in active_at_T[j]:
                Y[ki, j] = 100.0 * Pgi[idx, p_idx, j]

    indices = np.asarray(global_order, dtype=int)
    return {
        "Ts": Ts,
        "pressure_pa": pressure_used,
        "p_idx": p_idx,
        "structure_indices": indices,
        "Y": Y,
        "color_values": color_values[indices],
    }


def contribution_lines_at_pressure(
    gc_data: dict,
    pressure_pa: float,
    *,
    shift_bare_dft: bool = True,
) -> tuple[np.ndarray, float, list[tuple[str, np.ndarray]]]:
    """ΔG contribution lines vs ``T`` at fixed ``pO₂`` (top-left panel of ``G_contributions_3x2``).

    Returns ``(Ts, pressure_used, [(label, values), ...])`` with one curve per contribution.
    """
    missing = [k for _, k in GC_CONTRIBUTION_SERIES if k not in gc_data]
    if missing:
        raise KeyError(
            f"GC NPZ missing decomposition fields {missing}. "
            "Re-run cache with include_decomposition=True."
        )

    Ts = np.asarray(gc_data["Ts"], dtype=float)
    Ps = np.asarray(gc_data["Ps"], dtype=float)
    p_idx, pressure_used = pressure_grid_index(Ps, pressure_pa)

    curves: list[tuple[str, np.ndarray]] = []
    for label, key in GC_CONTRIBUTION_SERIES:
        y = np.asarray(gc_data[key][p_idx], dtype=float).copy()
        if shift_bare_dft and key == "E_dft_avg":
            y -= float(np.min(y))
        curves.append((label, y))

    return Ts, pressure_used, curves


def global_minimum_population_map(gc_data: dict) -> np.ndarray:
    """Population (%) of the global-minimum structure at each (T, P).

    At each grid point, ``phase_index`` is the structure with lowest ``G``; this
    returns ``100 * Pgi[phase_index]`` — same field as
    ``plot_grandcanonical.plot_minima_population``.
    """
    Pgi = np.asarray(gc_data["Pgi"])
    phase_index = np.asarray(gc_data["phase_index"], dtype=int)
    return 100.0 * np.take_along_axis(Pgi, phase_index[None, :, :], axis=0)[0]
