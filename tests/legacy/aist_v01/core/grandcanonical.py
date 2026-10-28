"""Grand-canonical ensemble computation (moved from scripts/compute_grandcanonical.py)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np
from ase import Atoms
from ase.io import read, write
from pymatgen.core.structure import Molecule
from pymatgen.symmetry.analyzer import PointGroupAnalyzer
from scipy.constants import Boltzmann, electron_volt

from aist_v01.composedthermo import compute_thermo_contributions
from aist_v01.inertia import (
    get_hirshfeld_volumes,
    principal_moments_sorted_hirshfeld,
    principal_moments_sorted_mass,
)
from aist_v01.stoichiometry import balance_reaction
from aist_v01.thermo import POINT_GROUP_DEGEN, ThermoPartitionFunctions
from aist_v01.utils import ensure_output_dir, get_energy


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
        return load_ir_reference(self.reference_file)


def mean_ir_hirshfeld_charge(atoms) -> float:
    h = atoms.info.get("hirshfeld", None)
    if not (isinstance(h, dict) and "charges" in h):
        raise ValueError("Missing atoms.info['hirshfeld']['charges']")
    charges = np.asarray(h["charges"], dtype=float)
    symbols = np.asarray(atoms.get_chemical_symbols())
    mask = symbols == "Ir"
    if not np.any(mask):
        raise ValueError("No Ir atoms found for Hirshfeld charge average")
    return float(charges[mask].mean())


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
    reference_structure = cfg.reference_structure()

    Ts = np.linspace(cfg.T_min, cfg.T_max, cfg.n_T)
    Ps = np.logspace(np.log10(cfg.P_min), np.log10(cfg.P_max), cfg.n_P)
    Tg, Pg = np.meshgrid(Ts, Ps, indexing="xy")
    Pg_dilute = np.full_like(Pg, cfg.P_dilute, dtype=float)

    elec_O2 = [(0.0, 3)] if cfg.include_e_mu else None
    mu, mu_contrib = compute_thermo_contributions(
        structure_pressure,
        Tg,
        Pg,
        elec_states=elec_O2,
        include_v=cfg.include_v_mu,
        include_zpe=cfg.include_zpe_mu,
        include_r=cfg.include_r_mu,
        include_t=cfg.include_t_mu,
    )
    Eo = get_energy(structure_pressure)
    E_ir = get_energy(reference_structure)

    Er_val, ref_contrib = compute_thermo_contributions(
        reference_structure,
        Tg,
        cfg.P_beam,
        elec_states=None,
        include_v=False,
        include_zpe=False,
        include_r=False,
        include_t=False,
    )

    tp = ThermoPartitionFunctions()
    warned_missing_hirshfeld = False

    def _g_from(contrib_dict, e_key, s_key=None):
        if s_key is None:
            s_key = "s" + e_key[1:]
        return contrib_dict.get(e_key, 0.0) - Tg * contrib_dict.get(s_key, 0.0)

    G_list = []
    formulas = []
    o_counts = []
    non_o_counts = []
    q_ir_mean = []
    moments = []
    moments_vol = []
    E0_dft_structures = []
    Eform_dft_structures = []
    r_p_coeffs_list = []

    for structure in structures_main:
        symbols = structure.get_chemical_symbols()
        n_M = sum(1 for s in symbols if s != "O")
        n_O = symbols.count("O")

        spin_info = structure.info.get("total_spin", {})
        J = spin_info.get("J", None)
        elec_states = [(0.0, J)] if (cfg.include_e_product and J is not None) else None

        Ep, _ = compute_thermo_contributions(
            structure,
            Tg,
            Pg_dilute,
            elec_states=elec_states,
            include_v=cfg.include_v_product,
            include_zpe=cfg.include_zpe_product,
            include_r=cfg.include_r_product,
            include_t=cfg.include_t_product,
        )

        r_coeffs, p_coeffs = balance_reaction(
            [reference_structure, structure_pressure], [structure]
        )
        r_p_coeffs_list.append((r_coeffs, p_coeffs))

        G = p_coeffs[0] * Ep + r_coeffs[0] * Er_val + r_coeffs[1] * mu
        G_list.append(G.astype(np.float32))

        formulas.append(structure.get_chemical_formula())
        o_counts.append(n_O)
        non_o_counts.append(n_M)
        q_ir_mean.append(mean_ir_hirshfeld_charge(structure))
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
        Eform_dft_structures.append(
            p_coeffs[0] * E_struct + r_coeffs[0] * E_ir + r_coeffs[1] * Eo
        )

    G_tensor = np.stack(G_list, axis=0)

    kB = Boltzmann / electron_volt
    beta = 1.0 / (kB * Tg)
    G_min = np.min(G_tensor, axis=0, keepdims=True)
    weights = np.exp(-beta * (G_tensor - G_min))

    degeneracies = []
    for structure in structures_main:
        mol = Molecule(
            species=structure.get_chemical_symbols(),
            coords=structure.get_positions(),
        )
        pg = PointGroupAnalyzer(mol).sch_symbol
        degeneracies.append(POINT_GROUP_DEGEN.get(pg, 1))
    deg = np.array(degeneracies)[:, None, None]
    weights *= deg

    Xi = np.sum(weights, axis=0)
    Pgi = (weights / Xi).astype(np.float32)

    phase_index = np.argmin(G_tensor, axis=0)

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

    o_counts = np.array(o_counts, dtype=float)
    non_o_counts = np.array(non_o_counts, dtype=float)
    ratio_o_non_o = o_counts / non_o_counts
    x_per_non_o = np.sum(Pgi * ratio_o_non_o[:, None, None], axis=0)
    stoich = ratio_o_non_o

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
        "formulas": np.array(formulas, dtype=object),
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
        "degeneracy": np.array(degeneracies, dtype=float),
        "G_avg": G_avg,
        "S_conf": S_conf,
        "include_decomposition": np.array([int(cfg.include_decomposition)], dtype=int),
    }

    if cfg.include_decomposition:
        E_dft_avg = np.zeros(Tg.shape, dtype=float)
        E_elec_avg = np.zeros(Tg.shape, dtype=float)
        E_vib_avg = np.zeros(Tg.shape, dtype=float)
        E_rot_avg = np.zeros(Tg.shape, dtype=float)
        E_trans_avg = np.zeros(Tg.shape, dtype=float)
        Eo_b = np.full_like(mu, Eo)

        for i, structure in enumerate(structures_main):
            w = Pgi[i].astype(float)
            r_coeffs, p_coeffs = r_p_coeffs_list[i]

            spin_info = structure.info.get("total_spin", {})
            J = spin_info.get("J", None)
            elec_states = [(0.0, J)] if (cfg.include_e_product and J is not None) else None

            _, contrib = compute_thermo_contributions(
                structure,
                Tg,
                Pg_dilute,
                elec_states=elec_states,
                include_v=cfg.include_v_product,
                include_zpe=cfg.include_zpe_product,
                include_r=cfg.include_r_product,
                include_t=cfg.include_t_product,
            )

            E_dft_avg += w * (
                p_coeffs[0] * get_energy(structure)
                + r_coeffs[0] * Er_val
                + r_coeffs[1] * Eo_b
            )
            E_elec_avg += w * (
                p_coeffs[0] * _g_from(contrib, "eelec", "selec")
                + r_coeffs[0] * _g_from(ref_contrib, "eelec", "selec")
                + r_coeffs[1] * _g_from(mu_contrib, "eelec", "selec")
            )
            E_vib_avg += w * (
                p_coeffs[0]
                * (
                    contrib.get("ev", 0.0)
                    + contrib.get("zpe", 0.0)
                    - Tg * contrib.get("sv", 0.0)
                )
                + r_coeffs[0]
                * (
                    ref_contrib.get("ev", 0.0)
                    + ref_contrib.get("zpe", 0.0)
                    - Tg * ref_contrib.get("sv", 0.0)
                )
                + r_coeffs[1]
                * (
                    mu_contrib.get("ev", 0.0)
                    + mu_contrib.get("zpe", 0.0)
                    - Tg * mu_contrib.get("sv", 0.0)
                )
            )
            E_rot_avg += w * (
                p_coeffs[0] * _g_from(contrib, "er", "sr")
                + r_coeffs[0] * _g_from(ref_contrib, "er", "sr")
                + r_coeffs[1] * _g_from(mu_contrib, "er", "sr")
            )
            E_trans_avg += w * (
                p_coeffs[0] * _g_from(contrib, "et", "st")
                + r_coeffs[0] * _g_from(ref_contrib, "et", "st")
                + r_coeffs[1] * _g_from(mu_contrib, "et", "st")
            )

        data.update(
            {
                "E_dft_avg": E_dft_avg,
                "E_elec_avg": E_elec_avg,
                "E_vib_avg": E_vib_avg,
                "E_rot_avg": E_rot_avg,
                "E_trans_avg": E_trans_avg,
            }
        )

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
