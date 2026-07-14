"""Per-Ir-size ΔG contribution lines vs T (Fig. 8 top-left, split by ``n_Ir``)."""
from __future__ import annotations

import numpy as np
from ase import Atoms
from ase.io import read

from aist.composedthermo import compute_thermo_contributions
from aist.core.grandcanonical import (
    GC_CONTRIBUTION_SERIES,
    GCConfig,
    pressure_grid_index,
)
from aist.stoichiometry import balance_reaction
from aist.utils import get_energy

from figures.config import IR_REF_PBE, SUPP_IR_COUNTS, VIB_O2_REF, gc_grid_config
from figures.utils.gc_subset import _renorm_pgi_slab, ir_size_reservoir_mask
from figures.utils.vib_metrics import tchar_energy_weighted_ensemble_map


def _g_from_contrib(contrib_dict, Tg, e_key: str, s_key: str | None = None):
    if s_key is None:
        s_key = "s" + e_key[1:]
    return contrib_dict.get(e_key, 0.0) - Tg * contrib_dict.get(s_key, 0.0)


def _n_ir_n_o(atoms: Atoms) -> tuple[int, int]:
    syms = atoms.get_chemical_symbols()
    return sum(1 for s in syms if s == "Ir"), sum(1 for s in syms if s == "O")


def compute_per_structure_contribution_slab(
    structures: list[Atoms],
    Ts: np.ndarray,
    Ps: np.ndarray,
    p_idx: int,
    *,
    config: GCConfig | None = None,
    o2_path: str = VIB_O2_REF,
) -> dict[str, np.ndarray]:
    """Per-structure ΔG contribution vs ``T`` at fixed ``p_idx`` (shape ``n_s × n_T``)."""
    cfg = config or gc_grid_config(reference_file=IR_REF_PBE, include_decomposition=True)
    Ts = np.asarray(Ts, dtype=float)
    Ps = np.asarray(Ps, dtype=float)
    Tg, Pg = np.meshgrid(Ts, Ps, indexing="xy")
    Pg_dilute = np.full_like(Pg, cfg.P_dilute, dtype=float)

    structure_pressure = read(o2_path)
    if not isinstance(structure_pressure, Atoms):
        structure_pressure = structure_pressure[0]
    reference = cfg.reference_structure()

    elec_O2 = [(0.0, 3)] if cfg.include_e_mu else None
    _, mu_contrib = compute_thermo_contributions(
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
    Er_val, ref_contrib = compute_thermo_contributions(
        reference,
        Tg,
        np.full_like(Tg, cfg.P_beam),
        elec_states=None,
        include_v=False,
        include_zpe=False,
        include_r=False,
        include_t=False,
    )
    Eo_b = np.full_like(Pg, Eo, dtype=float)

    keys = [key for _, key in GC_CONTRIBUTION_SERIES]
    slabs: dict[str, list[np.ndarray]] = {key: [] for key in keys}

    for structure in structures:
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
        r_coeffs, p_coeffs = balance_reaction(
            [reference, structure_pressure], [structure],
        )

        e_dft = (
            p_coeffs[0] * get_energy(structure)
            + r_coeffs[0] * Er_val
            + r_coeffs[1] * Eo_b
        )
        e_elec = (
            p_coeffs[0] * _g_from_contrib(contrib, Tg, "eelec", "selec")
            + r_coeffs[0] * _g_from_contrib(ref_contrib, Tg, "eelec", "selec")
            + r_coeffs[1] * _g_from_contrib(mu_contrib, Tg, "eelec", "selec")
        )
        e_vib = (
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
        e_rot = (
            p_coeffs[0] * _g_from_contrib(contrib, Tg, "er", "sr")
            + r_coeffs[0] * _g_from_contrib(ref_contrib, Tg, "er", "sr")
            + r_coeffs[1] * _g_from_contrib(mu_contrib, Tg, "er", "sr")
        )
        e_trans = (
            p_coeffs[0] * _g_from_contrib(contrib, Tg, "et", "st")
            + r_coeffs[0] * _g_from_contrib(ref_contrib, Tg, "et", "st")
            + r_coeffs[1] * _g_from_contrib(mu_contrib, Tg, "et", "st")
        )

        by_key = {
            "E_dft_avg": e_dft,
            "E_elec_avg": e_elec,
            "E_vib_avg": e_vib,
            "E_rot_avg": e_rot,
            "E_trans_avg": e_trans,
        }
        for key in keys:
            slabs[key].append(np.asarray(by_key[key][p_idx, :], dtype=float))

    return {key: np.stack(rows, axis=0) for key, rows in slabs.items()}


def boltzmann_reaction_vib_curves_by_stoichiometry(
    structures: list[Atoms],
    Pgi: np.ndarray,
    Ts: np.ndarray,
    Ps: np.ndarray,
    *,
    pressure_pa: float,
    ir_symbol: str = "Ir",
    o_symbol: str = "O",
) -> tuple[np.ndarray, float, dict[int, list[dict]]]:
    """Boltzmann reaction ``⟨ΔG_vib⟩(T)`` per stoichiometry bucket, grouped by ``n_Ir``.

    Uses the balanced-reaction vibrational contribution (cluster + reference + O₂
    reservoir) with ``Pgi`` renormalized within each ``(n_Ir, n_O)`` bucket.
    """
    from aist.core.vibrational import (
        _renormalized_boltzmann_weights,
        _structure_stoichiometry_groups,
        _weighted_mean,
    )

    Ts = np.asarray(Ts, dtype=float)
    Ps = np.asarray(Ps, dtype=float)
    p_idx, p_used = pressure_grid_index(Ps, pressure_pa)
    n_T = int(Ts.size)

    n_s = min(len(structures), Pgi.shape[0])
    structures = structures[:n_s]
    Pgi = Pgi[:n_s]

    slabs = compute_per_structure_contribution_slab(structures, Ts, Ps, p_idx)
    dg_vib = slabs["E_vib_avg"]

    groups = _structure_stoichiometry_groups(
        structures, ir_symbol=ir_symbol, o_symbol=o_symbol,
    )

    by_ir: dict[int, list[dict]] = {}
    for (n_ir, n_o), idxs in groups.items():
        idx = np.asarray(idxs, dtype=int)
        g_curve = np.empty(n_T, dtype=float)
        for t_idx in range(n_T):
            w = _renormalized_boltzmann_weights(Pgi, idxs, p_idx, t_idx)
            g_curve[t_idx] = _weighted_mean(dg_vib[idx, t_idx], w)
        by_ir.setdefault(int(n_ir), []).append({
            "n_o": int(n_o),
            "stoich": float(n_o) / float(n_ir),
            "dg_vib": g_curve,
        })

    curves_by_ir: dict[int, list[dict]] = {}
    for n_ir, rows in by_ir.items():
        curves_by_ir[n_ir] = sorted(rows, key=lambda r: (float(r["stoich"]), int(r["n_o"])))

    return Ts, float(p_used), curves_by_ir


def compute_per_structure_e_vib_tensor(
    structures: list[Atoms],
    Ts: np.ndarray,
    Ps: np.ndarray,
    *,
    config: GCConfig | None = None,
    o2_path: str = VIB_O2_REF,
) -> np.ndarray:
    """Per-structure vibrational ΔG contribution on the GC grid (``n_s × n_P × n_T``)."""
    cfg = config or gc_grid_config(reference_file=IR_REF_PBE, include_decomposition=True)
    Ts = np.asarray(Ts, dtype=float)
    Ps = np.asarray(Ps, dtype=float)
    Tg, Pg = np.meshgrid(Ts, Ps, indexing="xy")
    Pg_dilute = np.full_like(Pg, cfg.P_dilute, dtype=float)

    structure_pressure = read(o2_path)
    if not isinstance(structure_pressure, Atoms):
        structure_pressure = structure_pressure[0]
    reference = cfg.reference_structure()

    elec_O2 = [(0.0, 3)] if cfg.include_e_mu else None
    _, mu_contrib = compute_thermo_contributions(
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
    Er_val, ref_contrib = compute_thermo_contributions(
        reference,
        Tg,
        np.full_like(Tg, cfg.P_beam),
        elec_states=None,
        include_v=False,
        include_zpe=False,
        include_r=False,
        include_t=False,
    )
    Eo_b = np.full_like(Pg, Eo, dtype=float)

    rows: list[np.ndarray] = []
    for structure in structures:
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
        r_coeffs, p_coeffs = balance_reaction(
            [reference, structure_pressure], [structure],
        )
        e_vib = (
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
        rows.append(np.asarray(e_vib, dtype=float))

    return np.stack(rows, axis=0)


def tchar_map_by_n_ir(
    structures: list[Atoms],
    Pgi: np.ndarray,
    Ts: np.ndarray,
    Ps: np.ndarray,
    n_ir_per_struct: np.ndarray,
    n_ir_target: int,
    formulas: np.ndarray | None = None,
) -> dict | None:
    """Energy-weighted ``⟨Θ_eff⟩`` map for one Ir bucket (cluster isomers only)."""
    mask = ir_size_reservoir_mask(
        n_ir_per_struct, n_ir_target, formulas, include_gas_reservoir=False,
    )
    if not np.any(mask):
        return None

    idx = np.flatnonzero(mask)
    structures_sub = [structures[i] for i in idx]
    Pgi_sub = _renorm_pgi_slab(Pgi, mask)
    tc_map = tchar_energy_weighted_ensemble_map(structures_sub, Pgi_sub, Ts)
    out: dict = {
        "Ts": Ts,
        "Ps": Ps,
        "tchar_map": tc_map,
        "phase_index": np.argmax(Pgi_sub, axis=0),
        "n_struct": int(mask.sum()),
    }
    if formulas is not None:
        out["formulas"] = np.asarray(formulas[mask], dtype=object)
    return out


def e_vib_avg_map_by_n_ir(
    e_vib_tensor: np.ndarray,
    Pgi: np.ndarray,
    Ts: np.ndarray,
    Ps: np.ndarray,
    n_ir_per_struct: np.ndarray,
    n_ir_target: int,
    formulas: np.ndarray | None = None,
) -> dict | None:
    """Ensemble ``⟨G_vib⟩`` map for one Ir bucket (cluster isomers only)."""
    mask = ir_size_reservoir_mask(
        n_ir_per_struct, n_ir_target, formulas, include_gas_reservoir=False,
    )
    if not np.any(mask):
        return None

    Pgi_n = _renorm_pgi_slab(Pgi, mask)
    e_vib_m = np.asarray(e_vib_tensor[mask], dtype=float)
    weight_sum = Pgi_n.sum(axis=0)
    e_vib_avg = np.einsum("ijk,ijk->jk", Pgi_n, e_vib_m)
    with np.errstate(invalid="ignore", divide="ignore"):
        e_vib_avg = np.where(weight_sum > 0, e_vib_avg / weight_sum, np.nan)

    out: dict = {
        "Ts": Ts,
        "Ps": Ps,
        "e_vib_avg": e_vib_avg,
        "phase_index": np.argmax(Pgi_n, axis=0),
        "n_struct": int(mask.sum()),
    }
    if formulas is not None:
        out["formulas"] = np.asarray(formulas[mask], dtype=object)
    return out


def contribution_lines_by_ir_at_pressure(
    gc: dict,
    structures: list[Atoms],
    n_ir_per_struct: np.ndarray,
    pressure_pa: float,
    *,
    ir_counts: list[int] | None = None,
    formulas: np.ndarray | None = None,
) -> tuple[np.ndarray, float, dict[int, list[tuple[str, np.ndarray]]]]:
    """ΔG contribution lines vs ``T`` at fixed ``pO₂``, one set per ``n_Ir`` bucket."""
    Ts = np.asarray(gc["Ts"], dtype=float)
    Ps = np.asarray(gc["Ps"], dtype=float)
    Pgi = np.asarray(gc["Pgi"], dtype=float)
    p_idx, pressure_used = pressure_grid_index(Ps, pressure_pa)

    n_s = min(len(structures), Pgi.shape[0])
    structures = structures[:n_s]
    Pgi = Pgi[:n_s]
    n_ir_per_struct = np.asarray(n_ir_per_struct[:n_s], dtype=int)

    slabs = compute_per_structure_contribution_slab(
        structures, Ts, Ps, p_idx,
    )

    if ir_counts is None:
        ir_counts = list(SUPP_IR_COUNTS)

    out: dict[int, list[tuple[str, np.ndarray]]] = {}
    for n in ir_counts:
        mask = ir_size_reservoir_mask(
            n_ir_per_struct, int(n), formulas, include_gas_reservoir=False,
        )
        if not np.any(mask):
            continue
        Pgi_n = _renorm_pgi_slab(Pgi, mask, p_idx=p_idx)
        curves: list[tuple[str, np.ndarray]] = []
        for label, key in GC_CONTRIBUTION_SERIES:
            contrib = slabs[key][mask]
            y = np.sum(Pgi_n * contrib, axis=0)
            if key == "E_dft_avg":
                y = y - float(np.min(y))
            curves.append((label, y))
        out[int(n)] = curves

    return Ts, pressure_used, out
