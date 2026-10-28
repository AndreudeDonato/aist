"""Grand-canonical views restricted to one metal-atom count (``n_Ir`` in the IryOx naming).

``n_ir_per_struct`` is the number of non-O atoms of every GC row (``compute_gc`` ``non_o_counts``); gas
reservoir rows are recognised by their formula (``gas_formulas``) and may be kept in every subset.
"""
from __future__ import annotations

import numpy as np

from aist_v01.core.ensemble import lowest_g_index, renormalize_populations
from aist_v01.core.grandcanonical import pressure_grid_index

__all__ = [
    "lowest_g_index",
    "renormalize_populations",
    "gas_formula_mask",
    "ir_size_reservoir_mask",
    "renorm_pgi_at_indices",
    "subset_joined_table_by_n_ir",
    "oxidation_map_by_n_ir",
    "stoich_winner_map_by_n_ir",
    "structure_g_curves_by_ir_at_temperature",
    "oxidation_vs_t_by_ir_at_pressure",
    "property_vs_t_by_ir_at_pressure",
    "ranked_structure_pops_by_ir_at_fixed_point",
]


def gas_formula_mask(formulas, gas_formulas) -> np.ndarray:
    """Boolean mask of the rows whose formula is one of ``gas_formulas``."""
    gas = {str(f) for f in gas_formulas}
    return np.array([str(f) in gas for f in np.asarray(formulas, dtype=object)], dtype=bool)


def ir_size_reservoir_mask(
    n_ir_per_struct, n_ir_target: int, formulas=None, *, include_gas_reservoir: bool = False, gas_formulas=(),
) -> np.ndarray:
    """Rows with ``n_Ir == target``, plus the gas reservoir rows when asked."""
    mask = np.asarray(n_ir_per_struct, dtype=int) == int(n_ir_target)
    if include_gas_reservoir and formulas is not None:
        mask = mask | gas_formula_mask(formulas, gas_formulas)
    return mask


def renorm_pgi_at_indices(Pgi, stoich, mask, *, p_idx: int | None = None):
    """Renormalise ``Pgi`` on ``mask``; return ``(stoich[mask], renormalised slab)``."""
    Pgi_m = renormalize_populations(Pgi, mask, p_idx=p_idx)
    stoich_m = np.asarray(stoich[mask], dtype=np.float64)
    return stoich_m, Pgi_m


def subset_joined_table_by_n_ir(
    jt: dict, n_ir_target: int, *, include_gas_reservoir: bool = False, gas_formulas=(),
) -> dict | None:
    """Restrict a joined GC–ACE table (``aist.ace.join.build_joined_table``) to one Ir bucket; renormalise ``Pgi``."""
    mask = ir_size_reservoir_mask(
        jt["n_Ir"], n_ir_target, jt.get("formulas_gc"),
        include_gas_reservoir=include_gas_reservoir, gas_formulas=gas_formulas,
    )
    if not np.any(mask):
        return None
    Pgi_m = renormalize_populations(jt["Pgi"], mask)
    n_O = np.asarray(jt["n_O"][mask], dtype=float)
    n_Ir = np.asarray(jt["n_Ir"][mask], dtype=float)
    stoich_m = n_O / n_Ir
    x_per_non_o = np.tensordot(stoich_m, Pgi_m, axes=1)
    return {
        "dataset": jt["dataset"],
        "n": int(mask.sum()),
        "frame_index": np.asarray(jt["frame_index"][mask], dtype=int),
        "formula": np.asarray(jt["formula"][mask], dtype=object),
        "n_Ir": np.asarray(jt["n_Ir"][mask], dtype=int),
        "n_O": np.asarray(jt["n_O"][mask], dtype=int),
        "cluster": np.asarray(jt["cluster"][mask], dtype=int),
        "ir_frac": np.asarray(jt["ir_frac"][mask], dtype=float),
        "o_frac": np.asarray(jt["o_frac"][mask], dtype=float),
        "formulas_gc": np.asarray(jt["formulas_gc"][mask], dtype=object),
        "Pgi": Pgi_m,
        "degeneracy": np.asarray(jt["degeneracy"])[mask],
        "Ts": jt["Ts"],
        "Ps": jt["Ps"],
        "phase_index": lowest_g_index(Pgi_m, np.asarray(jt["degeneracy"])[mask]),
        "x_per_non_o": x_per_non_o,
        "n_ace_clusters": jt.get("n_ace_clusters"),
        "gas_cluster_id": jt.get("gas_cluster_id"),
    }


def oxidation_map_by_n_ir(
    Ts, Ps, Pgi, stoich, n_ir_per_struct, n_ir_target: int, formulas=None, *, degeneracy,
    include_gas_reservoir: bool = False, gas_formulas=(),
) -> dict | None:
    """``⟨n_O/n_Ir⟩`` map and phase overlay for one Ir bucket (+ gas rows)."""
    mask = ir_size_reservoir_mask(
        n_ir_per_struct, n_ir_target, formulas, include_gas_reservoir=include_gas_reservoir, gas_formulas=gas_formulas,
    )
    if not np.any(mask):
        return None
    stoich_m, Pgi_n = renorm_pgi_at_indices(Pgi, stoich, mask)
    out: dict = {
        "Ts": Ts, "Ps": Ps,
        "x_per_non_o": np.tensordot(stoich_m, Pgi_n, axes=1),
        "phase_index": lowest_g_index(Pgi_n, np.asarray(degeneracy)[mask]),
        "n_struct": int(mask.sum()),
    }
    if formulas is not None:
        out["formulas"] = np.asarray(formulas[mask], dtype=object)
    return out


def stoich_winner_map_by_n_ir(
    Ts, Ps, Pgi, stoich, n_ir_per_struct, n_ir_target: int, formulas=None, *, degeneracy,
    include_gas_reservoir: bool = False, gas_formulas=(),
) -> dict | None:
    """Single-minimum ``n_O/n_Ir`` map for one Ir bucket: the lowest-G structure at each ``(p, T)``."""
    mask = ir_size_reservoir_mask(
        n_ir_per_struct, n_ir_target, formulas, include_gas_reservoir=include_gas_reservoir, gas_formulas=gas_formulas,
    )
    if not np.any(mask):
        return None
    stoich_m, Pgi_n = renorm_pgi_at_indices(Pgi, stoich, mask)
    winner = lowest_g_index(Pgi_n, np.asarray(degeneracy)[mask])
    out: dict = {"Ts": Ts, "Ps": Ps, "stoich_map": stoich_m[winner], "phase_index": winner, "n_struct": int(mask.sum())}
    if formulas is not None:
        out["formulas"] = np.asarray(formulas[mask], dtype=object)
    return out


def structure_g_curves_by_ir_at_temperature(
    G_tensor, stoich, n_ir_per_struct, t_idx: int, n_ir_target: int, formulas=None, *,
    include_gas_reservoir: bool = False, gas_formulas=(),
):
    """``G_i(p)`` at fixed ``T`` for one Ir bucket (+ gas rows): ``(G_sub, stoich_sub, formulas_sub)``."""
    mask = ir_size_reservoir_mask(
        n_ir_per_struct, n_ir_target, formulas, include_gas_reservoir=include_gas_reservoir, gas_formulas=gas_formulas,
    )
    if not np.any(mask):
        return None
    G_sub = np.asarray(G_tensor[mask, :, t_idx], dtype=np.float64)
    stoich_sub = np.asarray(stoich[mask], dtype=np.float64)
    formulas_sub = np.asarray(formulas[mask], dtype=object) if formulas is not None else np.array([], dtype=object)
    return G_sub, stoich_sub, formulas_sub


def _sizes(n_ir_per_struct, ir_counts):
    if ir_counts is None:
        return sorted(int(n) for n in np.unique(np.asarray(n_ir_per_struct, dtype=int)))
    return [int(n) for n in ir_counts]


def oxidation_vs_t_by_ir_at_pressure(
    gc: dict, stoich, n_ir_per_struct, pressure_pa: float, *, ir_counts=None, formulas=None,
    include_gas_reservoir: bool = False, gas_formulas=(),
):
    """``⟨n_O/n_Ir⟩`` vs ``T`` at fixed ``pO₂``, one curve per Ir bucket (default: every size present)."""
    p_idx, pressure_used = pressure_grid_index(gc["Ps"], pressure_pa)
    Ts = np.asarray(gc["Ts"], dtype=float)
    lines: dict[int, np.ndarray] = {}
    for n in _sizes(n_ir_per_struct, ir_counts):
        mask = ir_size_reservoir_mask(
            n_ir_per_struct, n, formulas, include_gas_reservoir=include_gas_reservoir, gas_formulas=gas_formulas,
        )
        if not np.any(mask):
            continue
        stoich_m, Pgi_n = renorm_pgi_at_indices(gc["Pgi"], stoich, mask, p_idx=p_idx)
        lines[n] = stoich_m @ Pgi_n
    return Ts, pressure_used, lines


def property_vs_t_by_ir_at_pressure(
    gc: dict, values, n_ir_per_struct, pressure_pa: float, *, ir_counts=None, formulas=None,
    include_gas_reservoir: bool = False, gas_formulas=(),
):
    """Ensemble average of a per-structure property vs ``T`` at fixed ``pO₂``, one curve per Ir bucket."""
    p_idx, pressure_used = pressure_grid_index(gc["Ps"], pressure_pa)
    Ts = np.asarray(gc["Ts"], dtype=float)
    values = np.asarray(values, dtype=float)
    lines: dict[int, np.ndarray] = {}
    for n in _sizes(n_ir_per_struct, ir_counts):
        mask = ir_size_reservoir_mask(
            n_ir_per_struct, n, formulas, include_gas_reservoir=include_gas_reservoir, gas_formulas=gas_formulas,
        )
        if not np.any(mask):
            continue
        Pgi_n = renormalize_populations(gc["Pgi"], mask, p_idx=p_idx)
        lines[n] = values[mask] @ Pgi_n
    return Ts, pressure_used, lines


def ranked_structure_pops_by_ir_at_fixed_point(
    novib_Pgi, full_Pgi, novib_G, n_ir_per_struct, Ts, Ps, T: float, P: float, *,
    ir_counts=None, formulas=None, cumulative: float = 0.97, max_structures: int | None = None,
    include_gas_reservoir: bool = False, gas_formulas=(),
):
    """Per Ir bucket: novib and full populations (%) in ascending novib-G order (s0 = the minimum).

    With ``max_structures`` the first that many are returned; otherwise the list stops once the running
    novib population reaches ``cumulative`` of the bucket total.
    """
    t_idx = int(np.argmin(np.abs(np.asarray(Ts, dtype=float) - float(T))))
    p_idx, p_used = pressure_grid_index(Ps, P)
    t_used = float(Ts[t_idx])
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for n in _sizes(n_ir_per_struct, ir_counts):
        mask = ir_size_reservoir_mask(
            n_ir_per_struct, n, formulas, include_gas_reservoir=include_gas_reservoir, gas_formulas=gas_formulas,
        )
        if not np.any(mask):
            continue
        G_sub = np.asarray(novib_G[mask, p_idx, t_idx], dtype=float)
        energy_order = np.argsort(G_sub)
        novib_sub = renormalize_populations(novib_Pgi, mask, p_idx=p_idx)[:, t_idx] * 100.0
        full_sub = renormalize_populations(full_Pgi, mask, p_idx=p_idx)[:, t_idx] * 100.0
        ordered_novib = novib_sub[energy_order]
        ordered_vib = full_sub[energy_order]
        if max_structures is not None:
            idx = np.arange(min(int(max_structures), ordered_novib.size), dtype=int)
        else:
            total = float(ordered_novib.sum())
            chosen: list[int] = []
            csum = 0.0
            for k, pop in enumerate(ordered_novib):
                chosen.append(k)
                csum += float(pop)
                if total > 0.0 and csum / total >= cumulative:
                    break
            if not chosen and ordered_novib.size:
                chosen = [0]
            idx = np.asarray(chosen, dtype=int)
        out[n] = (ordered_novib[idx], ordered_vib[idx])
    return t_used, p_used, out
