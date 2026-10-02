"""Grand-canonical views restricted to a fixed Ir-atom count."""
from __future__ import annotations

import numpy as np

from figures.config import THERMO_IR_COUNTS
from figures.config import GC_INCLUDE_GAS_RESERVOIR
from figures.utils.comparison import gas_structure_mask


def _mask_n_ir(n_ir_per_struct: np.ndarray, n_ir_target: int) -> np.ndarray:
    return np.asarray(n_ir_per_struct, dtype=int) == int(n_ir_target)


def ir_size_reservoir_mask(
    n_ir_per_struct: np.ndarray,
    n_ir_target: int,
    formulas,
    *,
    include_gas_reservoir: bool | None = None,
) -> np.ndarray:
    """Structures with ``n_Ir == target``, plus gas oxides (reservoir species, every row)."""
    if include_gas_reservoir is None:
        include_gas_reservoir = GC_INCLUDE_GAS_RESERVOIR
    mask = _mask_n_ir(n_ir_per_struct, n_ir_target)
    if include_gas_reservoir and formulas is not None:
        mask = mask | gas_structure_mask(formulas)
    return mask


def _renorm_pgi_slab(
    Pgi: np.ndarray,
    mask: np.ndarray,
    *,
    p_idx: int | None = None,
) -> np.ndarray:
    """Renormalise ``Pgi`` on ``mask`` (per ``(p, T)`` column).

    If ``p_idx`` is set, only the ``(structure, T)`` slab at that pressure is kept
    (shape ``(n_sub, n_T)``). Otherwise the full ``(n_sub, n_P, n_T)`` slab.
    """
    if p_idx is None:
        Pgi_m = np.asarray(Pgi[mask], dtype=np.float32)
    else:
        Pgi_m = np.asarray(Pgi[mask, p_idx, :], dtype=np.float32)
    wsum = Pgi_m.sum(axis=0, keepdims=True)
    np.divide(Pgi_m, wsum, out=Pgi_m, where=wsum > 0)
    return Pgi_m


def _renorm_pgi_at_indices(
    Pgi: np.ndarray,
    stoich: np.ndarray,
    mask: np.ndarray,
    *,
    p_idx: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Renormalise ``Pgi`` on ``mask``; return stoich weights and renormalised slab."""
    Pgi_m = _renorm_pgi_slab(Pgi, mask, p_idx=p_idx)
    stoich_m = np.asarray(stoich[mask], dtype=np.float64)
    return stoich_m, Pgi_m


def lowest_g_index(Pgi_m: np.ndarray, degeneracy_m: np.ndarray) -> np.ndarray:
    """Lowest-G structure of a bucket at each grid point (rows of ``Pgi_m``).

    ``Pgi`` is proportional to ``g_i exp(-beta G_i)``, so ``argmax(Pgi / g_i)`` is ``argmin G``, the
    global-minimum definition of the pooled maps (``compute_gc`` ``phase_index``); ``argmax(Pgi)``
    alone would pick a chiral (g = 2) isomer up to kT ln 2 above the minimum.
    """
    deg = np.asarray(degeneracy_m, dtype=float).reshape((-1,) + (1,) * (np.ndim(Pgi_m) - 1))
    return np.argmax(np.asarray(Pgi_m, dtype=float) / deg, axis=0)


def subset_joined_table_by_n_ir(
    jt: dict,
    n_ir_target: int,
    *,
    include_gas_reservoir: bool | None = None,
) -> dict | None:
    """Restrict a joined GC–ACE table to one Ir bucket + gas oxides; re-normalise ``Pgi``."""
    formulas = jt.get("formulas_gc")
    if include_gas_reservoir is None:
        include_gas_reservoir = GC_INCLUDE_GAS_RESERVOIR
    mask = ir_size_reservoir_mask(
        jt["n_Ir"], n_ir_target, formulas, include_gas_reservoir=include_gas_reservoir,
    )
    if not np.any(mask):
        return None

    Pgi_m = np.asarray(jt["Pgi"][mask], dtype=np.float32)
    wsum = Pgi_m.sum(axis=0, keepdims=True)
    np.divide(Pgi_m, wsum, out=Pgi_m, where=wsum > 0)

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
    Ts: np.ndarray,
    Ps: np.ndarray,
    Pgi: np.ndarray,
    stoich: np.ndarray,
    n_ir_per_struct: np.ndarray,
    n_ir_target: int,
    formulas: np.ndarray | None = None,
    *,
    degeneracy: np.ndarray,
    include_gas_reservoir: bool | None = None,
) -> dict | None:
    """``⟨n_O/n_Ir⟩`` map and phase overlay for one Ir bucket + gas oxides."""
    if include_gas_reservoir is None:
        include_gas_reservoir = GC_INCLUDE_GAS_RESERVOIR
    mask = ir_size_reservoir_mask(
        n_ir_per_struct, n_ir_target, formulas, include_gas_reservoir=include_gas_reservoir,
    )
    if not np.any(mask):
        return None

    stoich_m, Pgi_n = _renorm_pgi_at_indices(Pgi, stoich, mask)
    x_per_non_o = np.tensordot(stoich_m, Pgi_n, axes=1)
    out: dict = {
        "Ts": Ts,
        "Ps": Ps,
        "x_per_non_o": x_per_non_o,
        "phase_index": lowest_g_index(Pgi_n, np.asarray(degeneracy)[mask]),
        "n_struct": int(mask.sum()),
    }
    if formulas is not None:
        out["formulas"] = np.asarray(formulas[mask], dtype=object)
    return out


def stoich_winner_map_by_n_ir(
    Ts: np.ndarray,
    Ps: np.ndarray,
    Pgi: np.ndarray,
    stoich: np.ndarray,
    n_ir_per_struct: np.ndarray,
    n_ir_target: int,
    formulas: np.ndarray | None = None,
    *,
    degeneracy: np.ndarray,
    include_gas_reservoir: bool | None = None,
) -> dict | None:
    """Single-minimum ``n_O/n_Ir`` map for one Ir bucket + gas oxides.

    Winner at each ``(p, T)`` is the lowest-G structure of the bucket (``lowest_g_index``; same rule as
    the global ``*_stoich_map`` fields in ``gc_variants.npz``).
    """
    if include_gas_reservoir is None:
        include_gas_reservoir = GC_INCLUDE_GAS_RESERVOIR
    mask = ir_size_reservoir_mask(
        n_ir_per_struct, n_ir_target, formulas, include_gas_reservoir=include_gas_reservoir,
    )
    if not np.any(mask):
        return None

    stoich_m, Pgi_n = _renorm_pgi_at_indices(Pgi, stoich, mask)
    winner = lowest_g_index(Pgi_n, np.asarray(degeneracy)[mask])
    out: dict = {
        "Ts": Ts,
        "Ps": Ps,
        "stoich_map": stoich_m[winner],
        "phase_index": winner,
        "n_struct": int(mask.sum()),
    }
    if formulas is not None:
        out["formulas"] = np.asarray(formulas[mask], dtype=object)
    return out


def structure_g_curves_by_ir_at_temperature(
    G_tensor: np.ndarray,
    stoich: np.ndarray,
    n_ir_per_struct: np.ndarray,
    t_idx: int,
    n_ir_target: int,
    formulas: np.ndarray | None = None,
    *,
    include_gas_reservoir: bool | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """``G_i(p)`` at fixed ``T`` for Ir-size bucket + gas oxides.

    Returns ``(G_sub, stoich_sub, formulas_sub)``; ``formulas_sub`` is empty if
    ``formulas`` is not passed.
    """
    if include_gas_reservoir is None:
        include_gas_reservoir = GC_INCLUDE_GAS_RESERVOIR
    mask = ir_size_reservoir_mask(
        n_ir_per_struct, n_ir_target, formulas, include_gas_reservoir=include_gas_reservoir,
    )
    if not np.any(mask):
        return None
    G_sub = np.asarray(G_tensor[mask, :, t_idx], dtype=np.float64)
    stoich_sub = np.asarray(stoich[mask], dtype=np.float64)
    if formulas is not None:
        formulas_sub = np.asarray(formulas[mask], dtype=object)
    else:
        formulas_sub = np.array([], dtype=object)
    return G_sub, stoich_sub, formulas_sub


def oxidation_vs_t_by_ir_at_pressure(
    gc: dict,
    stoich: np.ndarray,
    n_ir_per_struct: np.ndarray,
    pressure_pa: float,
    *,
    ir_counts: range | list[int] | None = None,
    formulas: np.ndarray | None = None,
) -> tuple[np.ndarray, float, dict[int, np.ndarray]]:
    """``⟨n_O/n_Ir⟩`` vs ``T`` at fixed ``pO₂``, one curve per ``n_Ir`` bucket."""
    from aist.core.grandcanonical import pressure_grid_index

    if ir_counts is None:
        ir_counts = THERMO_IR_COUNTS
    p_idx, pressure_used = pressure_grid_index(gc["Ps"], pressure_pa)
    Ts = np.asarray(gc["Ts"], dtype=float)
    lines: dict[int, np.ndarray] = {}
    for n in ir_counts:
        mask = ir_size_reservoir_mask(n_ir_per_struct, int(n), formulas)
        if not np.any(mask):
            continue
        stoich_m, Pgi_n = _renorm_pgi_at_indices(
            gc["Pgi"], stoich, mask, p_idx=p_idx,
        )
        lines[int(n)] = stoich_m @ Pgi_n
    return Ts, pressure_used, lines


def homo_lumo_vs_t_by_ir_at_pressure(
    gc: dict,
    gaps: np.ndarray,
    n_ir_per_struct: np.ndarray,
    pressure_pa: float,
    *,
    ir_counts: range | list[int] | None = None,
    formulas: np.ndarray | None = None,
) -> tuple[np.ndarray, float, dict[int, np.ndarray]]:
    """Ensemble ``⟨HOMO–LUMO⟩`` vs ``T`` at fixed ``pO₂``, one curve per ``n_Ir`` bucket."""
    from aist.core.grandcanonical import pressure_grid_index

    if ir_counts is None:
        ir_counts = THERMO_IR_COUNTS
    p_idx, pressure_used = pressure_grid_index(gc["Ps"], pressure_pa)
    Ts = np.asarray(gc["Ts"], dtype=float)
    gaps = np.asarray(gaps, dtype=float)
    lines: dict[int, np.ndarray] = {}
    for n in ir_counts:
        mask = ir_size_reservoir_mask(n_ir_per_struct, int(n), formulas)
        if not np.any(mask):
            continue
        Pgi_n = _renorm_pgi_slab(gc["Pgi"], mask, p_idx=p_idx)
        lines[int(n)] = gaps[mask] @ Pgi_n
    return Ts, pressure_used, lines


def ranked_structure_pops_by_ir_at_fixed_point(
    novib_Pgi: np.ndarray,
    full_Pgi: np.ndarray,
    novib_G: np.ndarray,
    n_ir_per_struct: np.ndarray,
    Ts: np.ndarray,
    Ps: np.ndarray,
    T: float,
    P: float,
    *,
    ir_counts: range | list[int] | None = None,
    formulas: np.ndarray | None = None,
    cumulative: float = 0.97,
    max_structures: int | None = None,
    include_gas_reservoir: bool | None = None,
) -> tuple[float, float, dict[int, tuple[np.ndarray, np.ndarray]]]:
    """Per ``n_Ir``: novib and vib populations in novib energy order (s0, s1, …).

    Structures are ranked by ascending novib ``G`` within each Ir bucket (s0 is the
    global minimum). When ``max_structures`` is set, only the first ``max_structures``
    in that order are returned; otherwise the list is truncated once the running novib
    population reaches ``cumulative`` of the bucket total.
    """
    from aist.core.grandcanonical import pressure_grid_index

    if ir_counts is None:
        ir_counts = THERMO_IR_COUNTS
    t_idx = int(np.argmin(np.abs(np.asarray(Ts, dtype=float) - float(T))))
    p_idx, p_used = pressure_grid_index(Ps, P)
    t_used = float(Ts[t_idx])
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for n in ir_counts:
        mask = ir_size_reservoir_mask(
            n_ir_per_struct, int(n), formulas,
            include_gas_reservoir=include_gas_reservoir,
        )
        if not np.any(mask):
            continue
        G_sub = np.asarray(novib_G[mask, p_idx, t_idx], dtype=float)
        energy_order = np.argsort(G_sub)
        novib_sub = _renorm_pgi_slab(novib_Pgi, mask, p_idx=p_idx)[:, t_idx] * 100.0
        full_sub = _renorm_pgi_slab(full_Pgi, mask, p_idx=p_idx)[:, t_idx] * 100.0
        ordered_novib = novib_sub[energy_order]
        ordered_vib = full_sub[energy_order]

        if max_structures is not None:
            n_keep = min(int(max_structures), ordered_novib.size)
            idx = np.arange(n_keep, dtype=int)
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
        out[int(n)] = (ordered_novib[idx], ordered_vib[idx])
    return t_used, p_used, out
