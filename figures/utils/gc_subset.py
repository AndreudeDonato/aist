"""Grand-canonical views restricted to a fixed Ir-atom count."""
from __future__ import annotations

import numpy as np


def _mask_n_ir(n_ir_per_struct: np.ndarray, n_ir_target: int) -> np.ndarray:
    return np.asarray(n_ir_per_struct, dtype=int) == int(n_ir_target)


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


def subset_joined_table_by_n_ir(jt: dict, n_ir_target: int) -> dict | None:
    """Restrict a joined GC–ACE table to one Ir-count bucket with re-normalised ``Pgi``."""
    mask = np.asarray(jt["n_Ir"], dtype=int) == int(n_ir_target)
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
        "Ts": jt["Ts"],
        "Ps": jt["Ps"],
        "phase_index": np.argmax(Pgi_m, axis=0),
        "x_per_non_o": x_per_non_o,
    }


def oxidation_map_by_n_ir(
    Ts: np.ndarray,
    Ps: np.ndarray,
    Pgi: np.ndarray,
    stoich: np.ndarray,
    n_ir_per_struct: np.ndarray,
    n_ir_target: int,
    formulas: np.ndarray | None = None,
) -> dict | None:
    """``⟨n_O/n_Ir⟩`` map and phase overlay for one Ir-count bucket."""
    mask = _mask_n_ir(n_ir_per_struct, n_ir_target)
    if not np.any(mask):
        return None

    stoich_m, Pgi_n = _renorm_pgi_at_indices(Pgi, stoich, mask)
    x_per_non_o = np.tensordot(stoich_m, Pgi_n, axes=1)
    out: dict = {
        "Ts": Ts,
        "Ps": Ps,
        "x_per_non_o": x_per_non_o,
        "phase_index": np.argmax(Pgi_n, axis=0),
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """``G_i(p)`` at fixed ``T`` for every structure with ``n_Ir == n_ir_target``.

    Returns ``(G_sub, stoich_sub, formulas_sub)``; ``formulas_sub`` is empty if
    ``formulas`` is not passed.
    """
    mask = _mask_n_ir(n_ir_per_struct, n_ir_target)
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
) -> tuple[np.ndarray, float, dict[int, np.ndarray]]:
    """``⟨n_O/n_Ir⟩`` vs ``T`` at fixed ``pO₂``, one curve per ``n_Ir`` bucket."""
    from aist.core.grandcanonical import pressure_grid_index

    if ir_counts is None:
        ir_counts = range(3, 10)
    p_idx, pressure_used = pressure_grid_index(gc["Ps"], pressure_pa)
    Ts = np.asarray(gc["Ts"], dtype=float)
    lines: dict[int, np.ndarray] = {}
    for n in ir_counts:
        mask = _mask_n_ir(n_ir_per_struct, int(n))
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
) -> tuple[np.ndarray, float, dict[int, np.ndarray]]:
    """Ensemble ``⟨HOMO–LUMO⟩`` vs ``T`` at fixed ``pO₂``, one curve per ``n_Ir`` bucket."""
    from aist.core.grandcanonical import pressure_grid_index

    if ir_counts is None:
        ir_counts = range(3, 10)
    p_idx, pressure_used = pressure_grid_index(gc["Ps"], pressure_pa)
    Ts = np.asarray(gc["Ts"], dtype=float)
    gaps = np.asarray(gaps, dtype=float)
    lines: dict[int, np.ndarray] = {}
    for n in ir_counts:
        mask = _mask_n_ir(n_ir_per_struct, int(n))
        if not np.any(mask):
            continue
        Pgi_n = _renorm_pgi_slab(gc["Pgi"], mask, p_idx=p_idx)
        lines[int(n)] = gaps[mask] @ Pgi_n
    return Ts, pressure_used, lines
