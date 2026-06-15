"""Join ACE clustering features with grand-canonical NPZ outputs."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d

from aist.core.grandcanonical import pressure_grid_index


def build_joined_table(
    gc_data: dict,
    ace_rows: list[dict],
    dataset: str = "PBE",
    *,
    strict: bool = True,
) -> dict:
    """Align GC ensemble weights with per-structure ACE labels."""
    n_gc = gc_data["Pgi"].shape[0]
    n_ace = len(ace_rows)
    if n_gc != n_ace:
        if strict:
            raise ValueError(
                f"[{dataset}] GC has {n_gc} structures, ACE features has {n_ace} rows. "
                "Re-run cache with --force after updating data/ trajectories."
            )
        n = min(n_gc, n_ace)
        print(
            f"[warn][join] truncating to {n} structures "
            f"(GC={n_gc}, ACE={n_ace}, dataset={dataset})"
        )
        gc_pgi = gc_data["Pgi"][:n]
        gc_phase = gc_data["phase_index"]
        ace_rows = ace_rows[:n]
    else:
        n = n_gc
        gc_pgi = gc_data["Pgi"]
        gc_phase = gc_data["phase_index"]

    ir_cols = sorted(
        [k for k in ace_rows[0] if k.startswith("Ir-") and k.endswith("_frac")]
    )
    o_cols = sorted(
        [k for k in ace_rows[0] if k.startswith("O-") and k.endswith("_frac")]
    )

    return {
        "dataset": dataset,
        "n": n,
        "frame_index": np.array([r["frame_index"] for r in ace_rows], dtype=int),
        "formula": np.array([r["formula"] for r in ace_rows], dtype=object),
        "n_Ir": np.array([r["n_Ir"] for r in ace_rows], dtype=int),
        "n_O": np.array([r["n_O"] for r in ace_rows], dtype=int),
        "cluster": np.array([r["cluster"] for r in ace_rows], dtype=int),
        "ir_frac": np.array([[r[c] for c in ir_cols] for r in ace_rows], dtype=float),
        "o_frac": np.array([[r[c] for c in o_cols] for r in ace_rows], dtype=float),
        "formulas_gc": gc_data["formulas"][:n],
        "Pgi": gc_pgi,
        "Ts": gc_data["Ts"],
        "Ps": gc_data["Ps"],
        "phase_index": gc_phase,
        "x_per_non_o": gc_data["x_per_non_o"],
    }


def ensemble_weighted_mean_map(
    Pgi: np.ndarray,
    per_structure: np.ndarray,
) -> np.ndarray:
    """Ensemble average of a per-structure scalar on the (P, T) grid.

    Same construction as ``x_per_non_o`` and ``tchar_ensemble_map``:
    ``⟨v⟩(T,P) = Σ_i Pgi[i] · v_i / Σ_i Pgi[i]``.
    """
    per_structure = np.asarray(per_structure, dtype=float)
    numerator = np.einsum("i,ijk->jk", per_structure, Pgi)
    weight_sum = Pgi.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(weight_sum > 0, numerator / weight_sum, np.nan)


def ensemble_mean_cluster_map(joined: dict) -> np.ndarray:
    """Ensemble-averaged structure-cluster index ⟨C⟩ on (T, P)."""
    return ensemble_weighted_mean_map(joined["Pgi"], joined["cluster"].astype(float))


def ensemble_mean_motif_index_map(joined: dict, motif_key: str = "ir_frac") -> np.ndarray:
    """Ensemble-averaged mean atom-motif index ⟨m⟩ from per-structure motif fractions."""
    frac = np.asarray(joined[motif_key], dtype=float)
    k = frac.shape[1]
    motif_index = np.arange(k, dtype=float)
    per_struct = frac @ motif_index
    return ensemble_weighted_mean_map(joined["Pgi"], per_struct)


def cluster_population_fractions(joined: dict) -> np.ndarray:
    """Population fraction per cluster c at each (T, P); shape (k, nP, nT)."""
    Pgi = joined["Pgi"]
    cluster = joined["cluster"]
    k = int(cluster.max()) + 1
    nP, nT = Pgi.shape[1], Pgi.shape[2]
    f_c = np.zeros((k, nP, nT), dtype=float)
    for c in range(k):
        mask = cluster == c
        if mask.any():
            f_c[c] = Pgi[mask].sum(axis=0)
    den = f_c.sum(axis=0, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        f_c = np.where(den > 0, f_c / den, 0.0)
    return f_c


def ir_motif_population_fractions(joined: dict) -> np.ndarray:
    """Ir atom-motif population fractions at each (T, P); shape ``(k_Ir, nP, nT)``."""
    Pgi = joined["Pgi"]
    n_Ir = np.asarray(joined["n_Ir"], dtype=float)
    ir_frac = np.asarray(joined["ir_frac"], dtype=float)
    k = ir_frac.shape[1]
    nP, nT = Pgi.shape[1], Pgi.shape[2]
    f_m = np.zeros((k, nP, nT), dtype=float)
    for m in range(k):
        f_m[m] = np.einsum("i,ijk->jk", n_Ir * ir_frac[:, m], Pgi)
    den = f_m.sum(axis=0, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        f_m = np.where(den > 0, f_m / den, 0.0)
    return f_m


def o_motif_population_fractions(joined: dict) -> np.ndarray:
    """O atom-motif population fractions at each (T, P); shape ``(k_O, nP, nT)``."""
    Pgi = joined["Pgi"]
    n_O = np.asarray(joined["n_O"], dtype=float)
    o_frac = np.asarray(joined["o_frac"], dtype=float)
    k = o_frac.shape[1]
    nP, nT = Pgi.shape[1], Pgi.shape[2]
    f_m = np.zeros((k, nP, nT), dtype=float)
    for m in range(k):
        f_m[m] = np.einsum("i,ijk->jk", n_O * o_frac[:, m], Pgi)
    den = f_m.sum(axis=0, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        f_m = np.where(den > 0, f_m / den, 0.0)
    return f_m


def _smooth_population_stack_along_T(
    Y: np.ndarray,
    smooth_sigma: float,
) -> np.ndarray:
    if smooth_sigma > 0 and Y.shape[1] > 3:
        Y = gaussian_filter1d(Y, sigma=float(smooth_sigma), axis=1, mode="nearest")
        col_sum = Y.sum(axis=0, keepdims=True)
        np.divide(Y, col_sum, out=Y, where=col_sum > 0)
        Y *= 100.0
    return Y


def ir_motif_population_vs_T_at_pressure(
    joined: dict,
    pressure_pa: float,
    *,
    smooth_sigma: float = 0.0,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Ir atom-motif population (%) vs ``T`` at fixed ``pO₂``; shape ``(k_Ir, n_T)``."""
    p_idx, pressure_used = pressure_grid_index(joined["Ps"], pressure_pa)
    Y = 100.0 * ir_motif_population_fractions(joined)[:, p_idx, :]
    Y = _smooth_population_stack_along_T(Y, smooth_sigma)
    return np.asarray(joined["Ts"], dtype=float), pressure_used, Y


def o_motif_population_vs_T_at_pressure(
    joined: dict,
    pressure_pa: float,
    *,
    smooth_sigma: float = 0.0,
) -> tuple[np.ndarray, float, np.ndarray]:
    """O atom-motif population (%) vs ``T`` at fixed ``pO₂``; shape ``(k_O, n_T)``."""
    p_idx, pressure_used = pressure_grid_index(joined["Ps"], pressure_pa)
    Y = 100.0 * o_motif_population_fractions(joined)[:, p_idx, :]
    Y = _smooth_population_stack_along_T(Y, smooth_sigma)
    return np.asarray(joined["Ts"], dtype=float), pressure_used, Y


def cluster_population_vs_T_at_pressure(
    joined: dict,
    pressure_pa: float,
    *,
    smooth_sigma: float = 0.0,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Structure-cluster population (%) vs ``T`` at fixed ``pO₂``.

    Returns ``(Ts, pressure_used, Y)`` with ``Y`` shape ``(k, n_T)``.
    """
    p_idx, pressure_used = pressure_grid_index(joined["Ps"], pressure_pa)
    Y = 100.0 * cluster_population_fractions(joined)[:, p_idx, :]
    Y = _smooth_population_stack_along_T(Y, smooth_sigma)
    return np.asarray(joined["Ts"], dtype=float), pressure_used, Y


def atom_motif_population_vs_T_at_pressure(
    joined: dict,
    pressure_pa: float,
    *,
    smooth_sigma: float = 0.0,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Atom-motif population (%) vs ``T`` at fixed ``pO₂``, Ir motifs then O motifs.

    Each slice is the ensemble fraction of **all atoms** in motif ``Ir-A*`` or ``O-A*``.
    Stacking Ir then O motifs makes the boundary between blocks track oxidation.
    Returns ``(Ts, pressure_used, Y)`` with ``Y`` shape ``(k_Ir + k_O, n_T)``.
    """
    p_idx, pressure_used = pressure_grid_index(joined["Ps"], pressure_pa)
    Pgi = joined["Pgi"]
    n_Ir = np.asarray(joined["n_Ir"], dtype=float)
    n_O = np.asarray(joined["n_O"], dtype=float)
    ir_frac = np.asarray(joined["ir_frac"], dtype=float)
    o_frac = np.asarray(joined["o_frac"], dtype=float)
    k_ir, k_o = ir_frac.shape[1], o_frac.shape[1]
    n_T = int(Pgi.shape[2])

    Y = np.zeros((k_ir + k_o, n_T), dtype=float)
    for j in range(n_T):
        w = Pgi[:, p_idx, j]
        total_atoms = float(np.sum(w * (n_Ir + n_O)))
        if total_atoms <= 0:
            continue
        for m in range(k_ir):
            Y[m, j] = 100.0 * float(np.sum(w * n_Ir * ir_frac[:, m])) / total_atoms
        for m in range(k_o):
            Y[k_ir + m, j] = 100.0 * float(np.sum(w * n_O * o_frac[:, m])) / total_atoms

    if smooth_sigma > 0 and Y.shape[1] > 3:
        Y = gaussian_filter1d(Y, sigma=float(smooth_sigma), axis=1, mode="nearest")
        col_sum = Y.sum(axis=0, keepdims=True)
        np.divide(Y, col_sum, out=Y, where=col_sum > 0)
        Y *= 100.0

    return np.asarray(joined["Ts"], dtype=float), pressure_used, Y


def o_bridging_terminal_interface_vs_T_at_pressure(
    joined: dict,
    pressure_pa: float,
    bridging_o_index: int,
    *,
    smooth_sigma: float = 0.0,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Cumulative stack % where bridging O ends and terminal O begins.

    Matches the black Ir|O boundary plus the bridging O motif band in the
  atom-motif stack (Ir-A* below, O-A* above; interface after ``O-A{bridging}``).
    """
    Ts, pressure_used, Y = atom_motif_population_vs_T_at_pressure(
        joined, pressure_pa, smooth_sigma=smooth_sigma,
    )
    k_ir = int(joined["ir_frac"].shape[1])
    bridge_row = k_ir + int(bridging_o_index)
    interface = np.sum(Y[:k_ir, :], axis=0) + Y[bridge_row, :]
    return Ts, pressure_used, interface


def population_weighted_o_spins_by_motif_at_pressure(
    joined: dict,
    structures: list,
    motif_json_path: str,
    pressure_pa: float,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict]:
    """O-atom |Hirshfeld spin| samples weighted by GC population at fixed ``pO₂``.

    At each ``T`` on the joined grid, structure ``i`` contributes weight
    ``Pgi[i, p, T]`` for every O atom in that structure. Returns
    ``{motif_tag: (spins, weights)}`` and the motif JSON dict (colours).
    """
    from aist.ace.io import load_motif_atoms_json

    motif = load_motif_atoms_json(motif_json_path)
    order = list(motif.get("ir_motifs", [])) + list(motif.get("o_motifs", []))
    buckets: dict[str, list[tuple[float, float]]] = {tag: [] for tag in order}

    p_idx, _ = pressure_grid_index(joined["Ps"], pressure_pa)
    Pgi = joined["Pgi"]
    n_T = int(Pgi.shape[2])
    n = int(joined["n"])

    for i in range(n):
        fi = int(joined["frame_index"][i])
        if fi < 0 or fi >= len(structures):
            continue
        atoms = structures[fi]
        labels = motif.get("frames", {}).get(str(fi))
        if labels is None:
            continue
        h = atoms.info.get("hirshfeld")
        if not isinstance(h, dict) or "spins" not in h:
            continue
        spins = np.abs(np.asarray(h["spins"], dtype=float))
        syms = atoms.get_chemical_symbols()
        if len(labels) != len(syms) or len(spins) != len(syms):
            continue

        for ai, (sym, mot) in enumerate(zip(syms, labels)):
            if sym != "O" or mot not in buckets:
                continue
            spin = float(spins[ai])
            for j in range(n_T):
                w = float(Pgi[i, p_idx, j])
                if w > 0:
                    buckets[mot].append((spin, w))

    points = {
        tag: (
            np.array([p[0] for p in pts], dtype=float),
            np.array([p[1] for p in pts], dtype=float),
        )
        for tag, pts in buckets.items()
    }
    return points, motif


def cluster_population_map(joined: dict) -> np.ndarray:
    """Dominant cluster id at each (T, P) — legacy; prefer ``ensemble_mean_cluster_map``."""
    return np.argmax(cluster_population_fractions(joined), axis=0)


def motif_population_map(joined: dict, motif_key: str = "ir_frac") -> np.ndarray:
    """Dominant atom-motif id at each (T, P) — legacy; prefer ``ensemble_mean_motif_index_map``."""
    Pgi = joined["Pgi"]
    frac = joined[motif_key]
    k = frac.shape[1]
    nP, nT = Pgi.shape[1], Pgi.shape[2]
    f_m = np.zeros((k, nP, nT), dtype=float)
    for m in range(k):
        f_m[m] = np.einsum("i,ijk->jk", frac[:, m], Pgi)
    den = f_m.sum(axis=0, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        f_m = np.where(den > 0, f_m / den, 0.0)
    return np.argmax(f_m, axis=0)


def structure_populations_at_fixed_point(
    Pgi: np.ndarray,
    Ts: np.ndarray,
    Ps: np.ndarray,
    T: float,
    P: float,
) -> np.ndarray:
    """Per-structure population (%) at ``(T, P)``; shape ``(n_struct,)``."""
    t_idx = int(np.argmin(np.abs(Ts - T)))
    p_idx = int(np.argmin(np.abs(Ps - P)))
    return np.asarray(Pgi[:, p_idx, t_idx], dtype=float) * 100.0


def ranked_cumulative_population_lines(
    pops_order: np.ndarray,
    pops_overlay: np.ndarray,
    *,
    cumulative: float = 0.95,
) -> dict:
    """Structures covering ``cumulative`` of total pop., x ordered by ``pops_order`` (high→low).

    Returns dict with ``indices`` (structure ids in plot order), ``x`` (0…n−1),
    ``y_order`` (novib / reference populations), ``y_overlay`` (vib at same order).
    """
    pops_order = np.asarray(pops_order, dtype=float)
    pops_overlay = np.asarray(pops_overlay, dtype=float)
    total = float(pops_order.sum())
    if total <= 0:
        return {
            "indices": np.array([], dtype=int),
            "x": np.array([], dtype=float),
            "y_order": np.array([], dtype=float),
            "y_overlay": np.array([], dtype=float),
        }

    order = np.argsort(-pops_order)
    target = cumulative * total
    chosen: list[int] = []
    csum = 0.0
    for idx in order:
        chosen.append(int(idx))
        csum += float(pops_order[idx])
        if csum >= target:
            break

    indices = np.asarray(chosen, dtype=int)
    n = indices.size
    return {
        "indices": indices,
        "x": np.arange(n, dtype=float),
        "y_order": pops_order[indices],
        "y_overlay": pops_overlay[indices],
        "cumulative_fraction": csum / total,
        "n_total": pops_order.size,
    }


def fixed_point_cluster_populations(joined: dict, T: float, P: float) -> np.ndarray:
    Ts, Ps = joined["Ts"], joined["Ps"]
    t_idx = int(np.argmin(np.abs(Ts - T)))
    p_idx = int(np.argmin(np.abs(Ps - P)))
    Pgi = joined["Pgi"]
    cluster = joined["cluster"]
    k = int(cluster.max()) + 1
    pops = np.zeros(k, dtype=float)
    for c in range(k):
        mask = cluster == c
        pops[c] = float(Pgi[mask, p_idx, t_idx].sum()) * 100.0
    return pops


def fixed_point_cluster_populations_from_pgi(
    Pgi: np.ndarray,
    cluster: np.ndarray,
    Ts: np.ndarray,
    Ps: np.ndarray,
    T: float,
    P: float,
) -> np.ndarray:
    """Cluster populations at (T,P) from a standalone Pgi tensor."""
    t_idx = int(np.argmin(np.abs(Ts - T)))
    p_idx = int(np.argmin(np.abs(Ps - P)))
    k = int(cluster.max()) + 1
    pops = np.zeros(k, dtype=float)
    for c in range(k):
        mask = cluster == c
        pops[c] = float(Pgi[mask, p_idx, t_idx].sum()) * 100.0
    return pops
