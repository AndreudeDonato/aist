"""Join ACE clustering features with grand-canonical NPZ outputs."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d

from aist_v01.core.grandcanonical import pressure_grid_index


def align_ace_rows_to_gc(gc_data: dict, ace_rows: list[dict]) -> list[dict]:
    """ACE rows in the order of the GC structure rows, checked by formula.

    ``gc_data["source_frames"]`` (the ``<npz>_frames.npy`` sidecar) gives each GC row's trajectory frame
    (-1 = gas oxide). ACE rows are picked by their ``frame_index``; rows with ``frame_index < 0`` (synthetic
    gas rows) are kept at the end. Raises when the sidecar is missing, a frame has no ACE row, or the
    formulas of a GC row and its ACE row differ (e.g. ACE rows of the other functional).
    """
    src = gc_data.get("source_frames")
    if src is None:
        raise ValueError("GC data has no source_frames (the <npz>_frames.npy sidecar); rebuild the cache")
    frames = [int(s) for s in np.asarray(src) if int(s) >= 0]
    struct_rows = [r for r in ace_rows if int(r.get("frame_index", -1)) >= 0]
    gas_rows = [r for r in ace_rows if int(r.get("frame_index", -1)) < 0]
    by_frame = {int(r["frame_index"]): r for r in struct_rows}
    missing = [f for f in frames if f not in by_frame]
    if missing:
        raise ValueError(f"GC structures without an ACE row (frames {missing[:5]}…); rebuild the cache")
    aligned = [by_frame[f] for f in frames]
    gc_formulas = [str(f) for f in np.asarray(gc_data["formulas"])[: len(aligned)]]
    bad = [(f, r["formula"], g) for f, r, g in zip(frames, aligned, gc_formulas) if str(r["formula"]) != g]
    if bad:
        raise ValueError(f"ACE rows do not match the GC rows (frame, ACE, GC): {bad[:3]}… ({len(bad)} rows)")
    return aligned + gas_rows


def build_joined_table(
    gc_data: dict,
    ace_rows: list[dict],
    dataset: str = "PBE",
) -> dict:
    """Align GC ensemble weights with per-structure ACE labels."""
    ace_rows = align_ace_rows_to_gc(gc_data, ace_rows)
    n = gc_data["Pgi"].shape[0]
    if n != len(ace_rows):
        raise ValueError(
            f"[{dataset}] GC has {n} structures, ACE features has {len(ace_rows)} rows. "
            "Re-run cache with --force after updating data/ trajectories."
        )
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
        "degeneracy": np.asarray(gc_data["degeneracy"], dtype=float),
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


def ensemble_weighted_rgb_map(
    Pgi: np.ndarray,
    per_structure_rgb: np.ndarray,
) -> np.ndarray:
    """Population-weighted RGB field on (P, T); shape ``(nP, nT, 3)``."""
    per_structure_rgb = np.asarray(per_structure_rgb, dtype=float)
    numerator = np.einsum("il,ijk->jkl", per_structure_rgb, Pgi)
    weight_sum = Pgi.sum(axis=0)
    denom = weight_sum[..., np.newaxis]
    with np.errstate(invalid="ignore", divide="ignore"):
        rgb = np.divide(numerator, denom, out=np.zeros_like(numerator), where=denom > 0)
    return np.clip(rgb, 0.0, 1.0)


def cluster_population_fractions(joined: dict) -> np.ndarray:
    """Population fraction per cluster c at each (T, P); shape (k, nP, nT).

    k is the clustering's ``n_ace_clusters`` (+1 for a synthetic gas cluster) when the joined table carries
    it, so a subset without the top cluster still gives one row per cluster colour.
    """
    Pgi = joined["Pgi"]
    cluster = joined["cluster"]
    k = int(cluster.max()) + 1
    if joined.get("n_ace_clusters") is not None:
        k = max(k, int(joined["n_ace_clusters"]) + (joined.get("gas_cluster_id") is not None))
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


def ensemble_o_motif_per_ir_map(joined: dict, o_motif_index: int) -> np.ndarray:
    """Ensemble motif O per Ir on the ``(P, T)`` grid: ``Σ Pgi·n_O·f_m / Σ Pgi·n_Ir``."""
    Pgi = joined["Pgi"]
    n_Ir = np.asarray(joined["n_Ir"], dtype=float)
    n_O = np.asarray(joined["n_O"], dtype=float)
    o_frac = np.asarray(joined["o_frac"], dtype=float)
    m = int(o_motif_index)
    n_o_motif = n_O * o_frac[:, m]
    numerator = np.einsum("i,ijk->jk", n_o_motif, Pgi)
    denominator = np.einsum("i,ijk->jk", n_Ir, Pgi)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denominator > 0, numerator / denominator, np.nan)


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
