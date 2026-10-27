"""Re-weighting a grand-canonical ensemble from a (patched) free-energy tensor.

``compute_gc`` stores ``G_tensor`` (rows × P × T) and the chirality ``degeneracy`` of every row. Any
correction applied to ``G_tensor`` afterwards (a support term, a gas-reservoir patch) needs the populations,
the phase index and the ensemble averages recomputed with the same rule; these helpers do that.
"""
from __future__ import annotations

import numpy as np
from scipy.constants import Boltzmann, electron_volt

KB_EV = Boltzmann / electron_volt

# compute_gc ensemble averages that a patched G_tensor makes stale (their per-structure terms are not stored)
STALE_AFTER_PATCH = ("E_dft_avg", "E_elec_avg", "E_vib_avg", "E_rot_avg", "E_trans_avg")


def reensemble_from_g(g: np.ndarray, tg: np.ndarray, stoich: np.ndarray, degeneracy: np.ndarray) -> dict:
    """Populations from a free-energy tensor, with the same chirality degeneracies as ``compute_gc``.

    Returns ``Pgi`` (float32), ``phase_index`` (argmin G), ``x_per_non_o`` (ensemble ``n_O/n_M``), ``G_avg``,
    ``S_conf`` and ``pop_max``.
    """
    beta = 1.0 / (KB_EV * tg)
    weights = np.exp(-beta * (g - np.min(g, axis=0, keepdims=True)))
    weights *= np.asarray(degeneracy, dtype=float)[:, None, None]
    xi = np.sum(weights, axis=0)
    pgi = (weights / xi).astype(np.float32)
    phase_index = np.argmin(g, axis=0)
    return {
        "Pgi": pgi,
        "phase_index": phase_index,
        "x_per_non_o": np.sum(pgi * stoich[:, None, None], axis=0),
        "G_avg": np.sum(pgi * g, axis=0),
        "S_conf": -KB_EV * np.sum(pgi * np.log(pgi + 1e-20), axis=0),
        "pop_max": np.max(pgi, axis=(1, 2)),
    }


def refresh_ensemble(data: dict, g: np.ndarray, tg: np.ndarray, stoich: np.ndarray) -> None:
    """Re-ensemble ``data`` (a GC NPZ dict) in place from a patched ``g``: populations, phase, averages.

    ``q_ir_avg`` is recomputed from ``q_ir_mean``; the decomposition averages are dropped because they
    would keep the populations from before the patch.
    """
    data.update(reensemble_from_g(g, tg, stoich, data["degeneracy"]))
    if "q_ir_mean" in data:
        data["q_ir_avg"] = np.tensordot(np.asarray(data["q_ir_mean"], dtype=float), data["Pgi"], axes=1)
    for key in STALE_AFTER_PATCH:
        data.pop(key, None)


def renormalize_populations(Pgi: np.ndarray, mask: np.ndarray, *, p_idx: int | None = None) -> np.ndarray:
    """Renormalise ``Pgi`` on the rows selected by ``mask`` (per ``(p, T)`` column).

    If ``p_idx`` is set, only the ``(structure, T)`` slab at that pressure is kept (shape ``(n_sub, n_T)``);
    otherwise the full ``(n_sub, n_P, n_T)`` slab.
    """
    if p_idx is None:
        Pgi_m = np.asarray(Pgi[mask], dtype=np.float32)
    else:
        Pgi_m = np.asarray(Pgi[mask, p_idx, :], dtype=np.float32)
    wsum = Pgi_m.sum(axis=0, keepdims=True)
    np.divide(Pgi_m, wsum, out=Pgi_m, where=wsum > 0)
    return Pgi_m


def lowest_g_index(Pgi_m: np.ndarray, degeneracy_m: np.ndarray) -> np.ndarray:
    """Lowest-G structure of a subset at each grid point (rows of ``Pgi_m``).

    ``Pgi`` is proportional to ``g_i exp(-beta G_i)``, so ``argmax(Pgi / g_i)`` is ``argmin G``, the
    global-minimum definition of ``compute_gc``; ``argmax(Pgi)`` alone would pick a chiral (g = 2) isomer
    up to kT ln 2 above the minimum.
    """
    deg = np.asarray(degeneracy_m, dtype=float).reshape((-1,) + (1,) * (np.ndim(Pgi_m) - 1))
    return np.argmax(np.asarray(Pgi_m, dtype=float) / deg, axis=0)
