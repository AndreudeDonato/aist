"""I/O helpers for the paper figure pipeline.

All figure scripts should load data through these helpers to keep path logic
in one place (config.py) and validate inputs early.
"""
from __future__ import annotations

import csv
import os
import sys

import numpy as np
from ase.io import read

# Ensure root and analysis/ are importable
_HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in [_HERE, os.path.join(_HERE, "analysis")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


def load_npz(path: str, *, keys: tuple[str, ...] | None = None) -> dict:
    """Load a cache NPZ as a plain dict.

    Pass ``keys`` to load only those arrays (much lower RAM for large GC files).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"NPZ not found: {path}\nRun: python3 figures/utils/cache.py")
    data = np.load(path, allow_pickle=True)
    file_keys = keys if keys is not None else data.files
    out = {k: data[k] for k in file_keys if k in data.files}
    if "dataset_names" in out:
        out["dataset_names"] = [str(x) for x in np.asarray(out["dataset_names"]).tolist()]
    if "plot_axes" in out:
        out["plot_axes"] = tuple(np.asarray(out["plot_axes"]).tolist())
    return out


def _attach_source_frames(data: dict, npz_path: str) -> dict:
    """Attach the ``<npz>_frames.npy`` sidecar (trajectory frame of each GC row, -1 = gas oxide)."""
    side = npz_path[:-4] + "_frames.npy"
    if not os.path.exists(side):
        raise FileNotFoundError(f"GC frames sidecar not found: {side}\nRun: python3 figures/build_all.py --rebuild-cache")
    data["source_frames"] = np.load(side)
    return data


def load_variants_like(variants_npz: str, gc_ref: dict, keys: tuple[str, ...]) -> dict:
    """Fields of a GC variants NPZ, checked to hold the same rows (trajectory frames) as ``gc_ref``."""
    vd = _attach_source_frames(load_npz(variants_npz, keys=keys), variants_npz)
    ref = gc_ref.get("source_frames")
    if ref is None or not np.array_equal(np.asarray(ref), vd["source_frames"]):
        raise ValueError(f"{variants_npz} rows do not match the reference GC rows; rebuild the cache")
    for k in keys:
        if k.endswith("Pgi") and np.shape(vd[k])[0] != len(ref):
            raise ValueError(f"{variants_npz}: {k} has {np.shape(vd[k])[0]} rows, frames sidecar {len(ref)}")
    return vd


def load_gc_npz(path: str, *, light: bool = False, extra_keys: tuple[str, ...] = (), frames: bool = True) -> dict:
    """Load grandcanonical_results.npz and return as plain dict.

    With ``light=True``, skip ``G_tensor`` and other fields not needed for
    subset oxidation maps (saves ~100 MB per file at 150×150 grids); ``extra_keys`` adds fields
    to that light set. ``frames=False`` for ensembles that are not trajectory frames (rutile GC);
    otherwise the frames sidecar is attached as ``source_frames`` and must exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"GC results not found: {path}\n"
            "Run: python3 figures/utils/cache.py"
        )
    if not light:
        data = load_npz(path)
    else:
        data = load_npz(
            path,
            keys=("Ts", "Ps", "Pgi", "stoich", "formulas", "phase_index", "x_per_non_o", "degeneracy")
            + tuple(extra_keys),
        )
    return _attach_source_frames(data, path) if frames else data


def load_gc_pbe_novib_view(variants_npz: str, formulas_npz: str) -> dict:
    """PBE grand-canonical without vib/ZPE, keyed like ``load_gc_npz``.

    Built from ``gc_variants.npz`` (novib tensors) plus per-structure ``formulas``
    from the full PBE GC NPZ (same structure order; electronic G only changes weights).
    """
    vd = load_npz(
        variants_npz,
        keys=("Ts", "Ps", "stoich", "degeneracy", "novib_Pgi", "novib_phase_index", "novib_x_per_non_o"),
    )
    formulas = load_npz(formulas_npz, keys=("formulas",))["formulas"]
    return _attach_source_frames({
        "Ts": vd["Ts"],
        "Ps": vd["Ps"],
        "stoich": vd["stoich"],
        "degeneracy": vd["degeneracy"],
        "Pgi": vd["novib_Pgi"],
        "phase_index": vd["novib_phase_index"],
        "x_per_non_o": vd["novib_x_per_non_o"],
        "formulas": formulas,
    }, variants_npz)


def load_gc_pbe_novib_g_tensor(variants_npz: str) -> tuple[np.ndarray, dict]:
    """Novib ``G_tensor`` plus ``Ts``, ``Ps``, ``novib_Pgi``, ``stoich`` for G–μ slices."""
    vd = load_npz(
        variants_npz,
        keys=("Ts", "Ps", "novib_Pgi", "novib_G_tensor", "stoich"),
    )
    gc = {
        "Ts": vd["Ts"],
        "Ps": vd["Ps"],
        "Pgi": vd["novib_Pgi"],
    }
    return vd["novib_G_tensor"], gc


def load_ace_features(path: str) -> list[dict]:
    """Load ace_features_<label>.csv as a list of row dicts."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"ACE features CSV not found: {path}\n"
            "Run: python3 figures/utils/cache.py --only ace"
        )
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    # cast numeric columns
    for row in rows:
        row["frame_index"] = int(row["frame_index"])
        row["n_Ir"] = int(row["n_Ir"])
        row["n_O"] = int(row["n_O"])
        row["cluster"] = int(row["cluster"])
        for k in list(row.keys()):
            if k.endswith("_frac"):
                row[k] = float(row[k])
    return rows


def load_traj(path: str):
    """Load an ASE trajectory file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Trajectory not found: {path}")
    return read(path, ":")


def ensure_dirs(*dirs: str) -> None:
    """Create output directories if they do not exist."""
    for d in dirs:
        os.makedirs(d, exist_ok=True)
