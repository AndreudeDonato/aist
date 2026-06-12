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


def load_npz(path: str) -> dict:
    """Load any cache NPZ as a plain dict."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"NPZ not found: {path}\nRun: python3 figures/utils/cache.py")
    data = np.load(path, allow_pickle=True)
    out = {k: data[k] for k in data.files}
    if "dataset_names" in out:
        out["dataset_names"] = [str(x) for x in np.asarray(out["dataset_names"]).tolist()]
    if "plot_axes" in out:
        out["plot_axes"] = tuple(np.asarray(out["plot_axes"]).tolist())
    return out


def load_gc_npz(path: str) -> dict:
    """Load grandcanonical_results.npz and return as plain dict."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"GC results not found: {path}\n"
            "Run: python3 figures/utils/cache.py"
        )
    return load_npz(path)


def load_gc_pbe_novib_view(variants_npz: str, formulas_npz: str) -> dict:
    """PBE grand-canonical without vib/ZPE, keyed like ``load_gc_npz``.

    Built from ``gc_variants.npz`` (novib tensors) plus per-structure ``formulas``
    from the full PBE GC NPZ (same structure order; electronic G only changes weights).
    """
    vd = load_npz(variants_npz)
    formulas = load_npz(formulas_npz)["formulas"]
    return {
        "Ts": vd["Ts"],
        "Ps": vd["Ps"],
        "Pgi": vd["novib_Pgi"],
        "phase_index": vd["novib_phase_index"],
        "x_per_non_o": vd["novib_x_per_non_o"],
        "formulas": formulas,
    }


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
