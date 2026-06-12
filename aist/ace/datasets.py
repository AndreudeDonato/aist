"""Load ACE-labelled datasets from trajectory + features CSV."""
from __future__ import annotations

import csv

import numpy as np
from ase.io import read


def load_ace_features_csv(path: str) -> list[dict]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["frame_index"] = int(row["frame_index"])
        row["n_Ir"] = int(row["n_Ir"])
        row["n_O"] = int(row["n_O"])
        row["cluster"] = int(row["cluster"])
        for k in list(row.keys()):
            if k.endswith("_frac"):
                row[k] = float(row[k])
    return rows


def load_dataset(label: str, traj_path: str, features_csv: str) -> dict:
    """Build a dataset dict compatible with aist.ace.plotting population helpers."""
    structures = list(read(traj_path, ":"))
    rows = load_ace_features_csv(features_csv)
    if len(structures) != len(rows):
        raise ValueError(
            f"{label}: traj has {len(structures)} frames, CSV has {len(rows)} rows"
        )

    ir_cols = sorted([k for k in rows[0] if k.startswith("Ir-") and k.endswith("_frac")])
    o_cols = sorted([k for k in rows[0] if k.startswith("O-") and k.endswith("_frac")])
    k_ir, k_o = len(ir_cols), len(o_cols)

    per_struct_ir = []
    per_struct_o = []
    for row in rows:
        ir_fracs = np.array([row[c] for c in ir_cols])
        o_fracs = np.array([row[c] for c in o_cols])
        n_ir, n_o = int(row["n_Ir"]), int(row["n_O"])
        ir_lab = int(np.argmax(ir_fracs)) if ir_fracs.sum() > 0 else 0
        o_lab = int(np.argmax(o_fracs)) if o_fracs.sum() > 0 and n_o > 0 else 0
        per_struct_ir.append([ir_lab] * max(n_ir, 1))
        per_struct_o.append([o_lab] * max(n_o, 1) if n_o > 0 else [])

    return {
        "label": label,
        "structures": structures,
        "n_Ir": np.array([r["n_Ir"] for r in rows], dtype=int),
        "n_O": np.array([r["n_O"] for r in rows], dtype=int),
        "struct_labels": np.array([r["cluster"] for r in rows], dtype=int),
        "per_struct_ir_cluster": per_struct_ir,
        "per_struct_o_cluster": per_struct_o,
        "k_ir": k_ir,
        "k_o": k_o,
    }
