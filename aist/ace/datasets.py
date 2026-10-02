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


def _atom_labels(fracs, n_atoms: int, label: str, row: dict) -> list[int]:
    """Per-atom motif labels of one structure from its CSV motif fractions (counts = fraction x n_atoms).

    Atom order is not kept; the population stacks only count labels.
    """
    counts = np.rint(np.asarray(fracs, dtype=float) * n_atoms).astype(int)
    if counts.sum() != n_atoms:
        raise ValueError(f"{label} frame {row['frame_index']}: motif fractions {fracs} do not give {n_atoms} atoms")
    return np.repeat(np.arange(counts.size), counts).tolist()


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
        per_struct_ir.append(_atom_labels([row[c] for c in ir_cols], int(row["n_Ir"]), label, row))
        per_struct_o.append(_atom_labels([row[c] for c in o_cols], int(row["n_O"]), label, row))

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
