"""Per-functional inputs of the thermodynamics figures (PBE | PBE0 comparisons).

Each functional has its trajectory, O₂ and Ir references, grand-canonical NPZ, variants NPZ and ACE
features. The GC ensembles hold only the structures that enter the thermodynamics (Ir sizes in
``GC_IR_COUNTS`` and, with vibrations on, only structures with a spectrum); ``gc_structures`` returns
them in GC row order through the ``<npz>_frames.npy`` sidecar written with every cache.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
from ase import Atoms
from ase.io import read

from figures.config import (
    ACE_FEATURES,
    GC_NPZ_BY_DATASET,
    GC_VARIANTS_NPZ_BY_DATASET,
    IR_REF_PBE,
    IR_REF_PBE0,
    O2_REF_PBE,
    O2_REF_PBE0,
    TRAJ_PBE,
    TRAJ_PBE0,
    gc_gas_reservoir_specs,
    gc_grid_config,
)

DATASETS = ("PBE", "PBE0")


@dataclass(frozen=True)
class DatasetInputs:
    label: str
    traj: str
    o2_ref: str
    ir_ref: str
    gc_npz: str
    variants_npz: str
    ace_features: str

    def grid_config(self, **overrides):
        """``GCConfig`` on the figure T–p grid with this functional's Ir reference."""
        return gc_grid_config(reference_file=self.ir_ref, **overrides)


_INPUTS = {
    "PBE": DatasetInputs("PBE", TRAJ_PBE, O2_REF_PBE, IR_REF_PBE, GC_NPZ_BY_DATASET["PBE"],
                         GC_VARIANTS_NPZ_BY_DATASET["PBE"], ACE_FEATURES["PBE"]),
    "PBE0": DatasetInputs("PBE0", TRAJ_PBE0, O2_REF_PBE0, IR_REF_PBE0, GC_NPZ_BY_DATASET["PBE0"],
                          GC_VARIANTS_NPZ_BY_DATASET["PBE0"], ACE_FEATURES["PBE0"]),
}


def dataset_inputs(label: str) -> DatasetInputs:
    return _INPUTS[label]


def gc_frames(npz_path: str) -> np.ndarray:
    """Trajectory frame of every GC row of ``npz_path`` (-1 = gas oxide), from the cache sidecar."""
    path = npz_path[:-4] + "_frames.npy"
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} missing; rebuild the caches (figures/build_all.py --rebuild-cache)")
    return np.load(path)


def gc_structures(label: str, *, npz_path: str | None = None, include_gas: bool = False) -> list[Atoms]:
    """Structures in the GC row order of ``npz_path`` (default: the functional's grand-canonical NPZ),
    gas oxides appended after the clusters as ``merge_gas_oxides`` does when ``include_gas``."""
    inp = dataset_inputs(label)
    traj = read(inp.traj, index=":")
    out = [traj[int(i)] for i in gc_frames(npz_path or inp.gc_npz) if i >= 0]
    if include_gas:
        for path, _ in gc_gas_reservoir_specs(label):
            if os.path.exists(path):
                mol = read(path)
                out.append(mol if isinstance(mol, Atoms) else mol[0])
    return out
