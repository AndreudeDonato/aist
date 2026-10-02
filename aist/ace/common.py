"""Label helpers + discrete colour palettes for ACE atom / structure clusters."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def ace_motif_subscript(cluster_id: int) -> str:
    """Lowercase letter subscript for motif index (a, b, c, …)."""
    i = int(cluster_id)
    if 0 <= i < 26:
        return chr(ord("a") + i)
    return str(i)


def ace_atom_label(species: str, cluster_id: int) -> str:
    """Internal motif key (JSON / palette): ``Ir-A0``, ``O-A1``, …"""
    return f"{species}-A{int(cluster_id)}"


def ace_atom_display_label(species: str, cluster_id: int) -> str:
    """Matplotlib mathtext label for figures: Ir_a, O_b, …"""
    sub = ace_motif_subscript(cluster_id)
    return f"$\\mathrm{{{species}}}_{{{sub}}}$"


def ace_atom_display_from_key(key: str) -> str:
    """Map internal key ``Ir-A0`` to a figure display label."""
    if "-A" not in key:
        return key
    species, rest = key.split("-A", 1)
    try:
        return ace_atom_display_label(species, int(rest))
    except ValueError:
        return key


def discrete_palette(cmap_name: str, k: int):
    """``k`` evenly-spaced colours sampled from a matplotlib colormap."""
    cmap = plt.get_cmap(cmap_name)
    if k <= 1:
        return [cmap(0.5)]
    return [cmap(i / (k - 1)) for i in range(k)]


def struct_cluster_palette(k: int):
    """Qualitative palette for structure-level clusters (tab10 / tab20)."""
    return plt.get_cmap("tab10" if k <= 10 else "tab20")


DATASET_COLORS = {
    "PBE": "#2166ac",
    "PBE0": "#b2182b",
}


def composition_features_for_struct(
    n_structures: int,
    per_struct_atom_clusters: list[np.ndarray],
    species: str,
    k_clusters: int,
) -> np.ndarray:
    """Per-structure fractions of each atom cluster for one species."""
    _ = species  # kept for symmetry with caller; KMeans never produces noise
    out = np.zeros((n_structures, k_clusters), dtype=float)
    for si in range(n_structures):
        labs = np.asarray(per_struct_atom_clusters[si], dtype=int)
        if labs.size == 0:
            continue
        counts = np.bincount(labs[labs >= 0], minlength=k_clusters).astype(float)
        total = counts.sum()
        if total > 0:
            out[si] = counts / total
    return out
