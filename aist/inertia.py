import numpy as np


def get_hirshfeld_volumes(atoms):
    """Return Hirshfeld volumes array from atoms.info, or None if absent/invalid."""
    h = atoms.info.get("hirshfeld", None)
    if not (isinstance(h, dict) and "volumes" in h):
        return None
    volumes = np.asarray(h["volumes"], dtype=float)
    if volumes.size != len(atoms):
        return None
    return volumes


def inertia_tensor_weighted(positions, weights):
    """Compute an inertia tensor using arbitrary per-site weights.

    Positions are centered at the weighted centroid.
    """
    weights = np.asarray(weights, dtype=float)
    total_weight = float(np.sum(weights))
    if total_weight <= 0.0:
        raise ValueError("Total weight must be positive for inertia tensor computation")

    center = np.sum(positions * weights[:, None], axis=0) / total_weight
    positions = positions - center

    I = np.zeros((3, 3))
    for w, (x, y, z) in zip(weights, positions):
        I[0, 0] += w * (y**2 + z**2)
        I[1, 1] += w * (x**2 + z**2)
        I[2, 2] += w * (x**2 + y**2)
        I[0, 1] += -w * x * y
        I[1, 0] += -w * x * y
        I[0, 2] += -w * x * z
        I[2, 0] += -w * x * z
        I[1, 2] += -w * y * z
        I[2, 1] += -w * y * z
    return I


def principal_moments_sorted_mass(atoms, tp):
    """Principal moments (sorted) from the mass-weighted inertia tensor."""
    inertia = tp.get_inertia_tensor(atoms)
    eigvals = np.linalg.eigvalsh(inertia)
    return float(eigvals[0]), float(eigvals[1]), float(eigvals[2])


def principal_moments_sorted_hirshfeld(atoms, volumes):
    """Principal moments (sorted) from a Hirshfeld-volume-weighted inertia tensor."""
    positions = atoms.get_positions()
    inertia = inertia_tensor_weighted(positions, volumes)
    eigvals = np.linalg.eigvalsh(inertia)
    return float(eigvals[0]), float(eigvals[1]), float(eigvals[2])

