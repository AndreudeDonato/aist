"""Bond-orientational (Steinhardt) order parameters for finite clusters.

Implements per-atom :math:`q_l` and Lechner-Dellago averaged :math:`\\bar{q}_l`
using a chemical neighbor list, suitable for non-periodic molecular clusters
(e.g. cation-centered local environments in oxide nanoclusters):

.. math::

    q_{lm}(i)      &= \\frac{1}{N_b(i)} \\sum_{j \\in N_b(i)} Y_{lm}(\\hat r_{ij}) \\\\
    q_l(i)         &= \\sqrt{ \\frac{4\\pi}{2l+1} \\sum_{m=-l}^{l} |q_{lm}(i)|^2 } \\\\
    \\bar q_{lm}(i) &= \\frac{1}{N_s(i)+1}\\Big( q_{lm}(i) + \\sum_{k \\in N_s(i)} q_{lm}(k) \\Big) \\\\
    \\bar q_l(i)    &= \\sqrt{ \\frac{4\\pi}{2l+1} \\sum_{m=-l}^{l} |\\bar q_{lm}(i)|^2 }

where :math:`N_b(i)` is the *ligand* neighbor list defining the local environment
and :math:`N_s(i)` is a same-species shell list used to smooth the parameter
(Lechner & Dellago, *J. Chem. Phys.* 129, 114707 (2008)).

For Ir centers in IrO\\ :sub:`x`, the natural choice is

* :math:`N_b(i)` = O atoms within an Ir-O bond cutoff,
* :math:`N_s(i)` = Ir atoms within an Ir-Ir cutoff (or all Ir in the finite cluster).

The module is dependency-light: ``numpy``, ``scipy.special.sph_harm_y``,
``scipy.stats.gaussian_kde`` and ``ase``.
"""

from typing import Iterable, Sequence

import numpy as np
from scipy.signal import find_peaks
from scipy.special import sph_harm_y
from scipy.stats import gaussian_kde


__all__ = [
    "spherical_angles",
    "qlm_from_neighbors",
    "ql_from_qlm",
    "averaged_qlm",
    "neighbor_indices_within_cutoff",
    "displacement_vectors",
    "compute_steinhardt_per_center",
    "structure_summary_features",
    "derive_pair_cutoff_kde",
]


# ---------------------------------------------------------------------------
# Basic spherical-harmonic helpers
# ---------------------------------------------------------------------------


def spherical_angles(vec_xyz: np.ndarray):
    """Polar ``theta`` in [0, pi] and azimuth ``phi`` in [0, 2*pi].

    Follows SciPy's ``sph_harm_y`` convention: ``theta`` is the polar
    (colatitudinal) angle and ``phi`` the azimuthal angle.

    Parameters
    ----------
    vec_xyz : array_like, shape (..., 3)
        Cartesian displacement vectors.

    Returns
    -------
    theta : ndarray
    phi   : ndarray
    """
    vec = np.asarray(vec_xyz, dtype=float)
    x = vec[..., 0]
    y = vec[..., 1]
    z = vec[..., 2]
    r = np.linalg.norm(vec, axis=-1)
    safe = np.where(r > 0.0, r, 1.0)
    theta = np.arccos(np.clip(z / safe, -1.0, 1.0))
    phi = np.arctan2(y, x) % (2.0 * np.pi)
    return theta, phi


def qlm_from_neighbors(neighbor_vectors: np.ndarray, l: int) -> np.ndarray:
    """q_lm(i) given displacement vectors r_ij from a center i to its neighbors j.

    Parameters
    ----------
    neighbor_vectors : (Nb, 3) array
        Cartesian displacement vectors from the central atom to each neighbor.
    l : int
        Spherical-harmonic degree.

    Returns
    -------
    qlm : (2l+1,) complex array, indexed m = -l, ..., l.
    """
    vecs = np.asarray(neighbor_vectors, dtype=float)
    nb = vecs.shape[0]
    out = np.zeros(2 * l + 1, dtype=complex)
    if nb == 0:
        return out
    theta, phi = spherical_angles(vecs)
    for k, m in enumerate(range(-l, l + 1)):
        out[k] = np.mean(sph_harm_y(l, m, theta, phi))
    return out


def ql_from_qlm(qlm: np.ndarray, l: int) -> float:
    """Scalar q_l = sqrt( 4*pi / (2l+1) * sum_m |q_lm|^2 )."""
    qlm = np.asarray(qlm)
    return float(np.sqrt(4.0 * np.pi / (2.0 * l + 1.0) * np.sum(np.abs(qlm) ** 2)))


def averaged_qlm(qlm_self: np.ndarray, qlm_shell: np.ndarray) -> np.ndarray:
    """Lechner-Dellago q_bar_lm(i) = ( q_lm(i) + sum_k q_lm(k) ) / (Ns + 1).

    ``qlm_shell`` has shape (Ns, 2l+1); ``Ns`` may be 0 in which case the result
    equals ``qlm_self``.
    """
    qlm_shell = np.asarray(qlm_shell)
    ns = qlm_shell.shape[0]
    if ns == 0:
        return np.asarray(qlm_self).copy()
    return (np.asarray(qlm_self) + qlm_shell.sum(axis=0)) / (ns + 1)


# ---------------------------------------------------------------------------
# Neighbor utilities (chemical, non-periodic friendly)
# ---------------------------------------------------------------------------


def neighbor_indices_within_cutoff(
    atoms,
    center_index: int,
    neighbor_symbol: str,
    cutoff: float,
) -> np.ndarray:
    """Indices of ``neighbor_symbol`` atoms within ``cutoff`` of atom ``center_index``.

    Excludes the center itself. Uses ASE's minimum-image convention via
    :func:`ase.Atoms.get_distances`.
    """
    symbols = np.asarray(atoms.get_chemical_symbols())
    mask = (symbols == neighbor_symbol) & (np.arange(len(atoms)) != center_index)
    candidates = np.where(mask)[0]
    if candidates.size == 0:
        return candidates
    d = atoms.get_distances(center_index, candidates, mic=True)
    return candidates[d <= cutoff]


def displacement_vectors(atoms, center_index: int, neighbor_indices: np.ndarray) -> np.ndarray:
    """Vectors r_ij = r_j - r_i for j in neighbor_indices (MIC-aware)."""
    neighbor_indices = np.asarray(neighbor_indices)
    if neighbor_indices.size == 0:
        return np.zeros((0, 3), dtype=float)
    return atoms.get_distances(center_index, neighbor_indices, mic=True, vector=True)


# ---------------------------------------------------------------------------
# Per-center and per-structure Steinhardt features
# ---------------------------------------------------------------------------


def compute_steinhardt_per_center(
    atoms,
    center_symbol: str,
    ligand_symbol: str,
    bond_cutoff_ligand: float,
    shell_cutoff_same: float = np.inf,
    ls: Sequence[int] = (2, 4, 6),
    averaged_ls: Sequence[int] = (4, 6),
):
    """Compute q_l and q_bar_l for all atoms of ``center_symbol`` in ``atoms``.

    Parameters
    ----------
    atoms : ase.Atoms
    center_symbol : str
        Element used as the center of the local environment (e.g. ``"Ir"``).
    ligand_symbol : str
        Element used as the bonded neighbor (e.g. ``"O"``).
    bond_cutoff_ligand : float
        Distance cutoff defining the center-ligand bond (A).
    shell_cutoff_same : float, optional
        Distance cutoff between two centers for q_bar_l averaging (A).
        Default ``np.inf`` averages over all centers in the structure, which
        is the natural choice for a finite cluster.
    ls : tuple of int
        Steinhardt orders to compute as raw q_l.
    averaged_ls : tuple of int
        Steinhardt orders to compute as Lechner-Dellago q_bar_l.

    Returns
    -------
    dict with keys

    ``center_indices`` : (n,) int array of global atom indices in ``atoms``.
    ``names`` : list of feature names, ordered like
        ``[f"q{l}" for l in ls] + [f"qbar{l}" for l in averaged_ls] + ["coord"]``.
    ``features`` : (n, len(names)) float array.
    ``coord`` : (n,) int array (number of ligands within ``bond_cutoff_ligand``).
    """
    symbols = np.asarray(atoms.get_chemical_symbols())
    center_indices = np.where(symbols == center_symbol)[0]
    n_centers = center_indices.size
    names = [f"q{l}" for l in ls] + [f"qbar{l}" for l in averaged_ls] + ["coord"]

    if n_centers == 0:
        return {
            "center_indices": center_indices,
            "names": names,
            "features": np.zeros((0, len(names))),
            "coord": np.zeros(0, dtype=int),
        }

    all_ls = sorted(set(ls) | set(averaged_ls))
    qlm_per_center = {l: np.zeros((n_centers, 2 * l + 1), dtype=complex) for l in all_ls}
    coord = np.zeros(n_centers, dtype=int)
    for k, idx in enumerate(center_indices):
        lig_idx = neighbor_indices_within_cutoff(atoms, idx, ligand_symbol, bond_cutoff_ligand)
        coord[k] = lig_idx.size
        vecs = displacement_vectors(atoms, idx, lig_idx)
        for l in all_ls:
            qlm_per_center[l][k] = qlm_from_neighbors(vecs, l)

    # Same-species shell membership (local indices into center_indices)
    shell_membership = [np.zeros(0, dtype=int) for _ in range(n_centers)]
    if n_centers > 1:
        for k, idx in enumerate(center_indices):
            other_local = np.delete(np.arange(n_centers), k)
            other_global = center_indices[other_local]
            d = atoms.get_distances(idx, other_global, mic=True)
            shell_membership[k] = other_local[d <= shell_cutoff_same]

    feature_arrays = {}
    for l in ls:
        feature_arrays[f"q{l}"] = np.array(
            [ql_from_qlm(qlm_per_center[l][k], l) for k in range(n_centers)],
            dtype=float,
        )
    for l in averaged_ls:
        bar = np.zeros((n_centers, 2 * l + 1), dtype=complex)
        for k in range(n_centers):
            shell = shell_membership[k]
            qlm_shell = (
                qlm_per_center[l][shell]
                if shell.size > 0
                else np.zeros((0, 2 * l + 1), dtype=complex)
            )
            bar[k] = averaged_qlm(qlm_per_center[l][k], qlm_shell)
        feature_arrays[f"qbar{l}"] = np.array(
            [ql_from_qlm(bar[k], l) for k in range(n_centers)],
            dtype=float,
        )
    feature_arrays["coord"] = coord.astype(float)

    features = np.column_stack([feature_arrays[name] for name in names])
    return {
        "center_indices": center_indices,
        "names": names,
        "features": features,
        "coord": coord,
    }


def structure_summary_features(per_center_features: dict):
    """Aggregate per-center features into a per-structure fingerprint.

    For each per-center feature column the function keeps its mean and standard
    deviation across the centers in the structure. If a structure has no centers,
    a NaN vector of the appropriate length is returned.

    Parameters
    ----------
    per_center_features : dict
        Output of :func:`compute_steinhardt_per_center`.

    Returns
    -------
    names : list[str]
        Like ``["q2_mean", "q2_std", "q4_mean", "q4_std", ..., "coord_mean", "coord_std"]``.
    values : (2 * n_features,) float array
    """
    names_in = per_center_features["names"]
    features = per_center_features["features"]
    names_out: list = []
    for name in names_in:
        names_out.append(f"{name}_mean")
        names_out.append(f"{name}_std")
    if features.shape[0] == 0:
        return names_out, np.full(2 * len(names_in), np.nan)
    means = np.mean(features, axis=0)
    stds = np.std(features, axis=0)
    values = np.empty(2 * len(names_in))
    values[0::2] = means
    values[1::2] = stds
    return names_out, values


# ---------------------------------------------------------------------------
# Bond cutoff derivation from a trajectory (KDE-peak based, like bondanalysis.py)
# ---------------------------------------------------------------------------


def _peak_index_from_kde(
    density: np.ndarray,
    *,
    mode: str = "global",
    min_prominence_frac: float = 0.01,
) -> int:
    """Return the grid index of the KDE peak used for bond-cutoff derivation.

    ``global``
        Tallest KDE maximum (``np.argmax``). This is what ``bondanalysis.py``
        uses. For O–O it often selects an *environmental* shell (~2.8 Å,
        O atoms sharing an Ir centre) because that population dominates.

    ``first_local``
        Leftmost local maximum with prominence at least
        ``min_prominence_frac * max(density)``. For Ir–O and Ir–Ir this
        coincides with the global maximum; for O–O it selects the shorter
        direct-bond feature (~1.4 Å) instead of the taller environmental peak.
    """
    k_global = int(np.argmax(density))
    if mode == "global":
        return k_global
    if mode == "first_local":
        y_max = float(density.max())
        if y_max <= 0.0:
            return k_global
        peaks, _ = find_peaks(
            density,
            prominence=y_max * float(min_prominence_frac),
        )
        if peaks.size == 0:
            return k_global
        return int(peaks[0])
    raise ValueError(f"Unknown peak mode {mode!r}; expected 'global' or 'first_local'.")


def derive_pair_cutoff_kde(
    structures: Iterable,
    symbol_a: str,
    symbol_b: str,
    x_min: float = 0.5,
    x_max: float = 5.0,
    kde_grid_points: int = 1000,
    broaden: float = 2.0,
    peak_mode: str = "global",
    min_peak_prominence_frac: float = 0.01,
):
    """Derive an upper bond cutoff for atom pair ``(symbol_a, symbol_b)``.

    Collects every ``symbol_a``-``symbol_b`` interatomic distance in
    ``[x_min, x_max]`` across all structures, builds a Gaussian KDE, locates
    the bond peak (see ``peak_mode``), and finds the half-maximum on the
    *left* side. The half-width is ``x_peak - x_half_left``; the returned
    upper cutoff is ``x_peak + broaden * half_width``.

    Parameters
    ----------
    structures : iterable of ase.Atoms
    symbol_a, symbol_b : str
    x_min, x_max : float
        Distance window to scan (A).
    kde_grid_points : int
        Number of points used to evaluate the KDE.
    broaden : float
        Multiplicative factor applied to the left half-width to define how far
        beyond the peak the cutoff extends. ``2.0`` matches the broadening used
        in ``plot_bond_counts_datasets.py`` / ``plot_bridging_dangling.py``.
    peak_mode : {"global", "first_local"}
        How to pick ``x_peak`` on the KDE grid. ``global`` uses the tallest
        maximum (same as ``bondanalysis.py``). ``first_local`` uses the
        leftmost local maximum above a prominence threshold — appropriate when
        a shorter true bond peak is overshadowed by a taller environmental
        shell (typical for O–O in oxide clusters).
    min_peak_prominence_frac : float
        Minimum ``scipy.signal.find_peaks`` prominence as a fraction of the
        global KDE maximum; only used when ``peak_mode="first_local"``.

    Returns
    -------
    cutoff : float
        Upper limit of the accepted bond window (A).
    info : dict
        Diagnostics containing ``x_peak``, ``y_peak``, ``x_half_left``,
        ``half_width``, ``broaden``, ``cutoff_upper``, ``peak_mode``,
        ``x_global_peak`` and ``n_distances``.
    """
    distances: list = []
    for atoms in structures:
        symbols = np.asarray(atoms.get_chemical_symbols())
        idx_a = np.where(symbols == symbol_a)[0]
        if symbol_a == symbol_b:
            idx_b = idx_a
            if idx_a.size < 2:
                continue
            for i_pos in range(idx_a.size):
                for j_pos in range(i_pos + 1, idx_a.size):
                    d = float(atoms.get_distance(int(idx_a[i_pos]), int(idx_a[j_pos]), mic=True))
                    if x_min <= d <= x_max:
                        distances.append(d)
        else:
            idx_b = np.where(symbols == symbol_b)[0]
            if idx_a.size == 0 or idx_b.size == 0:
                continue
            for i in idx_a:
                for j in idx_b:
                    d = float(atoms.get_distance(int(i), int(j), mic=True))
                    if x_min <= d <= x_max:
                        distances.append(d)

    distances_arr = np.asarray(distances, dtype=float)
    if distances_arr.size < 5:
        raise ValueError(
            f"Not enough {symbol_a}-{symbol_b} distances in [{x_min},{x_max}] "
            f"to derive a cutoff ({distances_arr.size} found)."
        )

    kde = gaussian_kde(distances_arr)
    grid = np.linspace(x_min, x_max, int(kde_grid_points))
    density = kde(grid)
    k_global = int(np.argmax(density))
    k_peak = _peak_index_from_kde(
        density,
        mode=peak_mode,
        min_prominence_frac=min_peak_prominence_frac,
    )
    x_peak = float(grid[k_peak])
    y_peak = float(density[k_peak])
    x_global_peak = float(grid[k_global])

    half = 0.5 * y_peak
    left_density = density[: k_peak + 1]
    left_grid = grid[: k_peak + 1]
    below = np.where(left_density <= half)[0]
    x_half_left = float(left_grid[below[-1]]) if below.size > 0 else float(left_grid[0])
    half_width = x_peak - x_half_left
    cutoff_upper = x_peak + broaden * half_width

    return cutoff_upper, {
        "x_peak": x_peak,
        "y_peak": y_peak,
        "x_half_left": x_half_left,
        "half_width": half_width,
        "broaden": broaden,
        "cutoff_upper": cutoff_upper,
        "peak_mode": peak_mode,
        "x_global_peak": x_global_peak,
        "n_distances": int(distances_arr.size),
    }
