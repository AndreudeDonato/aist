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


def inertia_shape_extent_features(
    atoms, tp, *, extent_normalization: str = "partial", extent_blend: float = 0.5
):
    """Rotation-invariant shape + extent descriptors.

    Returns ``(I1/I3, I2/I3, log_extent)`` where ``I1 <= I2 <= I3`` are the
    mass-weighted principal moments of inertia.

    * Shape ``(I1/I3, I2/I3)`` — dimensionless, size-independent Nagai-triangle
      coordinates: ``(1,1)`` spherical, ``(0,1)`` linear, ``(0.5,0.5)`` planar.

    * Extent — partially shape-corrected mass-normalised spread::

          log_extent = (1/2) log(I_tot) - coeff * log(M)

      The coefficient ``coeff`` is controlled by ``extent_normalization``:

      ``"sphere"`` — fixed ``5/6`` (original sphere-only; linear structures are
          systematically overweighted for large N because I_tot ~ M^3 for chains
          but M^{5/3} for spheres, leaving a residual +(2/3)log M trend for L).

      ``"shape"`` — full shape correction: ``coeff = 1/2 + 1/d_eff`` where
          ``d_eff = 1 + 2*r13`` (L→1.5, P→1.0, S→5/6). Removes the N-trend
          exactly but over-corrects: spherical structures now appear more spread
          out than linear ones of the same physical size.

      ``"partial"`` — blend between sphere and full correction:
          ``coeff = 5/6 + b * (1/2 + 1/d_eff - 5/6)`` where ``b = extent_blend``
          (default 0.5). Reduces the between-shape N-scaling bias without
          eliminating it, preserving within-shape extent variation. At b=0.5,
          coefficients are S:5/6, P:11/12, L:7/6 — linear is still slightly
          favoured over spherical but within-class spread dominates.

      ``"none"`` — no mass normalisation: ``log_extent = (1/2) log(I_tot)``.
          Raw inertia scale only; extent grows with both size and spread.

      At any setting except ``"none"``, extent is batch-normalised to [0,1] in
      ``inertia_shape_weighted_by_extent`` before multiplying shape features,
      so the absolute scale does not affect clustering.
    """
    inertia = tp.get_inertia_tensor(atoms)
    I1, I2, I3 = np.linalg.eigvalsh(inertia)
    I_tot = I1 + I2 + I3
    M = float(np.sum(atoms.get_masses()))
    if I3 <= 0.0 or I_tot <= 0.0 or M <= 0.0:
        return 0.0, 0.0, 0.0
    r13 = float(I1 / I3)
    r23 = float(I2 / I3)
    coeff_sphere = 5.0 / 6.0
    d_eff = 1.0 + 2.0 * r13
    coeff_shape = 0.5 + 1.0 / d_eff
    if extent_normalization == "sphere":
        coeff = coeff_sphere
    elif extent_normalization == "shape":
        coeff = coeff_shape
    elif extent_normalization == "partial":
        b = float(extent_blend)
        coeff = coeff_sphere + b * (coeff_shape - coeff_sphere)
    elif extent_normalization == "none":
        coeff = 0.0
    else:
        raise ValueError(
            "extent_normalization must be 'sphere', 'shape', 'partial', or 'none'; "
            f"got {extent_normalization!r}"
        )
    log_extent = 0.5 * float(np.log(I_tot)) - coeff * float(np.log(M))
    return r13, r23, log_extent


def nagai_corner_purity(r13: float, r23: float) -> float:
    """Quadratic proximity to a pure Nagai corner (S, L, or P), in [0, 1].

    Each corner score is a product of squared barycentric-style factors peaked
    at the respective vertex: S ``(1,1)``, L ``(0,1)``, P ``(0.5,0.5)``.
    Returns the maximum over the three corners — high only when shape is
    close to spherical, linear, or planar, not in the ambiguous interior.
    """
    r13 = float(np.clip(r13, 0.0, 1.0))
    r23 = float(np.clip(r23, 0.0, 1.0))
    w_s = r13 ** 2 * r23 ** 2
    w_l = (1.0 - r13) ** 2 * r23 ** 2
    w_p = 16.0 * r13 * (1.0 - r13) * r23 * (1.0 - r23)
    return float(max(w_s, w_l, w_p))


def nagai_barycentric_weights(r13: float, r23: float) -> tuple[float, float, float]:
    """Barycentric weights toward Nagai corners L, P, S from ``(I1/I3, I2/I3)``.

    Vertices: L ``(0,1)``, P ``(0.5,0.5)``, S ``(1,1)``.  Returns
    ``(w_L, w_P, w_S)`` with sum 1 for points inside the triangle.
    """
    r13 = float(np.clip(r13, 0.0, 1.0))
    r23 = float(np.clip(r23, 0.0, 1.0))
    w_p = 2.0 * (1.0 - r23)
    w_s = r13 - (1.0 - r23)
    w_l = r23 - r13
    w_l = max(0.0, w_l)
    w_p = max(0.0, w_p)
    w_s = max(0.0, w_s)
    total = w_l + w_p + w_s
    if total <= 0.0:
        return 0.0, 0.0, 0.0
    return w_l / total, w_p / total, w_s / total


def shape_extent_display_rgb(
    r13: float,
    r23: float,
    extent_norm: float,
) -> tuple[float, float, float]:
    """RGB for inertia shape + extent: R=L, G=P, B=S; lightness from extent.

    ``extent_norm`` in [0, 1] linearly blends from white (low extent) to the
    pure shape colour (high extent): ``rgb = (1-t) + t * (w_L, w_P, w_S)``.
    """
    t = float(np.clip(extent_norm, 0.0, 1.0))
    w_l, w_p, w_s = nagai_barycentric_weights(r13, r23)
    return (1.0 - t) + t * w_l, (1.0 - t) + t * w_p, (1.0 - t) + t * w_s


def per_structure_shape_extent_rgb(
    structures,
    tp,
    *,
    extent_normalization: str = "partial",
    extent_blend: float = 0.1,
) -> np.ndarray:
    """Per-structure RGB ``(n, 3)`` for Nagai shape (R=L,G=P,B=S) × extent lightness."""
    if not structures:
        return np.zeros((0, 3), dtype=float)
    raw = [
        inertia_shape_extent_features(
            a, tp,
            extent_normalization=extent_normalization,
            extent_blend=extent_blend,
        )
        for a in structures
    ]
    log_e = np.array([r[2] for r in raw], dtype=float)
    e_min, e_max = float(log_e.min()), float(log_e.max())
    denom = max(e_max - e_min, 1e-12)
    extent_norm = (log_e - e_min) / denom
    out = np.zeros((len(raw), 3), dtype=float)
    for i, (r13, r23, _) in enumerate(raw):
        out[i] = shape_extent_display_rgb(r13, r23, extent_norm[i])
    return out


def inertia_shape_weighted_by_extent(
    structures,
    tp,
    *,
    alpha: float = 1.0,
    extent_normalization: str = "partial",
    extent_blend: float = 0.5,
    corner_purity_beta: float = 0.0,
) -> np.ndarray:
    """Shape features (I1/I3, I2/I3) scaled per structure by extent and corner purity.

    Extent is batch-normalised ``log_extent`` from ``inertia_shape_extent_features``.
    Compact structures get extent_norm ~ 0 so shape barely enters clustering;
    spread-out structures get extent_norm ~ 1.

  Optional corner boost: when ``corner_purity_beta > 0``, weight is further
    multiplied by ``1 + corner_purity_beta * purity**2`` where ``purity`` is
    ``nagai_corner_purity(r13, r23)`` — quadratic proximity to S/L/P corners.
    Mid-triangle shapes stay near unit boost; pure-like shapes gain extra weight
    without dominating unless beta is large.

    Returns ``(n_struct, 2)`` — extent is *not* appended as its own column.
    """
    if not structures:
        return np.zeros((0, 2), dtype=float)
    raw = [
        inertia_shape_extent_features(
            a, tp,
            extent_normalization=extent_normalization,
            extent_blend=extent_blend,
        )
        for a in structures
    ]
    log_e = np.array([r[2] for r in raw], dtype=float)
    e_min, e_max = float(log_e.min()), float(log_e.max())
    denom = max(e_max - e_min, 1e-12)
    extent_norm = (log_e - e_min) / denom
    beta = float(corner_purity_beta)
    out = np.zeros((len(raw), 2), dtype=float)
    for i, (r13, r23, _) in enumerate(raw):
        purity = nagai_corner_purity(r13, r23)
        corner_boost = 1.0 + beta * purity ** 2
        w = float(extent_norm[i]) * float(alpha) * corner_boost
        out[i, 0] = r13 * w
        out[i, 1] = r23 * w
    return out

