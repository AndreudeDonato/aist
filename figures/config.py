"""Central configuration for all paper figure scripts.

Data policy
-----------
* **Inputs only** from ``<repo>/data/`` (``RAW_DATA_DIR``). Nothing else is read as source data.
* **Everything derived** (ACE clustering, grand-canonical NPZ, GC variants, minima trajs)
  is written under ``figures/cache/`` and rebuilt from ``data/`` when missing or stale.
* **Figure SVGs** go to ``figures/outputs/``.

Run ``python3 figures/build_all.py`` (or ``figures/utils/cache.py``) to refresh the cache.
"""
from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# Repository layout
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Raw inputs  (user-provided; do not commit large binaries if policy requires)
# ---------------------------------------------------------------------------
RAW_DATA_DIR = os.path.join(ROOT, "data")

TRAJ_PBE = os.path.join(RAW_DATA_DIR, "IryOxPBE.traj")
TRAJ_PBE0 = os.path.join(RAW_DATA_DIR, "IryOxPBE0.traj")
O2_REF_PBE = os.path.join(RAW_DATA_DIR, "O2_PBE.xyz")
O2_REF_PBE0 = os.path.join(RAW_DATA_DIR, "O2_PBE0.xyz")
IR_REF_PBE = os.path.join(RAW_DATA_DIR, "IrPBE.xyz")
IR_REF_PBE0 = os.path.join(RAW_DATA_DIR, "IrPBE0.xyz")

REQUIRED_RAW_FILES = (
    TRAJ_PBE,
    TRAJ_PBE0,
    O2_REF_PBE,
    O2_REF_PBE0,
    IR_REF_PBE,
    IR_REF_PBE0,
)

# Grand-canonical / thermo figures: full PBE structures
GC_STRUCTURES = TRAJ_PBE
GC_N_T = 150
GC_N_P = 150

# ---------------------------------------------------------------------------
# Derived cache  (produced by figures/utils/cache.py from data/ only)
# ---------------------------------------------------------------------------
CACHE_DIR = os.path.join(FIGURES_DIR, "cache")
OUTPUT_DIR = os.path.join(FIGURES_DIR, "outputs")

ACE_ROOT = os.path.join(CACHE_DIR, "ace")
ACE_DIR = os.path.join(ACE_ROOT, "atoms")
ACE_PLOTS_DIR = os.path.join(ACE_ROOT, "plots")

ACE_FEATURES = {
    "PBE": os.path.join(ACE_DIR, "ace_features_PBE.csv"),
    "PBE0": os.path.join(ACE_DIR, "ace_features_PBE0.csv"),
}
ACE_CENTROIDS = {
    "PBE": os.path.join(ACE_DIR, "ace_centroids_PBE.traj"),
    "PBE0": os.path.join(ACE_DIR, "ace_centroids_PBE0.traj"),
    "pooled": os.path.join(ACE_DIR, "ace_centroids_pooled.traj"),
}
ACE_GROUPS = {
    "PBE": os.path.join(ACE_DIR, "ace_groups_PBE"),
    "PBE0": os.path.join(ACE_DIR, "ace_groups_PBE0"),
    "pooled": os.path.join(ACE_DIR, "ace_groups_pooled"),
}
MOTIF_JSON = {
    "PBE": os.path.join(ACE_DIR, "motif_atoms_PBE.json"),
    "PBE0": os.path.join(ACE_DIR, "motif_atoms_PBE0.json"),
    "pooled": os.path.join(ACE_DIR, "motif_atoms_pooled.json"),
}
VISUALIZE_SCRIPT = os.path.join(ACE_DIR, "visualize_motifs.py")

ACE_ATOM_IR_NPZ = os.path.join(ACE_ROOT, "ace_atom_ir.npz")
ACE_ATOM_O_NPZ = os.path.join(ACE_ROOT, "ace_atom_o.npz")
ACE_STRUCT_NPZ = os.path.join(ACE_ROOT, "ace_struct_embeddings.npz")

GC_NPZ = os.path.join(CACHE_DIR, "grandcanonical_results.npz")
GC_NPZ_PBE0 = os.path.join(CACHE_DIR, "grandcanonical_PBE0.npz")
GC_NPZ_BY_DATASET = {"PBE": GC_NPZ, "PBE0": GC_NPZ_PBE0}
GC_VARIANTS_NPZ = os.path.join(CACHE_DIR, "gc_variants.npz")

# Grand-canonical vib/ZPE on product and μ terms (per dataset NPZ cache).
# PBE NPZ (grandcanonical_results.npz) includes vib for Figures 7–8, 10, etc.
# Figure 5 PBE column uses novib tensors from gc_variants.npz (fair vs PBE0).
# PBE0 frequencies unfinished → electronic-only GC until traj is complete.
GC_USE_VIB_ZPE_BY_DATASET = {"PBE": True, "PBE0": False}

MINIMA_TRAJ = {
    "PBE": os.path.join(CACHE_DIR, "minima_PBE.traj"),
    "PBE0": os.path.join(CACHE_DIR, "minima_PBE0.traj"),
}

# ---------------------------------------------------------------------------
# Dataset strategy (vib / GC figures on PBE)
# ---------------------------------------------------------------------------
VIB_DATASET = "PBE"
VIB_TRAJ = TRAJ_PBE
VIB_ACE_FEATURES = ACE_FEATURES["PBE"]
VIB_O2_REF = O2_REF_PBE
VIB_GC_NPZ = GC_NPZ
GC_ACE_FEATURES = ACE_FEATURES["PBE"]

# ---------------------------------------------------------------------------
# Ambient O₂ partial pressure (~20% of 1 atm dry air)
ATMOSPHERIC_PRESSURE_PA = 101325.0
AMBIENT_PO2_FRACTION = 0.20
AMBIENT_PO2_PA = ATMOSPHERIC_PRESSURE_PA * AMBIENT_PO2_FRACTION

# Fixed thermodynamic points (Figures 9 population lines)
FIXED_POINTS = [
    {"label": "273K_ambient_pO2", "T": 273.0, "P": AMBIENT_PO2_PA},
    {"label": "500K_ambient_pO2", "T": 500.0, "P": AMBIENT_PO2_PA},
]

# Figure 6: motif populations vs T at ambient pO₂ (Pa), vertical 88 mm
FIG06_PRESSURE_PA = AMBIENT_PO2_PA
FIG06_SMOOTH_SIGMA = 1.0
FIG06_POP_ZOOM_MIN = 60.0   # O terminal/bridging panel: y from 60% to 100%
# O motif indices (SOAP–UMAP clustering): O-A1 ≈ terminal, O-A0 ≈ bridging
FIG06_O_TERMINAL_IDX = 1
FIG06_O_BRIDGING_IDX = 0
FIG06_SPIN_XMAX = 1.0   # |Hirshfeld spin| display limit for O-spin KDE panels

# Figure 8: 1D ΔG-contribution cut at this pO₂ (Pa)
FIG08_PRESSURE_PA = AMBIENT_PO2_PA

# ---------------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------------
IR_CMAP = "coolwarm"
O_CMAP = "plasma_r"
PHASE_CMAP = "coolwarm"
DATASET_COLORS = {"PBE": "#2166ac", "PBE0": "#b2182b"}
DPI = 800

# ACE clustering reproducibility (must match ``aist.ace.constants``)
ACE_RANDOM_STATE = 42
ACE_SILHOUETTE_SAMPLE = 5000


def gc_grid_config(**overrides):
    """``GCConfig`` with the figure-pipeline T–p grid (default 400×400).

    Override per-term flags, e.g. ``include_v_product``, ``include_zpe_mu``, etc.
    """
    from aist.core.grandcanonical import GCConfig

    params = {"n_T": GC_N_T, "n_P": GC_N_P}
    params.update(overrides)
    return GCConfig(**params)


def gc_cache_config(**overrides):
    """GC config for ``grandcanonical_results.npz`` / ``grandcanonical_PBE0.npz``.

  Pass ``use_vib_zpe`` explicitly, or rely on ``GC_USE_VIB_ZPE_BY_DATASET`` via cache builder.
    """
    use_vib = overrides.pop("use_vib_zpe", False)
    thermo = {
        "include_v_product": use_vib,
        "include_zpe_product": use_vib,
        "include_v_mu": use_vib,
        "include_zpe_mu": use_vib,
    }
    thermo.update(overrides)
    return gc_grid_config(include_decomposition=True, **thermo)
