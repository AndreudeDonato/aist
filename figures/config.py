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
IRO2_REF_PBE = os.path.join(RAW_DATA_DIR, "IrO2_PBE.traj")
IRO3_REF_PBE = os.path.join(RAW_DATA_DIR, "IrO3_PBE.traj")
IRO2_REF_PBE0 = os.path.join(RAW_DATA_DIR, "IrO2_PBE0.traj")
IRO3_REF_PBE0 = os.path.join(RAW_DATA_DIR, "IrO3_PBE0.traj")

IRO2_RUTILE_PBE = os.path.join(RAW_DATA_DIR, "IrO2_rutile_PBE.xyz")
IRO2_RUTILE_PBE0 = os.path.join(RAW_DATA_DIR, "IrO2_rutile_PBE0.xyz")

# Main-cluster GC gas pressures (Pa): fixed trace pressure, identical for both functionals.
GC_GAS_P = {
    "PBE": 1.0,
    "PBE0": 1.0,
}

# Gas-reservoir thermo on IrO₂(g) / IrO₃(g) rows (figures patch after ``run_gc``).
GC_GAS_INCLUDE_VIB = True
GC_GAS_INCLUDE_ROT_TRANS = True

# Support-stabilization proxy for the (approximately supported) clusters: every
# cluster row is shifted by G -= EV_PER_IR + EV_PER_O * (n_O/n_Ir) per Ir atom,
# i.e. -(EV_PER_IR*n_Ir + EV_PER_O*n_O) per cluster. Gas-reservoir rows are
# unaffected. Per-Ir correction spans ~0.65–0.95 eV for typical n_O/n_Ir.
GC_APPLY_SUPPORT_STAB = True
GC_SUPPORT_STAB_EV_PER_IR = 0.65
GC_SUPPORT_STAB_EV_PER_O = 0.1

# Rutile supplementary GC: same fixed trace gas pressure as main-cluster GC.
RUTILE_GC_GAS_P = 1.0

# Anchor (T, pO₂) marked on ``supp_rutile_gc_phase_maps``; ΔG at ``RUTILE_GC_GAS_P`` is stored.
RUTILE_GC_CALIBRATION_T_K = 1150.0

GC_GAS_OXIDES = {
    "PBE": [(IRO2_REF_PBE, GC_GAS_P["PBE"]), (IRO3_REF_PBE, GC_GAS_P["PBE"])],
    "PBE0": [(IRO2_REF_PBE0, GC_GAS_P["PBE0"]), (IRO3_REF_PBE0, GC_GAS_P["PBE0"])],
}

# Rutile supplementary GC: separate T–p grid (see also ``gc_rutile_grid_config``).
RUTILE_GC_T_MIN = 100.0
RUTILE_GC_T_MAX = 1500.0
RUTILE_GC_N_T = 500
RUTILE_GC_P_MIN = 1.0e-10
RUTILE_GC_P_MAX = 1.0e10
RUTILE_GC_N_P = 500
# Rutile supplementary GC is fully novib (rutile, O₂ μ, and gas reservoir).
RUTILE_GC_GAS_USE_VIB = False
RUTILE_GC_GAS_INCLUDE_ROT_TRANS = True

# Master switch: IrO₂ / IrO₃ trace-gas reservoir in the **main-cluster** GC cache
# and cluster figure joins (figs 3–4, 7–12, supp_ir_size_*, supp_gc_g_vs_mu).
# Set to False and run ``python3 figures/build_all.py --rebuild-cache`` to omit them.
# The rutile supplementary GC (``supp_rutile_gc_phase_maps``) always keeps gas oxides.
GC_INCLUDE_GAS_RESERVOIR = False


def gc_gas_reservoir_specs(dataset: str) -> list[tuple[str, float]]:
    """IrO₂ / IrO₃ file + partial-pressure pairs when the reservoir is enabled."""
    if not GC_INCLUDE_GAS_RESERVOIR:
        return []
    return list(GC_GAS_OXIDES.get(dataset, ()))


def gc_rutile_gas_specs(dataset: str) -> list[tuple[str, float]]:
    """IrO₂ / IrO₃ paths for the rutile supplementary GC (always enabled)."""
    p = RUTILE_GC_GAS_P
    if dataset == "PBE":
        return [(IRO2_REF_PBE, p), (IRO3_REF_PBE, p)]
    return [(IRO2_REF_PBE0, p), (IRO3_REF_PBE0, p)]


REQUIRED_RAW_FILES = (
    TRAJ_PBE,
    TRAJ_PBE0,
    O2_REF_PBE,
    O2_REF_PBE0,
    IR_REF_PBE,
    IR_REF_PBE0,
)

# Grand-canonical / thermo figures: full PBE structures
GC_T_MIN = 100.0
GC_T_MAX = 1000.0
GC_T_TICKS = (200, 400, 600, 800, 1000)
GC_N_T = 150
GC_N_P = 150

# ---------------------------------------------------------------------------
# Derived cache  (produced by figures/utils/cache.py from data/ only)
# ---------------------------------------------------------------------------
CACHE_DIR = os.path.join(FIGURES_DIR, "cache")
OUTPUT_DIR = os.path.join(FIGURES_DIR, "outputs")

# The manuscript repo, next to this one. build_all.py also writes every figure the paper uses into
# its figures/{main,supplementary}/ (SVG + PNG), so recompiling the paper picks up the latest version.
# Override with AIST_PAPER_FIGURES=<dir>; skip with build_all.py --no-paper.
PAPER_FIGURES_DIR = os.environ.get("AIST_PAPER_FIGURES", os.path.join(os.path.dirname(ROOT), "aist_paper", "figures"))
PAPER_EXPORT_DPI = 150   # as aist_paper/convert_svgs.sh

ACE_ROOT = os.path.join(CACHE_DIR, "ace")
ACE_DIR = os.path.join(ACE_ROOT, "atoms")

ACE_FEATURES = {
    "PBE": os.path.join(ACE_DIR, "ace_features_PBE.csv"),
    "PBE0": os.path.join(ACE_DIR, "ace_features_PBE0.csv"),
}
ACE_CENTROIDS = {
    "PBE": os.path.join(ACE_DIR, "ace_centroids_PBE.traj"),
    "PBE0": os.path.join(ACE_DIR, "ace_centroids_PBE0.traj"),
    "pooled": os.path.join(ACE_DIR, "ace_centroids_pooled.traj"),
}
MOTIF_JSON = {
    "PBE": os.path.join(ACE_DIR, "motif_atoms_PBE.json"),
    "PBE0": os.path.join(ACE_DIR, "motif_atoms_PBE0.json"),
    "pooled": os.path.join(ACE_DIR, "motif_atoms_pooled.json"),
}

ACE_ATOM_IR_NPZ = os.path.join(ACE_ROOT, "ace_atom_ir.npz")
ACE_ATOM_O_NPZ = os.path.join(ACE_ROOT, "ace_atom_o.npz")
ACE_STRUCT_NPZ = os.path.join(ACE_ROOT, "ace_struct_embeddings.npz")

GC_NPZ = os.path.join(CACHE_DIR, "grandcanonical_results.npz")
GC_NPZ_PBE0 = os.path.join(CACHE_DIR, "grandcanonical_PBE0.npz")
GC_NPZ_BY_DATASET = {"PBE": GC_NPZ, "PBE0": GC_NPZ_PBE0}
GC_VARIANTS_NPZ = os.path.join(CACHE_DIR, "gc_variants.npz")
GC_VARIANTS_NPZ_PBE0 = os.path.join(CACHE_DIR, "gc_variants_PBE0.npz")
GC_VARIANTS_NPZ_BY_DATASET = {"PBE": GC_VARIANTS_NPZ, "PBE0": GC_VARIANTS_NPZ_PBE0}

# Rutile + gas reservoir supplementary GC (3-species: IrO₂ rutile, IrO₂(g), IrO₃(g))
RUTILE_GC_CACHE_DIR = os.path.join(CACHE_DIR, "rutile_gc")
RUTILE_GC_NPZ = {
    "PBE": os.path.join(RUTILE_GC_CACHE_DIR, "grandcanonical_rutile_PBE.npz"),
    "PBE0": os.path.join(RUTILE_GC_CACHE_DIR, "grandcanonical_rutile_PBE0.npz"),
}
RUTILE_GC_STRUCTURES = {
    "PBE": os.path.join(RUTILE_GC_CACHE_DIR, "structures_PBE.traj"),
    "PBE0": os.path.join(RUTILE_GC_CACHE_DIR, "structures_PBE0.traj"),
}

# Grand-canonical vib/ZPE on product and μ terms (per dataset NPZ cache).
# 2026-10-01: vibrations for both functionals (PBE0 from the re-relaxed dataset, best available spectrum).
# With vibrations on, structures without frequencies are left out of that dataset's ensembles
# (cache/gc_excluded_<label>.tsv lists them) instead of entering without vib/rot terms.
GC_USE_VIB_ZPE_BY_DATASET = {"PBE": True, "PBE0": True}

# Version of the cached results (figures/utils/cache_stamp.py). Bump it when a code change alters cached
# numbers: every cache whose stamp differs is rebuilt. 2 = 2026-10-02 thermo fixes (pV term, O2 rotation
# without vibrations, degeneracies stored).
CACHE_VERSION = 2
# Level of the PBE | PBE0 comparison views (figures 3–5 and their supplementary versions):
#   "full"  — both functionals with vibrations (grandcanonical NPZs)
#   "novib" — electronic + rot/trans only, the paper-1 comparison level (from the variants NPZs)
GC_COMPARISON_LEVEL = "full"

MINIMA_TRAJ = {
    "PBE": os.path.join(CACHE_DIR, "minima_PBE.traj"),
    "PBE0": os.path.join(CACHE_DIR, "minima_PBE0.traj"),
}


# ---------------------------------------------------------------------------
# Ambient O₂ partial pressure (~20% of 1 atm dry air)
ATMOSPHERIC_PRESSURE_PA = 101325.0
AMBIENT_PO2_FRACTION = 0.20
AMBIENT_PO2_PA = ATMOSPHERIC_PRESSURE_PA * AMBIENT_PO2_FRACTION

# The four (T, pO₂) conditions of the size-trend and IR-spectra figures (Andreu, 2026-10-05): 273 and 500 K at
# low and ambient pO₂; read at the nearest grid point. Colours: blue = 273 K, orange = 500 K, light = low pO₂.
STANDARD_CONDITIONS = (
    {"label": r"273 K, $10^{-5}$ Pa", "T": 273.0, "P": 1.0e-5, "color": "#6baed6"},
    {"label": r"273 K, ambient", "T": 273.0, "P": AMBIENT_PO2_PA, "color": "#08519c"},
    {"label": r"500 K, $10^{-5}$ Pa", "T": 500.0, "P": 1.0e-5, "color": "#fd8d3c"},
    {"label": r"500 K, ambient", "T": 500.0, "P": AMBIENT_PO2_PA, "color": "#a63603"},
)

# Fixed thermodynamic points (Figures 9 population lines)
FIXED_POINTS = [
    {"label": "273K_ambient_pO2", "T": 273.0, "P": AMBIENT_PO2_PA},
    {"label": "500K_ambient_pO2", "T": 500.0, "P": AMBIENT_PO2_PA},
]

# Per-Ir supplementary rows: Ir₃…Ir₉ cluster buckets (gas oxides appear in every row) — structure figures
SUPP_IR_COUNTS = list(range(3, 10))
# Thermodynamics (anything grand-canonical, i.e. needing vibrations) is shown only for the Ir sizes whose
# vibrations are (nearly) complete in both functionals (Andreu, 2026-10-01).
THERMO_IR_COUNTS = (4, 5, 6)
# Ir sizes that enter the grand-canonical ensembles (both functionals; GC, variants and Fig. 6 caches), so
# pooled maps and populations are Ir4–Ir6 ensembles too. The SOAP/ACE clustering keeps every size.
GC_IR_COUNTS = THERMO_IR_COUNTS

# Figure 4: motif populations vs T at fixed pO₂ (Pa)
FIG04_LOW_PRESSURE_PA = 1.0e-5
FIG04_HIGH_PRESSURE_PA = AMBIENT_PO2_PA
FIG04_SMOOTH_SIGMA = 1.0
FIG04_IR_COUNTS = THERMO_IR_COUNTS   # main Fig. 4 atom-motif rows (was Ir3 | Ir6 | Ir9)
# O motif indices: O-A0 bridging, O-A1 terminal, O-A2 molecular — fixed by aist.ace.clustering.
# reorder_o_atom_clusters_by_bonding, which names the SOAP–UMAP clusters by bonding and stops if they are not these three
FIG04_O_TERMINAL_IDX = 1
FIG04_O_BRIDGING_IDX = 0

# Figure 6 (sublimation): PBE0 GC with gas + support variants
FIG06_CACHE_DIR = os.path.join(CACHE_DIR, "fig06")
FIG06_GC_NPZ_PBE0 = os.path.join(FIG06_CACHE_DIR, "grandcanonical_PBE0_with_gas_base.npz")
FIG06_T_MIN = 100.0
FIG06_T_MAX = 1400.0
FIG06_N_T = 200
FIG06_N_P = 150
FIG06_PRESSURE_PA = AMBIENT_PO2_PA
FIG06_SUPPORT_VARIANTS = (
    {"label": "No support", "ev_per_ir": 0.0, "ev_per_o": 0.0},
    {"label": r"0.5 eV/Ir, 0.05 eV/O", "ev_per_ir": 0.5, "ev_per_o": 0.05},
    {"label": r"0.75 eV/Ir, 0.1 eV/O", "ev_per_ir": 0.75, "ev_per_o": 0.1},
)

# Figure 8: 1D ΔG-contribution cut at this pO₂ (Pa)
FIG08_PRESSURE_PA = AMBIENT_PO2_PA
# Stoichiometry panels: G_vib/n_Ir evaluated at this T (K); ZPE panels use T → 0 limit
FIG08_VIB_G_TEMP_K = 273.0

# ---------------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------------
IR_CMAP = "coolwarm"
O_CMAP = "plasma_r"
PHASE_CMAP = "coolwarm"
DATASET_COLORS = {"PBE": "#2166ac", "PBE0": "#b2182b"}
DPI = 800

# ACE clustering seed (must match ``aist.ace.constants``)
ACE_RANDOM_STATE = 42


def gc_grid_config(**overrides):
    """``GCConfig`` with the figure-pipeline T–p grid (default 150×150, 100–1000 K).

    Override per-term flags, e.g. ``include_v_product``, ``include_zpe_mu``, etc.
    """
    from aist.core.grandcanonical import GCConfig

    params = {
        "T_min": GC_T_MIN,
        "T_max": GC_T_MAX,
        "n_T": GC_N_T,
        "n_P": GC_N_P,
    }
    params.update(overrides)
    return GCConfig(**params)


def gc_rutile_grid_config(**overrides):
    """T–p grid for the rutile + gas supplementary GC cache."""
    from aist.core.grandcanonical import GCConfig

    params = {
        "T_min": RUTILE_GC_T_MIN,
        "T_max": RUTILE_GC_T_MAX,
        "n_T": RUTILE_GC_N_T,
        "P_min": RUTILE_GC_P_MIN,
        "P_max": RUTILE_GC_P_MAX,
        "n_P": RUTILE_GC_N_P,
    }
    params.update(overrides)
    return GCConfig(**params)


def gc_rutile_cache_config(**overrides):
    """Novib GC config for the 3-species rutile supplementary diagrams."""
    use_vib = overrides.pop("use_vib_zpe", False)
    thermo = {
        "include_v_product": use_vib,
        "include_zpe_product": use_vib,
        "include_v_mu": use_vib,
        "include_zpe_mu": use_vib,
    }
    thermo.update(overrides)
    return gc_rutile_grid_config(include_decomposition=False, **thermo)


def gc_fig06_grid_config(**overrides):
    """T–p grid for the Fig. 6 sublimation GC cache (T up to 1400 K)."""
    from aist.core.grandcanonical import GCConfig

    params = {
        "T_min": FIG06_T_MIN,
        "T_max": FIG06_T_MAX,
        "n_T": FIG06_N_T,
        "n_P": FIG06_N_P,
    }
    params.update(overrides)
    return GCConfig(**params)


def gc_fig06_cache_config(**overrides):
    """GC config for the Fig. 6 sublimation cache (T up to 1400 K)."""
    use_vib = overrides.pop("use_vib_zpe", False)
    thermo = {
        "include_v_product": use_vib,
        "include_zpe_product": use_vib,
        "include_v_mu": use_vib,
        "include_zpe_mu": use_vib,
    }
    thermo.update(overrides)
    return gc_fig06_grid_config(include_decomposition=True, **thermo)


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
