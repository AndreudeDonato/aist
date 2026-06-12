# Paper Figures Plan (Working Notes)

This file collects the figure definitions as provided by the user, in order.

## Development Rules (Implementation Guardrails)

- **Single input root:** only `data/` is read as raw input. Everything else is built under `figures/cache/` via `figures/utils/cache.py`.
- **Allowed compute modules:** `aist/ace/` (clustering), `aist/core/` (grand-canonical, variants, vibrational, minima). `figures/` must not reimplement scientific logic.
- **Backward-compatible wrappers:** `analysis/aceclustering.py` and `scripts/compute_grandcanonical.py` delegate to `aist.*`.
- **`figures/` scope:** two explicit stages — **data production** then **plotting**.
- **Stage 1 — data production** (`figures/utils/cache.py`): reads only `data/`, writes numeric artifacts under `figures/cache/` (NPZ, ACE CSV, motif JSON, trajectories). No paper figure SVGs in this step.
- **Stage 2 — plotting** (`figures/figXX_*.py`): reads only `figures/cache/` (+ layout config), writes SVGs to `figures/outputs/`. Do not embed or load intermediate ACE diagnostic SVGs.
- **One script per figure** for stage 2; stage 1 is shared via the cache builder.
- **Master runner required:** keep a `figures/build_all.py` entry point to regenerate all figures, while preserving per-figure execution.
- **Shared helpers allowed only for glue code:** I/O paths, joins, validation, common styling, and vibrational metrics helpers (e.g. `figures/utils/vib_metrics.py`), not duplicated core physics/clustering.
- **Dataset policy:** when PBE0 vib/frequency data are missing, use PBE placeholders where defined in this plan; once complete, generate both PBE and PBE0 and route PBE to supplementary where requested.
- **Traceability:** cache artifacts live in `figures/cache/`; final SVGs in `figures/outputs/`. Rebuild with `python3 figures/build_all.py --rebuild-cache`.

## Architecture (post-refactor)

```text
data/  --[cache.py]-->  figures/cache/   (NPZ, CSV, JSON, trajs)
figures/cache/  --[figXX.py]-->  figures/outputs/*.svg
```

ACE cache artifacts (under `figures/cache/ace/`):
- `atoms/ace_features_{PBE,PBE0}.csv`
- `ace_atom_ir.npz`, `ace_atom_o.npz`, `ace_struct_embeddings.npz`
- `atoms/motif_atoms_*.json`, centroid/group trajs

GC cache: `grandcanonical_results.npz`, `gc_variants.npz` (with `full_Pgi`, `novib_Pgi`).

**Fast GC iteration:** in `figures/config.py`, `GC_N_T` and `GC_N_P` default to **100** (coarse
T–p grid) while structures remain full `IryOxPBE.traj`. Increase for production-quality maps.

| Module | Responsibility |
|--------|----------------|
| `aist/ace/` | SOAP descriptors, UMAP/KMeans, join helpers, population kernels |
| `aist/core/grandcanonical.py` | Grand-canonical `compute_gc` / `run_gc` |
| `aist/core/gc_variants.py` | Full vs novib variants (`full_Pgi`, `novib_Pgi`) |
| `aist/core/vibrational.py` | ZPE, Tchar, H/S thermo helpers |
| `aist/core/minima.py` | Stoichiometry minima selection |
| `figures/` | Cache orchestration, matplotlib layout, SVG export |

### Migration

- `from analysis.aceclustering...` → `from aist.ace...` (old paths still re-export).
- `scripts/compute_grandcanonical.py` → delegates to `aist.core.grandcanonical`.
- `analysis/aceclustering.py` → delegates to `aist.ace.pipeline`.

### Verification

```bash
python3 figures/utils/cache.py --force
python3 figures/build_all.py
```

GC row count must match `IryOxPBE.traj` and ACE CSV length; misalignment raises before plotting.

## Figure 1

- **Script:** `figures/fig01_minima.py` (uses `aist.core.minima` + same tile styling as `analysis/plotminima.py`)
- **Layout:** 180 × 230 mm combined; PBE and PBE0 panels each 88 × 230 mm with 4 mm blank gap; no text; shared `half_span` across both datasets (same zoom / Å scale)
- **Outputs:** `figures/outputs/fig01_minima.svg`, plus per-dataset `fig01_minima_PBE.svg` / `fig01_minima_PBE0.svg`
- **Status:** implemented

## Figure 2

- **Layout:** **2 × 3**, 180 × 120 mm (60 mm per row) — scatter | PBE | PBE0; row 1 Ir, row 2 O
- **Script:** `figures/fig02_atom_motifs.py`
- **Status:** implemented

## Figure 2–3 atoms (supplementary)

- **Script:** `figures/fig023_motif_minima_tiles.py`
- **Data:** `figures/cache/ace/atoms/ace_centroids_pooled.traj` only (one structure per pooled cluster C0…)
- **Motifs:** `motif_atoms_pooled.json` (global `frame_index` from centroid `atoms.info`)
- **Layout:** 88 × 200 mm — 7 rows × 2 columns (10 mm labels | 78 mm structures)
  - Left: coloured square + ``C*``; right: inertia-oriented centroid with Ir/O motif colours
- **Output:** `figures/outputs/fig023_motif_minima_tiles.svg`
- **Status:** implemented; placement in paper TBD

## Figure 3

- **Layout:** **1 × 3**, 180 × 60 mm (row height matches Figure 2): PCA scatter | PBE | PBE0
- **Scatter:** legend ``C0``, …; title ``structures - PCA`` (paper style, like Figure 2)
- **Script:** `figures/fig03_structure_motifs.py`
- **Status:** implemented

## Figure 4

- **Core scripts:** `scripts/compute_grandcanonical.py` and `scripts/plot_grandcanonical.py`
- **Thermo:** ``GC_USE_VIB_ZPE_BY_DATASET`` in ``figures/config.py`` — **PBE** GC includes vib/ZPE (frequencies in traj); **PBE0** GC stays electronic-only until ``IryOxPBE0.traj`` has frequencies. Rebuild with ``--rebuild-cache`` after changing flags.
- **Expected integration work:** modify these scripts and their outputs (NPZ/JSON or datasets) so clustering/motif metadata is available for plotting, potentially by carrying additional info in `atoms.info`
- **Layout:** **4 × 2** multipanel
  - left column: PBE (``grandcanonical_results.npz`` + ``ace_features_PBE.csv``)
  - right column: PBE0 (``grandcanonical_PBE0.npz`` + ``ace_features_PBE0.csv``)
  - same PBE/PBE0 convention as Figure 1
- **Row 1 (already implemented):**
  - standard phase diagram
  - colour by oxygen-fraction population
  - overlay/phase layers by stoichiometry
- **Row 2:** ensemble-averaged structure-cluster index ⟨C⟩ (``Pgi``-weighted, like ⟨n_O/n_Ir⟩)
- **Row 3:** ensemble-averaged Ir-atom motif index ⟨Ir-A⟩ from per-structure motif fractions
- **Row 4:** ensemble-averaged O-atom motif index ⟨O-A⟩ from per-structure motif fractions
- **Rows 2–4 colour:** continuous gradient through cluster/motif palette on the map; **discrete** colorbars (`C0…`, `Ir-A0…`, `O-A0…`) with solid blocks per category
- **Status:** implemented in `figures/fig04_gc_clustering_maps.py`

## Figure 5

- **Target dataset:** **PBE0 only** (paper intent)
- **Interim dataset:** use **PBE** GC data for now, because PBE0 still lacks complete vibrational-frequency data for a full vib/ZPE run
- **Core scripts:** `scripts/compute_grandcanonical.py` and `scripts/plot_grandcanonical.py`
- **Layout:** **2 × 2** multipanel of phase diagrams (all from the grand-canonical pipeline)

| Panel | Position | Phase diagram content |
|--------|----------|------------------------|
| **(1,1)** top-left | Full model | Standard phase diagram with **everything**: ensemble (`Pgi`), **vibrational + ZPE** contributions |
| **(1,2)** top-right | No vib/ZPE | Phase diagram **without vibrational nor ZPE** contributions |
| **(2,1)** bottom-left | No ensemble | **No ensemble character** (e.g. global minimum / single-isomer map instead of `Pgi`-weighted fields), but **with vib + ZPE** |
| **(2,2)** bottom-right | Minimal | **Without vib, without ZPE, and without ensemble character** |

- **Purpose:** isolate how much the diagram changes from (i) vibrations/ZPE and (ii) ensemble averaging vs a single-minimum picture
- **Status:** recorded; may require new GC compute modes or NPZ keys (`Pgi` vs `phase_index`, `include_v`/`include_zpe` flags) and matching panels in `plot_grandcanonical.py`

## Figure 6

- **Layout:** **2 × 2** multipanel (vibrational thermodynamics section)
- **Dataset:** **PBE** (`grandcanonical_results.npz`, ``IryOxPBE.traj`` frequencies). PBE0 vib data unfinished — not plotted.
- **Core sources:** `scripts/plot_grandcanonical.py`, `scripts/compute_grandcanonical.py`, `analysis/zpe_by_ir_content.py`, plus new per-structure **T_char** precompute

| Panel | Position | Content |
|--------|----------|---------|
| **(1,1)** | G-contribution 1D cut | All ensemble-averaged ΔG contributions vs **T** at fixed **pO₂** (top-left of 2×2) |
| **(1,2)** | Vibrational Gibbs map | `E_vib_avg` on **T × pO₂** with phase boundaries |
| **(2,1)** | ZPE vs composition | `zpe_by_ir_content` |
| **(2,2)** | T_char phase diagram | Ensemble ⟨T_char⟩ map + **T = T_char** contour |

- **T_char implementation notes:**
  - Compute `T_char,i` once per structure from `atoms.info['frequencies']` (same harmonic/Einstein pipeline as `aist/thermo.py`)
  - Phase-map field and contour use the same `Ts`, `Ps`, `Pgi` grid as other GC figures
  - **Do not use “average structure” tiles** for T_char-linked views: assign each **atom** the **T_char of its parent structure**, and colour atoms by their **motif group** (cluster / Ir-A* / O-A* as appropriate). Same rule applies to Figure 7.
- **Status:** recorded; panels (1,1) and (1,2) largely exist inside `plot_grandcanonical.py` (`G_contributions_3x2.svg`); (2,1) exists as standalone script; (2,2) requires new precompute + plotting

## Figure 7

- **Layout:** **1 × 2** — PBE | PBE0 Hirshfeld scatter (all atoms in each traj)
- **Axes:** **x** = Hirshfeld charge · **y** = |Hirshfeld spin| · **colour** = element (Ir / O)
- **Script:** `figures/fig07_tchar_motif_hs.py` → `figures/outputs/fig07_tchar_motif_hs.svg`
- **Status:** implemented (interim comparison of PBE vs PBE0 electronic structure in `atoms.info['hirshfeld']`)

## Figure 8

- **Dataset:** **PBE** — ``grandcanonical_results.npz`` (panel 1) + ``gc_variants.npz`` (novib / full ``Pgi`` at fixed points)
- **Layout:** **1 × 3** (final figure in the current list)
- **Panel 1 (already available):**
  - phase diagram coloured by **population of the lowest-energy structure** (global minimum population map)
- **Panels 2–3:** at **`pO2 = 101325 Pa`**, **`T = 273 K`** and **`T = 500 K`**
  - include all structures accounting for **97%** of total novib population at that point
  - **x-axis:** structures ranked by **novib** population (high → low)
  - **y-axis:** population (%) — line for **no vib**, line for **with vib** (same structure order as novib)
- **Purpose:** quantify how much vibrational terms change population distribution for named low-energy clusters at representative conditions
- **Status:** recorded; requires fixed-point extraction utility from GC outputs (`Pgi` for vib and no-vib runs) plus shared label mapping from phase-diagram clusters to paper names

