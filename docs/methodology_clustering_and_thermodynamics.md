# Methodology: clustering, ab initio thermodynamics, and their combination

This document describes the computational methodology implemented in the `aist` package and used by the paper figure pipeline (`figures/`). It is written for downstream use when drafting the **Methods** section of the paper.

**Implementation roots**

| Topic | Code |
|-------|------|
| Clustering | `aist/ace/` (`pipeline.py`, `clustering.py`, `descriptors.py`, `join.py`), `aist/inertia.py` |
| Thermodynamics | `aist/core/grandcanonical.py`, `aist/composedthermo.py`, `aist/thermo.py` |
| Vibrational metrics | `aist/core/vibrational.py` |
| Cache orchestration | `figures/utils/cache.py` |
| Configuration | `figures/config.py` |

**Raw inputs** (only source data): `data/IryOxPBE.traj`, `data/IryOxPBE0.traj`, `data/O2_PBE.xyz`, `data/O2_PBE0.xyz`, `data/IrPBE.xyz`, `data/IrPBE0.xyz`.

**Derived cache**: `figures/cache/` (ACE CSV/NPZ/JSON, grand-canonical NPZ, GC variants).

For support-stabilization proxy, IrO₂/IrO₃ gas reservoir, and literature framing (Reichenbach / Kempen), see [gc_support_gas_reservoir.md](gc_support_gas_reservoir.md).

---

## 1. Datasets and scope

Two DFT functionals are treated in parallel:

- **PBE** — full `IryOxPBE.traj` (~1065 Ir/O nanoparticle structures) with harmonic frequencies in `atoms.info['frequencies']` and ZPE in `atoms.info['zpe']` where available.
- **PBE0** — `IryOxPBE0.traj` with the same stoichiometry grid but **without complete vibrational data** at present; grand-canonical runs for PBE0 use **electronic + translational + rotational** terms only (no vib/ZPE on cluster or O₂ reference).

Clustering **pools PBE and PBE0** for atom-level and structure-level unsupervised classification so that motif labels are comparable across functionals. Thermodynamics are computed **per functional** with the matching O₂ reference (`O2_PBE.xyz` / `O2_PBE0.xyz`) and Ir reference (`IrPBE.xyz` / `IrPBE0.xyz`).

---

## 2. Atom and structure clustering (ACE-style pipeline)

### 2.1 Motivation

Structures in the grand-canonical ensemble differ in geometry, oxidation, and local coordination. To summarise chemical diversity beyond stoichiometry, we classify:

1. **Ir atoms** and **O atoms** into local **atom motifs** (Ir-A*, O-A*).
2. **Whole structures** into **structure clusters** (C*) using the atom-motif composition of each structure.

The pipeline follows an “ACE-style” workflow: per-atom environment descriptors → dimensionality reduction → clustering, then a second clustering stage on structure-level composition vectors.

### 2.2 Per-atom descriptors (SOAP)

For every structure, each atom receives a **SOAP** (Smooth Overlap of Atomic Positions) vector from `dscribe`:

| Parameter | Value |
|-----------|-------|
| Species | Ir, O |
| Cutoff `r_cut` | 5.0 Å |
| Radial basis size `n_max` | 8 |
| Angular basis size `l_max` | 6 |
| Gaussian width `sigma` | 0.5 Å |

SOAP encodes the local radial and angular distribution of neighbouring atoms around each centre; it is a standard geometry-sensitive fingerprint suitable for unsupervised grouping of chemically similar sites.

### 2.3 Pooling across functionals

All Ir atoms from PBE and PBE0 are pooled into one dataset; all O atoms are pooled separately. Provenance `(structure_index, atom_index)` is tracked so labels can be mapped back to each structure in each trajectory.

Structure-level stoichiometry arrays `n_Ir`, `n_O` are pooled in the same order for later oxidation-based reordering.

### 2.4 Atom-motif clustering (Ir and O separately)

For each species block (Ir or O):

1. **StandardScaler** on SOAP vectors (enabled by default).
2. **PCA** retaining **95%** of cumulative variance (`pca_variance_target = 0.95`).
3. **UMAP** to 2D (`n_neighbors`, `min_dist = 0.0`, `metric = euclidean`, `init = pca`, `random_state = 42`).
4. **KMeans** on the UMAP embedding; cluster count chosen by **maximum silhouette score** on a random subsample of 5000 points (full silhouette is O(n²)), unless a fixed **k** is set.

After KMeans, clusters are **relabelled by size** (largest → index 0).

**Production hyperparameters** (`aist/ace/pipeline.py`):

| Species | UMAP `n_neighbors` | Motif count **k** |
|---------|-------------------|-------------------|
| Ir | 60 (wider neighbourhood to resolve oxidised regimes) | **4** fixed (Ir-A0…Ir-A3) |
| O | 30 | **3** fixed (O-A0…O-A2) |

**Ir atom motifs** are then **reordered by oxidation**: each Ir atom inherits the `n_O / n_Ir` ratio of its parent structure; clusters are renumbered so **Ir-A0** has the lowest mean oxidation and **Ir-A{k−1}** the highest. This makes colour maps interpretable along the oxidation axis (coolwarm).

**O atom motifs** keep size-based ordering (no oxidation reorder pass).

Labels are written back to per-structure per-atom lists and stored in `motif_atoms_*.json` for visualisation.

### 2.5 Structure-level clustering

Structure clustering uses a **concatenated descriptor** per structure (PBE and PBE0 **pooled**), then **StandardScaler → PCA (95% variance) → KMeans** with silhouette **k ∈ {2,…,8}** unless a fixed **k** is set. Structure clusters are **reordered by mean `n_O / n_Ir`** so **C0** is least oxidised and **C{k−1}** most oxidised.

**Production descriptor** (dimension = k_Ir + k_O + 2 + 1 = **10** with k_Ir = 4, k_O = 3):

| Block | Columns | Role |
|-------|---------|------|
| Ir motif fractions | 4 | `Ir-A0_frac` … `Ir-A3_frac` |
| O motif fractions | 3 | `O-A0_frac` … `O-A2_frac` |
| Extent-weighted shape | 2 | `(r_{13}, r_{23})` scaled per structure (see below) |
| Molecular O₂ flag | 1 | binary: structure contains an O–O pair ≤ 1.6 Å |

**Fixed structure cluster count:** **k = 9** (`struct_k = 9`).

**Molecular O₂ emphasis:** the O₂ column is multiplied by **1.4** after standardisation (`o2_feature_weight`) so all O₂-bearing structures tend to coalesce into one interpretable cluster without letting that single flag dominate the first principal component.

The same cluster label is assigned to both PBE and PBE0 structures in the pooled index; per-dataset CSV files (`ace_features_PBE.csv`, `ace_features_PBE0.csv`) store `frame_index`, stoichiometry, cluster id, and motif fractions.

#### Principal moments: definition and ordering

For each structure, the **mass-weighted inertia tensor** is built from atomic positions and masses (`ThermoPartitionFunctions.get_inertia_tensor` in `aist/thermo.py`). Its eigenvalues are the **principal moments of inertia**; we sort them ascending:

\[
\lambda_1 \le \lambda_2 \le \lambda_3
\]

(\(\lambda_1\) smallest, \(\lambda_3\) largest). This ordering is applied automatically via `numpy.linalg.eigvalsh`. They are **not** tied to Cartesian \(x,y,z\) labels until a structure is oriented for display; in moment-ratio plots the horizontal axis is **\(I_x/I_z\)** and the vertical **\(I_y/I_z\)**, meaning **ratio of smallest to largest** and **middle to largest** principal moments:

\[
r_{13} = \frac{\lambda_1}{\lambda_3},\qquad r_{23} = \frac{\lambda_2}{\lambda_3}.
\]

**Shape limits in the moment-ratio plane:**

| Corner | \((r_{13}, r_{23})\) | Geometry |
|--------|----------------------|----------|
| **S** (spherical) | \((1, 1)\) | \(\lambda_1 = \lambda_2 = \lambda_3\) |
| **L** (linear) | \((0, 1)\) | mass along one axis |
| **P** (planar) | \((\tfrac{1}{2}, \tfrac{1}{2})\) | coplanar mass distribution |

#### Extent: mass-normalised spread

**Extent** measures how spread-out the mass distribution is, relative to cluster size and morphology. It is **not** a third shape ratio and is **not** appended as its own clustering column; instead it **gates** how strongly shape enters structure clustering (see below).

Define total trace moment \(I_{\mathrm{tot}} = \lambda_1 + \lambda_2 + \lambda_3\) and total mass \(M\). The **log-extent** is

\[
\log_{\mathrm{extent}}
= \tfrac{1}{2}\log I_{\mathrm{tot}} - c_{\mathrm{eff}}\,\log M,
\]

where \(c_{\mathrm{eff}}\) depends on the chosen mass normalisation (see table below). With radius of gyration \(R_g = \sqrt{I_{\mathrm{tot}}/M}\), this is equivalent to \(\log R_g - (c_{\mathrm{eff}} - \tfrac{1}{2})\log M\).

**Why shape-dependent mass correction?** For ideal objects of effective spatial dimension \(d\), \(I_{\mathrm{tot}} \sim M^{1+2/d}\). A fixed sphere-only exponent removes the atom-count trend only for compact 3D objects:

| Geometry | \(d\) | \(I_{\mathrm{tot}}\sim\) | Residual \(\sim \tfrac{1}{2}\log M\) if \(c_{\mathrm{eff}} = 5/6\) |
|----------|-------|---------------------------|---------------------------------------------------------------------|
| Chain (L) | 1 | \(M^3\) | \(+\tfrac{2}{3}\log M\) |
| Disk (P) | 2 | \(M^2\) | \(+\tfrac{1}{6}\log M\) |
| Sphere (S) | 3 | \(M^{5/3}\) | 0 |

Using only the sphere formula (`extent_normalization = "sphere"`, \(c_{\mathrm{eff}} = 5/6\)) therefore **over-weights linear structures** in any step that multiplies shape by extent: elongated oxides accumulate higher extent with \(N\) than compact or planar ones of the same physical spread.

**Effective dimension** — interpolating across shape corners S \((1,1)\to d=3\), P \((\tfrac{1}{2},\tfrac{1}{2})\to d=2\), L \((0,1)\to d=1\) via \(r_{13}\) gives

\[
d_{\mathrm{eff}} = 1 + 2\,r_{13},
\]

and the **full shape-corrected** exponent is \(c_{\mathrm{shape}} = \tfrac{1}{2} + 1/d_{\mathrm{eff}}\) (`extent_normalization = "shape"`).

**Production setting: partial blend** (`extent_normalization = "partial"`, `extent_blend = 0.1`):

\[
c_{\mathrm{eff}}
= \tfrac{5}{6} + b\left(\tfrac{1}{2} + \frac{1}{d_{\mathrm{eff}}} - \tfrac{5}{6}\right),
\quad b = 0.1.
\]

At \(b = 0.1\) the correction is **mostly sphere-like** but retains a small morphology-dependent term. This keeps **within-shape extent variation** (how open vs compact a given motif family is) while avoiding the over-correction of full shape normalisation, where spherical clusters can appear artificially more “extended” than linear ones of the same size. The same `extent_blend = 0.1` is used for figure maps (`figures/fig03`, `figures/supp_moments_of_inertia`, imported from `aist/ace/pipeline.py`).

For maps and moment-ratio supplementary figures, batch-normalised extent is mapped to **lightness** on an RGB simplex (R = linear **L**, G = planar **P**, B = spherical **S**); compact structures appear pale, extended ones saturated.

#### Extent-weighted shape in clustering

Structure clustering does **not** use raw \((r_{13}, r_{23})\). Instead `inertia_shape_weighted_by_extent` (`aist/inertia.py`) builds two columns per structure:

1. Compute \(\log_{\mathrm{extent}}\) for every structure in the pooled batch.
2. **Batch-normalise** to \([0,1]\): \(\mathrm{extent\_norm} = (\log_{\mathrm{extent}} - \min)/(\max - \min)\).
3. Optional **corner-purity boost** (`shape_corner_purity_beta = 0.5`): weight \(w_i = \mathrm{extent\_norm}_i \cdot \alpha \cdot (1 + \beta \cdot \mathrm{purity}_i^2)\), where \(\mathrm{purity}\) is `moment_ratio_corner_purity` — high only near S, L, or P shape corners, not in the ambiguous interior.
4. Clustering features: \((r_{13,i}\, w_i,\; r_{23,i}\, w_i)\) with \(\alpha = 1.0\) (`shape_extent_alpha`).

**Effect on oxide diversity:**

- **Compact, near-spherical nanoparticles** (low \(\mathrm{extent\_norm}\)): \(w_i \approx 0\) → the two shape columns are **near zero** → clustering is driven by **atom-motif composition** and the O₂ flag. Many stoichiometries with similar local coordination but different global size therefore stay together if they share motif fractions.
- **Extended, fragmented, or elongated oxides** (high \(\mathrm{extent\_norm}\)): \(w_i \to 1\) (or higher near pure S/L/P corners) → **full shape ratios enter** → structures with similar motif vectors but different global mass distributions (e.g. chain-like IrOₙ vs compact IrOₙ islands, or partially dissociated O₂ layouts) **split into distinct clusters**.
- This is the main mechanism by which the pipeline classifies **chemically similar but geometrically diverse** oxides beyond what motif fractions alone can separate: extent acts as a **confidence gate** on shape, not as an independent clustering axis.

Extent is always batch-normalised before weighting, so the absolute scale of \(\log_{\mathrm{extent}}\) does not directly shift cluster boundaries—only **relative** spread within the dataset matters for clustering weights.

#### Achievable moment-ratio region

Sorted moments satisfy \(r_{13} \le r_{23} \le 1\). A second constraint from the inertia tensor (\(\lambda_1 + \lambda_2 \ge \lambda_3\) for physical mass distributions) gives

\[
r_{23} \ge \max\!\left(r_{13},\,\tfrac{1}{2}\right).
\]

The wedge below \(r_{23} = \tfrac{1}{2}\) is **mathematically empty** for any physical inertia tensor—not merely unpopulated in our dataset. Supplementary Figure `supp_momentsofinertia` plots \((r_{13}, r_{23}, \log_{\mathrm{extent}})\) in 3D with **partial** extent normalisation (\(b = 0.1\)) on the **E** axis and motif-composition colouring on atom panels; Figure 3 includes an inertia RGB row on the GC maps.

### 2.6 Chemical interpretation of production labels

The assignments below summarise the **current** pooled ACE cache (`figures/cache/ace/`, \(n = 1782\) structures: 1065 PBE + 717 PBE0). They are **not** hard-coded names in the pipeline—only integer labels ordered by oxidation (Ir motifs, structure clusters) or size (O motifs before geometry check). Re-run `python3 analysis/Iratoms/check_o_motif_identity.py` and `python3 analysis/Iratoms/inspect_struct_clusters.py` after any ACE rebuild to refresh the statistics.

#### O atom motifs (O-A0 … O-A2)

Geometry from Ir–O coordination (cutoff 2.4 Å) and O–O pairs (molecular cutoff 1.6 Å, same as the structure-level O₂ flag):

| Label | Chemical role | Geometry (pooled atoms) |
|-------|----------------|-------------------------|
| **O-A0** | **Bridging O** | \(\langle n_{\mathrm{Ir\,neigh}} \rangle \approx 2.1\); ~92% of atoms have two Ir neighbours within 2.4 Å |
| **O-A1** | **Terminal O** | \(\langle n_{\mathrm{Ir\,neigh}} \rangle = 1.0\); 100% one Ir neighbour |
| **O-A2** | **Molecular O₂** | 100% of atoms belong to an O–O pair \< 1.6 Å (dissociative/adsorbed O₂ layouts) |

In figures, **O-A0 = bridging** and **O-A1 = terminal** (`FIG04_O_BRIDGING_IDX = 0`, `FIG04_O_TERMINAL_IDX = 1` in `figures/config.py`).

#### Ir atom motifs (Ir-A0 … Ir-A3)

Ir motifs are **ordered by parent-structure oxidation** after clustering: **Ir-A0** is the SOAP motif family with the lowest mean \(n_O/n_{\mathrm{Ir}}\), **Ir-A3** the highest. They capture distinct Ir local environments (coordination and oxide character) that track the stoichiometry ladder rather than a single scalar oxidation state:

- **Ir-A0** — metal-like / weakly oxidised Ir (dominant in C0–C1).
- **Ir-A1** — early oxide Ir sites (dominant in C2; mixed in C3).
- **Ir-A2** — mid–high oxidation Ir environments (dominant in C4–C5).
- **Ir-A3** — strongly oxidised Ir (dominant in C6–C8; essentially all Ir in C7).

#### Structure clusters (C0 … C8)

Clusters are ordered by **mean** \(n_O/n_{\mathrm{Ir}}\) (C0 lowest, C8 highest). Several groups are chemically sharp; others overlap in stoichiometry and are separated by **motif composition**, the **O₂ binary flag**, or **inertia shape** (especially linear vs compact layouts at high oxidation).

| Cluster | \(n\) | Mean \(n_O/n_{\mathrm{Ir}}\) | Dominant chemistry (motifs + shape) | Narrative |
|---------|------|-------------------------------|-------------------------------------|-----------|
| **C0** | 220 | 0.27 | Terminal O (52%), Ir-A0; compact planar-ish \((r_{13}, r_{23}) \approx (0.53, 0.83)\) | Low-oxidation Ir-rich cores with **terminal** O termination; mostly PBE |
| **C1** | 230 | 0.42 | **Bridging O (97%)**, Ir-A0; similar compact shape | Early oxide with **bridge-rich** O layer on still metal-like Ir; lowest oxides that are almost exclusively bridging |
| **C2** | 163 | 1.12 | Terminal O (84%), Ir-A1 | Mid-oxidation (\(\sim\)1:1 O/Ir) **terminal-oxide** family; strongly PBE-weighted |
| **C3** | 138 | 1.18 | Bridging O (74%) + terminal, Ir-A1/Ir-A2 | Same stoichiometry band as C2 but **bridge-heavy** and more oxidised Ir motifs |
| **C4** | 306 | 1.36 | Bridging O (91%), Ir-A2 | Sub-stoichiometric to near-stoichiometric **bridging oxide** nanoparticles; largest mid-ox group |
| **C5** | 216 | 1.87 | Bridging O (79%), Ir-A2/Ir-A3 | Higher oxidation **bridging** layouts; somewhat more extended mass distribution than C4 |
| **C6** | 99 | 2.54 | **O-A2 (molecular O₂) 34%**, mixed bridging/terminal, Ir-A2/Ir-A3 | **All O₂-containing structures** in the dataset (O–O \< 1.6 Å); dissociative/adsorbed O₂ chemistry isolated by the binary O₂ feature |
| **C7** | 224 | 2.65 | Mixed bridging/terminal (58/42%), **Ir-A3 only** | High oxidation (\(n_O/n_{\mathrm{Ir}} \ge 2\)) **compact** IrOₓ with fully oxidised Ir motifs |
| **C8** | 186 | 2.74 | 50/50 bridging/terminal, Ir-A3; **linear** \((r_{13}, r_{23}) \approx (0.25, 0.85)\) | Highest mean oxidation; **elongated / chain-like** IrOₓ (linear **L** shape corner) distinct from compact high-ox C7 despite similar stoichiometry |

**PBE vs PBE0:** labels are pooled, so both functionals share C*. C2 is mostly PBE; C5, C6, and C8 contain more PBE0 structures—consistent with PBE0 favouring higher-oxidation / molecular-O₂ configurations in this stoichiometry grid. Functional differences in phase maps therefore reflect both electronic structure and slightly different population weights across these groups.

**Role of extent in splitting “similar” oxides:** C7 vs C8 is the clearest example at high oxidation: similar \(n_O/n_{\mathrm{Ir}}\) and Ir-A3-dominated motifs, but C8’s **low** \(r_{13}\) (linear mass distribution) enters clustering because extent is high enough to unlock shape features—separating **compact high-oxide particles** from **linear oxide chains/fragments**. At low oxidation, C0 vs C1 splits largely on **terminal vs bridging O** motif fractions with shape still near-planar and extent low (shape columns small).

### 2.7 Reproducibility

- Global seed: `ACE_RANDOM_STATE = 42` (`aist/ace/constants.py`).
- UMAP and KMeans use `random_state = 42`; UMAP `n_jobs = 1`.
- `set_reproducible_seed()` is called at pipeline start and in `figures/build_all.py`.

### 2.8 Clustering outputs (cache)

Under `figures/cache/ace/`:

| Artifact | Content |
|----------|---------|
| `atoms/ace_features_{PBE,PBE0}.csv` | Per-structure cluster + motif fractions |
| `atoms/motif_atoms_{PBE,PBE0,pooled}.json` | Per-atom motif colours and indices |
| `atoms/ace_centroids_*.traj`, `ace_groups_*/` | Representative structures per cluster |
| `ace_atom_ir.npz`, `ace_atom_o.npz` | UMAP embeddings and labels for atoms |
| `ace_struct_embeddings.npz` | PCA coordinates and labels for structures |

Diagnostic SVG plots under `ace/plots/` are optional (`write_plots=False` in the figure cache builder).

---

## 3. Ab initio grand-canonical thermodynamics

### 3.1 Chemical reaction and references

Grand-canonical assembly follows a balanced reaction between:

- **Reference Ir** (single atom from `IrPBE.xyz` or `IrPBE0.xyz`)
- **O₂** (from `O2_PBE.xyz` or `O2_PBE0.xyz`)
- **Cluster product** (each structure in the trajectory)

Stoichiometric coefficients `(r_coeffs, p_coeffs)` are obtained by solving the element balance with `aist/stoichiometry.balance_reaction` (sympy null-space; first reactant normalised to 1).

For each cluster structure *i*, the **Gibbs energy surface** on the `(T, p_O₂)` grid is:

\[
G_i(T,p) = p_i \, G_{\mathrm{cluster},i}(T, P_{\mathrm{dilute}}) + r_{i,0}\, G_{\mathrm{Ir}}(T, P_{\mathrm{beam}}) + r_{i,1}\, \mu_{\mathrm{O_2}}(T,p)
\]

- `P_dilute = 10⁻² Pa` for cluster translational partition (dilute limit).
- `P_beam = 10⁻² Pa` for the Ir reference beam.
- `μ_O₂(T,p)` is the O₂ chemical potential at the **actual** grid pressure `p`.

### 3.2 Thermodynamic contributions

Partition functions use `ThermoPartitionFunctions` / `ThermoFromAtoms` (`aist/thermo.py`). For each species, `compute_thermo_contributions` (`aist/composedthermo.py`) returns **G = E + ZPE + E_trans + E_vib + E_rot + E_elec − T·S** and per-term energy/entropy contributions.

| Term | Cluster product | O₂ (μ) | Ir reference |
|------|-----------------|--------|--------------|
| Electronic | Yes (from `total_spin` → J) | Yes (triplet O₂, J=3) | No |
| Translational | No (dilute P on cluster) | Yes | No |
| Rotational | Optional (off for cluster by default) | Yes | No |
| Vibrational + ZPE | Configurable | Configurable | No |

**Vibrational modes**: harmonic frequencies from `atoms.info['frequencies']` (cm⁻¹); **5** (linear) or **6** (non-linear) lowest modes removed as translations/rotations (`process_frequencies`). ZPE from `atoms.info['zpe']` when `include_zpe` is true.

**Configurable flags** (`GCConfig` in `aist/core/grandcanonical.py`):

- `include_v_product`, `include_zpe_product`, `include_v_mu`, `include_zpe_mu` — vibrational and zero-point terms on cluster and O₂.
- `include_e_product`, `include_e_mu` — electronic/spin contributions.
- `include_r_mu`, `include_t_mu` — rotational and translational on O₂ reference.
- `include_decomposition` — store ensemble-averaged contribution maps (`E_dft_avg`, `E_elec_avg`, `E_vib_avg`, …).

**Dataset policy** (`figures/config.py`):

```python
GC_USE_VIB_ZPE_BY_DATASET = {"PBE": True, "PBE0": False}
```

PBE grand-canonical NPZ includes full vib/ZPE; PBE0 is electronic-only until frequencies are added to `IryOxPBE0.traj`.

### 3.3 Ensemble weights and observables

For each grid point, Boltzmann weights are built from all structures:

\[
w_i = \exp\bigl(-\beta (G_i - G_{\min})\bigr) \times g_i
\]

- \(\beta = 1/(k_B T)\)
- \(G_{\min}\) = minimum \(G_i\) at that \((T,p)\) (numerical stability)
- \(g_i\) = **point-group degeneracy** for achiral vs chiral groups (`POINT_GROUP_DEGEN` in `thermo.py`, from `pymatgen.symmetry.analyzer.PointGroupAnalyzer`)

Normalised populations:

\[
P_{g,i}(T,p) = \frac{w_i}{\sum_j w_j}
\]

**Global minimum map**: `phase_index(T,p) = \arg\min_i G_i`.

**Ensemble-averaged oxidation**:

\[
\langle n_O/n_{\mathrm{Ir}} \rangle(T,p) = \sum_i P_{g,i}\, (n_O/n_{\mathrm{Ir}})_i
\]

Analogous ensemble averages are computed for Hirshfeld Ir charge, moments of inertia, configurational entropy \(S_{\mathrm{conf}} = -k_B \sum_i P_{g,i} \ln P_{g,i}\), and (when decomposition is enabled) each ΔG contribution term.

### 3.4 Thermodynamic grid

Default production grid (`figures/config.py`):

| Parameter | Value |
|-----------|-------|
| `T_min`, `T_max` | 100 K, 1000 K |
| `n_T` | 400 (linear spacing) |
| `P_min`, `P_max` | 10⁻¹⁰ Pa, 10¹⁰ Pa |
| `n_P` | 400 (log spacing) |

Structures are enumerated in trajectory order; **row count must match** ACE feature CSV length (strict check in `build_joined_table`).

### 3.5 GC variants (full vs novib)

`aist/core/gc_variants.py` runs two PBE calculations on the same grid:

| Variant | Vibrational / ZPE | Stored tensors |
|---------|-------------------|----------------|
| **full** | On cluster + O₂ | `full_Pgi`, `full_phase_index`, `full_x_per_non_o` |
| **novib** | Off | `novib_Pgi`, `novib_phase_index`, `novib_x_per_non_o` |

Saved to `figures/cache/gc_variants.npz`. Used to compare how vibrations shift populations and phase boundaries (Figures 7, 9, 10; PBE novib view in Figures 3 and 12) without recomputing on every plot.

### 3.6 Vibrational analysis beyond GC

`aist/core/vibrational.py` provides structure-level metrics used in figures:

- **ZPE** per structure and **ΔZPE** relative to bare Irₙ stoichiometry.
- **Characteristic temperature** Θᵢ = hνᵢ/k_B for each vibrational mode.
- **Energy-weighted ensemble Θ_eff(T)**: at each T, modes contribute to a thermal energy-weighted mean Θ; structures are combined with \(P_{g,i}\) on the GC grid.
- **T = Θ_eff contour** on phase maps (where \(T/\Theta_{\mathrm{eff}} = 1\)).

### 3.7 Grand-canonical cache files

| File | Content |
|------|---------|
| `grandcanonical_results.npz` | Full PBE GC (vib/ZPE, decomposition, `G_tensor`, `Pgi`, …) |
| `grandcanonical_PBE0.npz` | PBE0 GC (electronic-only) |
| `gc_variants.npz` | PBE full vs novib `Pgi` tensors |

Typical size at 400×400 grid: ~1.3 GB per full PBE NPZ.

---

## 4. Combining clustering with thermodynamics

Clustering and GC are **independent calculations** on the same structure lists. They are merged only at the **per-structure index** (trajectory frame order).

### 4.1 Join table

`build_joined_table(gc_data, ace_rows, dataset)` (`aist/ace/join.py`):

1. Verifies `len(Pgi) == len(ace_rows)` (raises if mismatch).
2. Aligns `Pgi`, `phase_index`, `x_per_non_o`, `Ts`, `Ps` with ACE columns: `cluster`, `Ir-A*_frac`, `O-A*_frac`, `n_Ir`, `n_O`, `formula`.

The join is the basis for all “clustering × thermodynamics” figures.

### 4.2 Ensemble-weighted scalar maps

For any per-structure scalar \(v_i\) (cluster index, HOMO–LUMO gap, etc.):

\[
\langle v \rangle(T,p) = \frac{\sum_i P_{g,i}(T,p)\, v_i}{\sum_i P_{g,i}(T,p)}
\]

Implemented as `ensemble_weighted_mean_map(Pgi, per_structure)`.

### 4.3 Category population fractions on the phase diagram

For structure clusters or atom motifs, the **population fraction** of category \(c\) at \((T,p)\):

**Structure clusters:**

\[
f_c(T,p) = \frac{\sum_{i:\, C_i = c} P_{g,i}(T,p)}{\sum_j P_{g,j}(T,p)}
\]

**Ir motifs** (atom-weighted):

\[
f_m^{\mathrm{Ir}}(T,p) = \frac{\sum_i P_{g,i}\, n_{\mathrm{Ir},i}\, f_{m,i}^{\mathrm{Ir}}}{\sum_i P_{g,i}\, n_{\mathrm{Ir},i}}
\]

**O motifs** analogously with \(n_{\mathrm{O},i}\).

These produce fields of shape `(k_categories, n_P, n_T)` for RGB mixing or stacked population vs T plots.

### 4.4 Slices at fixed pressure

At fixed \(p_{\mathrm{O_2}}\) (e.g. ambient 101325 Pa):

- `atom_motif_population_vs_T_at_pressure` — smoothed stacks of Ir + O motif populations vs T.
- `o_bridging_terminal_interface_vs_T_at_pressure` — ratio of O-A bridging vs terminal motif populations (indices O-A0 / O-A1 in current convention).
- `oxidation_slice_at_pressure` — ensemble and global-minimum \(n_O/n_{\mathrm{Ir}}\) vs T.

Gaussian smoothing along T (`sigma = 1.0` by default for Figure 4) is applied to population stacks after binning.

### 4.5 Fixed thermodynamic points

Representative conditions for population line plots:

| Label | T (K) | p_O₂ (Pa) |
|-------|-------|-----------|
| 273 K, 1 atm | 273 | 101325 |
| 500 K, 1 atm | 500 | 101325 |

`structure_populations_at_fixed_point` interpolates to the nearest grid `(T,p)` and returns `Pgi[:, p_idx, t_idx]` for each structure. `ranked_cumulative_population_lines` orders structures by novib population and selects the set covering 97% cumulative population for side-by-side vib vs novib comparison.

### 4.6 Fair PBE vs PBE0 comparisons

When comparing functionals on phase maps **without** vibrational bias:

- Load PBE tensors from `gc_variants.npz` **novib** branch (`load_gc_pbe_novib_view`).
- Load PBE0 from `grandcanonical_PBE0.npz` (already electronic-only).

This is used in Figures 3, 10, and 12 so PBE columns use the same level of thermodynamic theory as PBE0.

---

## 5. End-to-end workflow

```text
data/*.traj, *.xyz
        │
        ├─► aist/ace/pipeline.py  ──► figures/cache/ace/  (motifs, clusters, CSV)
        │
        ├─► aist/core/grandcanonical.py  ──► grandcanonical_{PBE,PBE0}.npz
        │
        └─► aist/core/gc_variants.py  ──► gc_variants.npz

figures/cache/
        │
        └─► figures/figXX_*.py  ──► figures/outputs/*.svg
```

Rebuild everything:

```bash
python3 figures/build_all.py --rebuild-cache
```

Cache staleness: GC NPZ is rebuilt if raw trajectories change or if `GC_N_T` / `GC_N_P` in `figures/config.py` differ from stored `Ts`/`Ps` lengths.

---

## 6. Suggested paper Methods outline

The following mapping may help when drafting prose:

1. **Structure set** — DFT geometries (PBE, PBE0), stoichiometry grid, energies and frequencies in trajectory metadata.
2. **Local motif identification** — SOAP, PCA, UMAP, fixed k for Ir (4) and O (3); oxidation ordering of Ir motifs; O motif geometry (bridging / terminal / O₂); structure clustering on motif fractions + extent-gated shape + O₂ flag (k = 9) with chemical interpretation of C0–C8 (§2.4–2.6).
3. **Grand-canonical ensemble** — reaction scheme, partition functions, degeneracy, grid; distinction PBE (full thermo) vs PBE0 (electronic).
4. **Observables** — \(P_{g,i}\), ensemble averages, motif/cluster population fields.
5. **Sensitivity analyses** — novib vs full (`gc_variants`); global minimum vs ensemble (Figure 7 quadrants).
6. **Electronic structure proxies** — Hirshfeld charges/spins, HOMO–LUMO gaps from `atoms.info` (not recomputed in this pipeline).

All numerical defaults above are the ones used for the current figure set unless noted otherwise in `figures/config.py`.
