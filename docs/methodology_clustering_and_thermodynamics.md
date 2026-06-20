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

1. **StandardScaler** on SOAP vectors (optional, enabled by default).
2. **PCA** retaining **95%** of cumulative variance (`pca_variance_target = 0.95`).
3. **UMAP** to 2D (`n_neighbors = 30`, `min_dist = 0.0`, `metric = euclidean`, `init = pca`, `random_state = 42`).
4. **KMeans** on the UMAP embedding; cluster count **k ∈ {2,…,8}** chosen by **maximum silhouette score** on a random subsample of 5000 points (full silhouette is O(n²)).

After KMeans, clusters are **relabelled by size** (largest → index 0).

**Ir atom motifs** are then **reordered by oxidation**: each Ir atom inherits the `n_O / n_Ir` ratio of its parent structure; clusters are renumbered so **Ir-A0** has the lowest mean oxidation and **Ir-A{k−1}** the highest. This makes colour maps interpretable along the oxidation axis (coolwarm).

**O atom motifs** keep size-based ordering (no oxidation reorder pass in the current code).

Labels are written back to per-structure per-atom lists and stored in `motif_atoms_*.json` for visualisation.

### 2.5 Structure-level clustering

For each structure, a **composition feature vector** is built:

- Fraction of Ir atoms in each Ir motif: `Ir-A0_frac`, …, `Ir-A{k−1}_frac`
- Fraction of O atoms in each O motif: `O-A0_frac`, …, `O-A{k−1}_frac`

Vectors are **concatenated** (length = k_Ir + k_O). PBE and PBE0 structures are **pooled** and clustered jointly:

1. StandardScaler → PCA (95% variance) → KMeans with silhouette **k ∈ {2,…,8}**.
2. Structure clusters **reordered by mean `n_O / n_Ir`** so **C0** is least oxidised and **C{k−1}** most oxidised.

The same cluster label is assigned to both PBE and PBE0 structures in the pooled index; per-dataset CSV files (`ace_features_PBE.csv`, `ace_features_PBE0.csv`) store `frame_index`, stoichiometry, cluster id, and motif fractions.

#### Inertia shape and extent (structure descriptors)

When `use_inertia_struct_features` is enabled (`aist/ace/pipeline.py`), each structure receives **mass-weighted** inertia descriptors from `aist/inertia.py` (`ThermoPartitionFunctions.get_inertia_tensor`). Principal moments \(\lambda_1 \le \lambda_2 \le \lambda_3\) are the sorted eigenvalues of that tensor.

**Shape ratios** (Nagai-triangle coordinates, dimensionless):

\[
r_{13} = \frac{\lambda_1}{\lambda_3},\qquad r_{23} = \frac{\lambda_2}{\lambda_3}
\]

\((1,1)\) is the spherical limit **S**; the linear limit **L** is at \((0,1)\); the plane limit **P** is at \((\tfrac{1}{2},\tfrac{1}{2})\), where \(r_{13}=r_{23}=\tfrac{1}{2}\) (the symmetric minimum of the achievable band).

**Extent** (mass-normalised spread, not a third shape ratio):

\[
\log_{\mathrm{extent}}
= \tfrac{1}{2}\log(\lambda_1+\lambda_2+\lambda_3)
- \left(\tfrac{1}{2}+\frac{1}{d_{\mathrm{eff}}}\right)\log M
\]

with \(R_g = \sqrt{(\lambda_1+\lambda_2+\lambda_3)/M}\) and total mass \(M\). The exponent on \(M\) is **shape-dependent** (see below); it removes the atom-count trend in a geometry-agnostic way so extent reflects how spread-out mass is *for its mass and morphology*, not simply how many atoms the cluster has.

For **structure clustering**, only **shape** enters the feature vector, weighted per structure by batch-normalised extent (`inertia_shape_weighted_by_extent`, parameter `shape_extent_alpha`): compact structures contribute little shape information; extended oxides contribute full \((r_{13}, r_{23})\). Extent is **not** appended as its own clustering column.

#### Shape-corrected extent and bias removal

The earlier fixed normalisation (sphere-only),

\[
\tfrac{1}{2}\log I_{\mathrm{tot}} - \tfrac{5}{6}\log M
\quad\text{(\(\equiv \log R_g - \tfrac{1}{3}\log M\) with exponent \(5/6\) on \(M\))},
\]

is correct only for compact 3D objects where \(I_{\mathrm{tot}} \sim M^{5/3}\). For ideal geometries of effective spatial dimension \(d\), \(I_{\mathrm{tot}} \sim M^{1+2/d}\), so the residual \(N\)-trend left in the old extent is

| Geometry | \(d\) | \(I_{\mathrm{tot}}\sim\) | Residual \(\sim \tfrac{1}{2}\log M\) in old extent |
|----------|-------|---------------------------|-----------------------------------------------------|
| Chain (L) | 1 | \(M^3\) | \(+\tfrac{2}{3}\log M\) |
| Disk (P) | 2 | \(M^2\) | \(+\tfrac{1}{6}\log M\) |
| Sphere (S) | 3 | \(M^{5/3}\) | 0 |

Because `inertia_shape_weighted_by_extent` multiplies \((r_{13}, r_{23})\) by batch-normalised extent, linear structures accumulated extent faster with \(N\) than planar or spherical ones, inflating their weight in structure clustering.

**Effective dimension** — barycentric interpolation across Nagai corners S \((1,1)\to d=3\), P \((\tfrac{1}{2},\tfrac{1}{2})\to d=2\), L \((0,1)\to d=1\), using \(r_{13}=\lambda_1/\lambda_3\) as the interpolation coordinate, gives

\[
d_{\mathrm{eff}} = 1 + 2\,r_{13},
\]

which matches all three corners. The corrected mass exponent is \(\tfrac{1}{2} + 1/d_{\mathrm{eff}}\) (implemented in `aist/inertia.py` as `extent_normalization="shape"`; the legacy sphere formula remains available as `extent_normalization="sphere"` for comparison runs).

**Feasible region in the \((r_{13}, r_{23})\) plane.** Sorted moments always satisfy \(\lambda_1 \le \lambda_2 \le \lambda_3\), so

\[
r_{13} \le r_{23} \le 1 .
\]

A second constraint follows from the definition of the inertia tensor for point masses at \((x_i,y_i,z_i)\) relative to the centroid:

\[
I_{xx} + I_{yy} = I_{zz} + 2\sum_i m_i z_i^2 \ge I_{zz}
\]

(and cyclically: any sum of two diagonal components is at least the third). In the principal-axis frame this implies

\[
\lambda_1 + \lambda_2 \ge \lambda_3,
\]

with **equality only for coplanar** mass distributions (\(\sum_i m_i z_i^2 = 0\) in that frame). Truly three-dimensional distributions have \(\lambda_1 + \lambda_2 > \lambda_3\) strictly.

With \(\lambda_1 \le \lambda_2\),

\[
\lambda_3 \le \lambda_1 + \lambda_2 \le 2\lambda_2
\quad\Rightarrow\quad
r_{23} = \frac{\lambda_2}{\lambda_3} \ge \frac{1}{2}.
\]

The **full** achievable region is therefore not the entire upper triangle above \(r_{23} = r_{13}\), but the band

\[
r_{23} \ge \max\!\left(r_{13},\,\tfrac{1}{2}\right).
\]

The wedge below \(r_{23} = \tfrac{1}{2}\) (and the half-plane below \(r_{23} = r_{13}\)) is **mathematically empty** for any physical inertia tensor—not merely unpopulated in our dataset. Supplementary Figure `supp_momentsofinertia` plots \((r_{13}, r_{23}, \log_{\mathrm{extent}})\) in 3D with shape-corrected extent on the **E** axis and motif-composition colouring on atom panels.

### 2.6 Reproducibility

- Global seed: `ACE_RANDOM_STATE = 42` (`aist/ace/constants.py`).
- UMAP and KMeans use `random_state = 42`; UMAP `n_jobs = 1`.
- `set_reproducible_seed()` is called at pipeline start and in `figures/build_all.py`.

### 2.7 Clustering outputs (cache)

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

Saved to `figures/cache/gc_variants.npz`. Used to compare how vibrations shift populations and phase boundaries (Figures 5, 7, 9, 10) without recomputing on every plot.

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

Gaussian smoothing along T (`sigma = 1.0` by default for Figure 6) is applied to population stacks after binning.

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

This is used in Figures 5 and 10 so both columns use the same level of thermodynamic theory.

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
2. **Local motif identification** — SOAP, PCA, UMAP, KMeans, silhouette k-selection; oxidation ordering of Ir motifs and structure clusters; inertia shape/extent descriptors and Nagai-plane constraints (§2.5).
3. **Grand-canonical ensemble** — reaction scheme, partition functions, degeneracy, grid; distinction PBE (full thermo) vs PBE0 (electronic).
4. **Observables** — \(P_{g,i}\), ensemble averages, motif/cluster population fields.
5. **Sensitivity analyses** — novib vs full (`gc_variants`); global minimum vs ensemble (Figure 7 quadrants).
6. **Electronic structure proxies** — Hirshfeld charges/spins, HOMO–LUMO gaps from `atoms.info` (not recomputed in this pipeline).

All numerical defaults above are the ones used for the current figure set unless noted otherwise in `figures/config.py`.
