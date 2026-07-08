# Paper figures

Reference for the **main and supplementary figures** in the Ir/O nanoparticle paper. Each entry states what the figure shows, which data it uses, and how to regenerate it.

For computational methodology (clustering, thermodynamics, joining), see [methodology_clustering_and_thermodynamics.md](methodology_clustering_and_thermodynamics.md). For support proxy, gas reservoir, and Kempen/Reichenbach framing, see [gc_support_gas_reservoir.md](gc_support_gas_reservoir.md).

**Regeneration**

```bash
python3 figures/build_all.py                  # all figures + supplementary
python3 figures/build_all.py --only 5 8       # specific figures
python3 figures/build_all.py --rebuild-cache  # refresh figures/cache/ from data/
```

**Outputs**: `figures/outputs/*.svg`  
**Cache**: `figures/cache/` (ACE, grand-canonical NPZ)  
**Grid**: 400×400 points on \(T \in [100,1000]\) K and \(p_{\mathrm{O_2}} \in [10^{-10},10^{10}]\) Pa (configurable in `figures/config.py`).

---

## Main figures

### Figure 1 — Stoichiometry minima tiles (PBE | PBE0)

| | |
|---|---|
| **Script** | `figures/fig01_minima.py` |
| **Output** | `fig01_minima.svg` |
| **Layout** | 180 × 230 mm; PBE panel (88 mm) + 4 mm gap + PBE0 panel (88 mm); no titles |
| **Data** | `data/IryOxPBE.traj`, `data/IryOxPBE0.traj` |
| **What it shows** | Lowest-energy isomer at each Ir:O stoichiometry, rendered as inertia-oriented structure tiles. Both functionals share one **half_span** (identical Å scale / zoom). |
| **Method** | `aist.core.minima.select_stoichiometry_minima` — minimum DFT energy per \((n_{\mathrm{Ir}}, n_{\mathrm{O}})\), oriented by principal axes of inertia. |

**Supplementary outputs** (same script): `supp_minima_PBE.svg`, `supp_minima_PBE0.svg` — single-column 88 × 230 mm versions.

---

### Figure 2 — Motif clustering panel (atom + structure + centroids)

| | |
|---|---|
| **Script** | `figures/fig02_motif_clustering_panel.py` |
| **Output** | `fig02_motif_clustering_panel.svg` |
| **Layout** | 180 × 164 mm: 3×4 top (scatter \| PBE \| PBE0; Ir/O rows + moment ratios) + horizontal centroid strip |
| **Data** | `ace_atom_ir.npz`, `ace_atom_o.npz`, `ace_struct_embeddings.npz`, `ace_features_{PBE,PBE0}.csv`, `ace_centroids_pooled.traj`, `motif_atoms_pooled.json` |
| **What it shows** | Combined atom-motif UMAP + population stacks, structure-cluster PCA + stacks, principal moment-ratio triangles, pooled centroid tiles. |
| **Purpose** | Single integrated view linking atom motifs, structure clusters, inertia shape, and representative geometries. Structure PCA/stacks and pooled centroid tiles are not duplicated as separate figures. |

---

### Figure 3 — Grand-canonical phase diagrams with clustering overlays

| | |
|---|---|
| **Script** | `figures/fig03_gc_clustering_maps.py` |
| **Output** | `fig03_gc_clustering_maps.svg` |
| **Layout** | 5 × 2 (180 × 285 mm); columns PBE \| PBE0 |
| **Data** | PBE: `gc_variants.npz` (novib) + `ace_features_PBE.csv`; PBE0: `grandcanonical_PBE0.npz` + `ace_features_PBE0.csv` |
| **Thermo level** | **Electronic only** (no vib/ZPE on either column — fair cross-functional comparison). |

| Row | Panel content |
|-----|----------------|
| 1 | Ensemble \(\langle n_O/n_{\mathrm{Ir}} \rangle(T,p)\); global-minimum stoichiometry labels and phase boundaries |
| 2 | **Structure-cluster** population map — RGB mix of cluster colours; C* labels on global minimum |
| 3 | **Ir motif** population map (RGB mix of Ir-A* colours) |
| 4 | **O motif** population map (RGB mix of O-A* colours) |
| 5 | **Inertia** shape × extent (R=L, G=P, B=S; lightness = extent) |

**Purpose** | Relate thermodynamic phase behaviour to unsupervised structure and atom motifs across functionals.

---

### Figure 4 — Low-\(p_{\mathrm{O_2}}\) motifs and O per-Ir maps

| | |
|---|---|
| **Script** | `figures/fig04_gc_pressure_slices.py` |
| **Output** | `fig04_gc_pressure_slices.svg` |
| **Layout** | 3 × 2 (180 × 160 mm); columns PBE \| PBE0 |
| **Data** | `grandcanonical_results.npz` / `grandcanonical_PBE0.npz`, ACE CSVs |
| **Fixed pressure** | Motif stacks at \(p_{\mathrm{O_2}} = 10^{-5}\) Pa (`FIG04_LOW_PRESSURE_PA`) |

| Row | Panel content |
|-----|----------------|
| 1 | ⟨terminal O / \(n_{\mathrm{Ir}}\)⟩ phase map |
| 2 | ⟨bridging O / \(n_{\mathrm{Ir}}\)⟩ phase map |
| 3 | Atom motif population (%) vs **T** at low \(p_{\mathrm{O_2}}\) — O motifs below Ir motifs |

**Purpose** | Terminal/bridging O motif enrichment across \((T,p)\) and atom-level motif evolution at low oxygen pressure.

**Supplementary** | O and Ir \|Hirshfeld spin\| KDE by motif: `supp_motif_spin_kde.svg`.

---

### Figure 7 — Vibrational vs non-vibrational phase diagrams (PBE)

| | |
|---|---|
| **Script** | `figures/fig07_vib_vs_novib_quadrants.py` |
| **Output** | `fig07_vib_vs_novib_quadrants.svg` |
| **Layout** | 2 × 2 quadrant map |
| **Data** | `grandcanonical_results.npz`, `gc_variants.npz` |
| **Dataset** | **PBE only** (frequencies in traj) |

| Quadrant | Map field |
|----------|-----------|
| (0,0) Top-left | Full thermo + ensemble: \(\langle n_O/n_{\mathrm{Ir}} \rangle\) from full GC |
| (0,1) Top-right | No vib/ZPE + ensemble: novib \(\langle n_O/n_{\mathrm{Ir}} \rangle\) |
| (1,0) Bottom-left | Full thermo, **no ensemble**: stoichiometry of **global minimum** structure (full G) |
| (1,1) Bottom-right | No vib/ZPE, no ensemble: global minimum from novib G |

**Purpose** | Isolate the effect of (i) vibrational/ZPE terms and (ii) ensemble averaging vs a single-minimum picture.

---

### Figure 8 — Vibrational contributions and characteristic temperature (PBE)

| | |
|---|---|
| **Script** | `figures/fig08_vib_contrib_tchar.py` |
| **Output** | `fig08_vib_contrib_tchar.svg` |
| **Layout** | 2 × 2 |
| **Data** | `grandcanonical_results.npz`, `data/IryOxPBE.traj` |
| **Dataset** | **PBE only** |

| Panel | Content |
|-------|---------|
| (0,0) | Ensemble-averaged **ΔG contribution terms** vs **T** at \(p_{\mathrm{O_2}} = 101325\) Pa (bare DFT, electronic, vibrational, rotational, translational) |
| (0,1) | **Vibrational** ensemble contribution map \(\langle G_{\mathrm{vib}} \rangle(T,p)\) (`E_vib_avg`) with phase boundaries |
| (1,0) | **ΔZPE** of global-minimum structures vs O content, grouped by Ir count |
| (1,1) | Ensemble **energy-weighted** \(\langle \Theta_{\mathrm{eff}}(T) \rangle\) map + contour where \(T = \Theta_{\mathrm{eff}}\) |

**Purpose** | Vibrational thermodynamics in the grand-canonical framework: decomposition, maps, ZPE trends, and characteristic temperatures.

---

### Figure 9 — Global-minimum population and fixed-point structure lines (PBE)

| | |
|---|---|
| **Script** | `figures/fig09_min_pop_and_fixedpoint_lines.py` |
| **Output** | `fig09_min_pop_fixedpoints.svg` |
| **Layout** | 3 × 1 vertical (88 mm width) |
| **Data** | `grandcanonical_results.npz`, `gc_variants.npz` (`full_Pgi`, `novib_Pgi`) |

| Panel | Content |
|-------|---------|
| Top | **Population (%) of the global-minimum structure** at each \((T,p)\) — \(100 \times P_{g,i_{\min}}\) |
| Middle | At **273 K**, 1 atm: structures covering **97%** of novib population, ordered by novib pop.; lines for novib vs full (vib) population |
| Bottom | Same at **500 K**, 1 atm (primed tick labels s′1, s′2, …) |

**Purpose** | Show how much vibrational terms redistribute population among low-energy isomers at representative T–p conditions.

---

### Figure 10 — HOMO–LUMO gap maps and oxidation envelope

| | |
|---|---|
| **Script** | `figures/fig10_homo_lumo_gap_maps.py` |
| **Output** | `fig10_homo_lumo_gap_maps.svg` |
| **Layout** | 3 × 1 |
| **Data** | PBE novib GC + traj `HOMO-LUMO`; PBE0 GC + traj; `gc_variants.npz` for PBE |
| **Thermo level** | Electronic only (novib PBE, electronic PBE0) |

| Panel | Content |
|-------|---------|
| Top | PBE ensemble \(\langle \mathrm{HOMO-LUMO} \rangle(T,p)\) with stoichiometry overlay |
| Middle | PBE0 ensemble \(\langle \mathrm{HOMO-LUMO} \rangle(T,p)\) |
| Bottom | **Min/max** HOMO–LUMO gap vs \(n_O/n_{\mathrm{Ir}}\) per dataset (solid envelope lines, one legend entry per functional) |

**Purpose** | Relate electronic structure (gap) to oxidation and functional; ensemble maps plus stoichiometry-resolved spread.

---

## Supplementary figures

### Supplementary — Hirshfeld charge vs spin by atom motif

| | |
|---|---|
| **Script** | `figures/supp_atom_hirshfeld.py` |
| **Output** | `supp_atom_hirshfeld.svg` |
| **Layout** | 1 × 2 (PBE \| PBE0) |
| **Data** | Raw traj + `motif_atoms_{PBE,PBE0}.json` |
| **What it shows** | Scatter: Hirshfeld **charge** (x) vs **\|spin\|** (y), points coloured by atom motif (Ir-A*, O-A*). |

### Supplementary — Per-functional stoichiometry minima

Produced by Figure 1 script: `supp_minima_PBE.svg`, `supp_minima_PBE0.svg`.

---

## Figure–methodology map

| Figure | Clustering | Grand-canonical | Join |
|--------|------------|-----------------|------|
| 1 | — | — | — |
| 2 | Atom motifs | — | — |
| 3 | Structure centroids + atom colours | — | — |
| 4 | Structure clusters | — | — |
| 5 | Cluster + motif population maps | Novib / PBE0 electronic | `build_joined_table` |
| 6 | Motif stacks, O interface | Full GC populations | Join + fixed-p slice |
| 7 | — | Full vs novib quadrants | — |
| 8 | — | Decomposition + vib maps | ZPE / Θ from traj |
| 9 | — | Global-min pop + variants | Fixed-point `Pgi` |
| 10 | — | Ensemble gap maps | Join + traj metadata |
| Supp Hirshfeld | Atom motifs | — | Hirshfeld from traj |

---

## Scripts not in main numbering

| Path | Role |
|------|------|
| `figures/build_all.py` | Master driver for cache + all figures |
| `figures/utils/cache.py` | Builds `figures/cache/` from `data/` |
| `figures/cache/ace/atoms/visualize_motifs.py` | Interactive motif viewer (not a paper SVG) |
| `figures/cache/ace/plots/*.svg` | ACE diagnostic plots (optional; not paper figures) |
