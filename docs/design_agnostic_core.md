# Design note: a species-agnostic, ASE-native grand-canonical core

Branch `agnostic-core`, started 2026-10-28. Decided by Andreu on 2026-10-28:
- make the code agnostic;
- keep `aist_paper` working but separate from the library;
- allow several chemical potentials at once (example: a map at fixed T with an O2 and an H2O reservoir);
- make `aist` public when this work is done.

Points marked **(proposal)** are still Andreu's to confirm.

## 1. Goals and non-goals

Goals:
1. **Species-agnostic.**
   - Any reference element(s) and any reservoir molecule(s).
   - Compositions are read from each structure's symbols.
   - Every free-energy contribution is the user's choice (`ThermoTerms` for structures and for each reservoir,
     switches on gas species). An unbalanced element, a dependent set of reservoirs, or an electronic term switched
     on without a multiplicity stops with an error that names the problem.
2. **Several chemical potentials.** Any array-valued choice of T, p_k or Δμ_k becomes a grid axis; scalars are fixed.
   1D cuts and 2D maps (and N-D grids) use the same code.
3. **Works on ASE objects.** It takes lists of `Atoms` in memory and needs no files. Energies come from the attached
   calculator or `info["energy"]`; frequencies, zpe and spin come from `atoms.info`, as today.
4. **Cheap variants.** Per-structure thermodynamic terms are kept separately, so switching a term on or off, or
   rebuilding with replaced frequencies or energies, recomputes only what changed. Reservoir μ grids are cached in
   the `Reservoir` object.
5. **Exact continuity.** The new API reproduces the Paper I ensembles bit for bit. The old public names stay as thin
   functions over the new core until `aist_paper` has switched.

Non-goals for this branch:
- anharmonic or quasi-RRHO corrections;
- kinetics;
- periodic supports (the support proxy stays a free-energy correction).

## 2. Physics

Structure *i* has element counts *n_ie*. References *r* are fixed chemical potentials per atom (normally the
isolated metal atom). Reservoirs *k* are gas molecules with μ_k(T, p_k).

**Balance.** Solve n_i = Σ_r a_ir c_r + Σ_k ν_ik c_k exactly (rational arithmetic, then float), where c_r and c_k
are the composition vectors of the references and reservoirs. The set [c_r | c_k] must:
- be linearly independent. Otherwise it is an error: O2 + H2 + H2O together is underdetermined, since
  μ_H2O = μ_H2 + μ_O2/2 at equilibrium.
- span every element of every structure. Otherwise it is an error that names the element and a structure.

**Free energy of a structure.** G_i(T) = E_i − kT ln J_i + [ZPE_i] + [F_vib(T)] + [F_rot] + [F_trans(p_s)]. The terms in
brackets are switched on by the user. **Decided (Andreu, 2026-10-28):** with no `terms=`, only the electronic
contribution enters. Translation needs an explicit pressure p_s. The IryOx papers pass vibrations + ZPE explicitly.
- J_i comes from `info["total_spin"]["J"]`.
- A structure without J is treated as a singlet, with one warning listing the structures concerned.

**Chemical potential of a reservoir.** μ_k(T, p) is the full ideal-gas free energy of the molecule:
- translation, including pV;
- rotation, with the symmetry number from the point group or given explicitly;
- vibration and ZPE;
- electronic degeneracy.

Every term is a switch (`terms=`, default all on). The electronic term needs the multiplicity, given explicitly or
present in `info["total_spin"]`. A reservoir without one works with that term switched off; the 0.1 API keeps its O2
triplet for O2 files without spin data.

**Formation free energy.** ΔG_i = (1/N_i) [G_i − Σ_r a_ir μ_r − Σ_k ν_ik μ_k]. The choice of N_i is
`normalize="reference"` (default, the number of reference atoms: per Ir in Paper I), `"cluster"` (N_i = 1), or an
element symbol (that element's count). **Default per reference atom (decided, Andreu 2026-10-28).**

**Ensemble.** P_i = g_i exp(−ΔG_i / kT) / Ξ. The degeneracy g_i defaults to the chirality factor from the point group
(2 for chiral groups), as today. An explicit array or `None` (all 1) is also accepted.

**Lowest-G structure.** The "most stable" structure is argmin ΔG, not argmax P: with g = 2, argmax P would pick a
chiral isomer up to kT ln 2 above the minimum.

## 3. Layout (decided by Andreu, 2026-10-28: "this is the way")

The core is the free-energy contributions of one `Atoms` object. How they are combined into an ensemble is the
user's choice, so the library offers general building blocks and one generalised construction. Systems such as
IryOx or several chemical potentials live in example scripts, never hard-coded.

- **`aist.contributions` (the core).** Plain functions on an `Atoms` object, on any T (and p):
  - `electronic`, `zero_point_energy`, `vibrational`, `rotational`, `translational`, each returning (U, S);
  - `free_energy(atoms, T, p, terms)` = E + ZPE + Σ U − T Σ S over the switched-on terms;
  - `contributions(...)`: the per-term pieces.

  `ThermoTerms` holds the switches: by default the electronic term only, and `ThermoTerms.gas()` switches every
  term on. Without a known multiplicity the electronic term uses degeneracy 1 and warns, the same rule for every
  `Atoms` object.
- **`aist.ensembles` (building blocks).**
  - `boltzmann_populations(G, T, degeneracy)` and `chirality_degeneracy(atoms)`;
  - `Reference` (fixed μ per atom) and `Reservoir` (μ = the molecule's free energy with its terms, all on by
    default);
  - `GrandCanonicalEnsemble(structures, references, reservoirs, terms, normalize, degeneracy)`: `terms` is one
    `ThermoTerms` or one per row, so gas-phase products are just rows with their own terms. Methods: `evaluate`,
    `variant`, `with_structures`, `count`.
  - `EnsembleResult`: `G`, `populations`, `phase_index`; `composition`, `average`, `most_stable`, `subset`,
    `shifted` (any correction, e.g. a support proxy), `decomposition`, `index`.
- **`aist.balance`** (exact balance, `BalanceError`) and **`aist.conditions`** (T, p and Δμ axes).
- **`examples/`:** `water_contributions.py` (the core), `isomers_canonical.py` (an ensemble built by hand),
  `cu_oxide_emt.py` (the grand-canonical construction).
- **Not in the library (Andreu, 2026-10-29):** the SOAP/UMAP/KMeans motif classification (`aist.ace`), the
  inertia shape descriptors, and the IryOx vibrational and minima analyses. They are paper analyses and live in the
  paper repository (`figures/ace`, `figures/utils/{inertia,vibrational,minima}.py`).

ΔG is stored in float32 by default. Populations are computed with the arithmetic of the 0.1 code, which keeps the
Paper I ensembles bit-identical. Gas rows are computed in one step: 0.1 rounded them twice (base, then patch), so
they agree with it to float32 rounding (≤ 1e-6 eV), with the same lowest-G structure everywhere.

## 4. Continuity with `aist_paper`

The paper's figure code was switched to these names by the FABLE session (aist_paper branch `agnostic-switch`).
47 of 49 figures are byte-identical. Fig. 6 and the rutile maps agree to float32 rounding (ΔG ≤ 1e-6 eV, the same
lowest-G structure everywhere).

After that switch, the 0.1 layer was deleted (Andreu, 2026-10-28):
- the file API: `compute_gc`, `run_gc`, `run_gc_variants`, `GCConfig`;
- the NPZ patch and view helpers;
- `cache_stamp`, `composedthermo`, `stoichiometry` and `steinhardt`;
- the frozen 0.1 copy used as a test oracle.

The paper keeps its own copies of the helpers it needs. `tests/test_paper_reproduction.py` remains the bit-for-bit
anchor.

## 5. Tests

1. **Contributions** against ASE `IdealGasThermo` / `HarmonicThermo` and closed forms. Each term can be switched
   off, the default is electronic only, and a missing spin means degeneracy 1 with a warning.
2. **Ensembles:**
   - exact balance; dependent reservoirs and unsupplied elements raise;
   - generality: Pt equals Ir relabelled, Ir + H2, O2 + H2O at fixed T, a reservoir basis change, the Δμ axis
     equals the pressure axis;
   - per-row terms; variants; `with_structures`; subsets; `shifted`;
   - `contributions` add up to ΔG and `decomposition` is their average.
3. **Paper I reproduction** (needs `AIST_PAPER_DIR`): main ensembles and the four variants, both functionals, bit
   for bit.

## 6. State (2026-10-28)

1. Done on branch `contributions-core`:
   - the core and the building blocks;
   - the 0.1 API (`compute_gc`, `run_gc_variants`) as a thin layer, bit-identical to 0.1 on every key (later
     removed, section 4);
   - Paper I reproduced bit for bit;
   - licence, packaging, CI, README and examples.
2. Done: aist_paper switched; the 0.1 layer removed (see section 4).
3. Done: the motif classification and the IryOx analysis helpers moved to the paper repository (Andreu,
   2026-10-29). The motif refactor was cancelled.

## 7. Open points for Andreu

1. Object model and names: section 3, decided.
2. Default normalisation per reference atom: decided. Default terms: electronic only, decided.
3. Light install with an optional `motifs` extra (proposal).
4. Licence: MIT (proposal).
5. Add-on to ASE rather than a merge (recommended).
6. Package name on PyPI: `aist` is taken (an unrelated "AI Service Tools", v0.2.3). The import name can stay `aist`;
   the distribution needs another name, e.g. `aist-thermo` (free as of 2026-10-28).
7. Git identity: every commit in both repositories, and the global git config, have the author email wrapped in
   guillemets (`«…@gmail.com»`). GitHub cannot link such commits to an account. Fix the global config and rewrite
   the email when the repository is recreated.
