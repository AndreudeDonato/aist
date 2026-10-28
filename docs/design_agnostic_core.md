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

## 3. Objects (proposal: object model, names, ASE-style `get_*` methods)

```python
from ase.io import read
from aist import GrandCanonicalEnsemble, Reference, Reservoir, ThermoTerms

ens = GrandCanonicalEnsemble(
    read("clusters.traj", ":"),
    references=[Reference(read("Pt_atom.xyz"))],                 # mu per Pt atom = E / natoms
    reservoirs=[Reservoir(read("O2.xyz"), multiplicity=3),
                Reservoir(read("H2O.xyz"), multiplicity=1)],
    terms=ThermoTerms(vibrations=True, zpe=True, electronic=True, rotation=False, translation=False),
)
res = ens.evaluate(T=500.0, p={"O2": np.logspace(-10, 5, 100), "H2O": np.logspace(-10, 5, 100)})
res.populations            # (n_structures, 100, 100)
res.composition("O")       # <n_O / N_ref> map
res.average(values)        # population-weighted mean of any per-structure property
res.most_stable()          # argmin ΔG index map
res.subset(counts={"Pt": 6})   # renormalised per-size view
res.shifted(delta)         # re-ensembled after a free-energy correction (support, user terms)
ens.get_populations(T=300.0, p={"O2": 2.1e4, "H2O": 1e3})   # ASE-style convenience: one weight per Atoms object
```

**Classes**
- `Reservoir(atoms, *, name=None, multiplicity=None, electronic_states=None, symmetry_number=None, terms=ThermoTerms.all())`
  - `name` defaults to the chemical formula.
  - `chemical_potential(T, p)` is cached per grid.
- `Reference(atoms=None, *, energy=None, element=None)`: an elemental `Atoms` (μ = E per atom) or an explicit energy.
- `ThermoTerms`: the per-term flags. Structures default to the electronic term only; reservoirs to all terms
  (`ThermoTerms.gas()`).
  `ens.variant(vibrations=False, zpe=False)` switches the same terms on structures and reservoirs: the full / novib /
  vib-only / ZPE-only variants.
- `GasSpecies(atoms, pressure, multiplicity=None)`: a volatile product (IrO2(g), IrO3(g)) appended to the pool. It
  takes the structure terms plus its own translation and rotation at its partial pressure.
- `GrandCanonicalEnsemble(structures, *, references, reservoirs, terms, normalize="reference", degeneracy="chirality", gas=())`
  - Attributes: `elements`, `compositions` (integer matrix), `coefficients` (a, ν), `normalization` (N_i), `formulas`.
  - `evaluate(T, p=None, dmu=None, axes=None, dtype=np.float32)` returns an `EnsembleResult`.
- `EnsembleResult`
  - Fields: `axes` (ordered name → values), `G` (n × grid), `populations`, `degeneracy`, the ensemble's composition data.
  - Methods: `composition`, `average`, `most_stable`, `most_stable_population`, `configurational_entropy`,
    `subset`, `shifted`, `index` (nearest grid point; log scale for pressures), `decomposition` (ensemble-averaged
    electronic / vibrational / rotational / translational / bare-DFT parts), and `to_legacy_dict` (the
    `compute_gc` NPZ keys).
- Corrections: `SupportShift(ev_per_reference, ev_per_element={"O": 0.1})` gives δ_i = −(a + Σ_e b_e n_ie/N_i) on
  cluster rows only; gas rows are untouched. Any array δ broadcastable to (n, *grid) also works.

**Conditions**
- `T` is a scalar or a 1D array (K).
- Each reservoir is fixed by exactly one of `p[name]` (Pa, scalar or array) or `dmu[name]` (eV, relative to its
  bare DFT energy, scalar or array). Giving both, or neither, is an error.
- Array entries become axes: by default reservoir axes in reservoir order, then T, which gives Paper I's (p, T)
  layout. `axes=` sets another order.

**Precision.** G is stored in `dtype` (default float32: about 1 μeV resolution at |ΔG| ≈ 10 eV, and half the
memory). Populations are computed in float64 from the stored G, plus any correction, exactly as the 0.1 code does,
so corrections applied in sequence reproduce the cached Paper I tensors. `dtype=np.float64` is available.

## 4. Continuity with `aist_paper` (the contract)

The contract is every name `aist_paper/figures` and `paper2` import today, as of 2026-10-28:
- `aist.core.grandcanonical`: `GCConfig`, `compute_gc`, `run_gc`, `GC_CONTRIBUTION_SERIES`, `pressure_grid_index`,
  `temperature_grid_index`, `global_minimum_population_map`, `mean_ir_hirshfeld_charge`;
- `aist.core.gc_variants.run_gc_variants`;
- `aist.core.ensemble`, `reservoir`, `support`, `reference`, `subset`, `gaps`, `calibration`, `vibrational`, `minima`
  (every name in use);
- `aist.composedthermo`, `aist.thermo`, `aist.stoichiometry`, `aist.utils`, `aist.inertia`, `aist.steinhardt`,
  `aist.cache_stamp`;
- `aist.ace.*` (pipeline, join, io, common, constants, clustering, datasets, descriptors, plotting, reproducibility).

These keep their signatures and outputs. The thermodynamic ones become thin functions over the new core (Ir/O
behaviour taken from the files they are given). The exploratory `analysis/` and `scripts/` folders are best effort.
`GCConfig` keeps its fields so that the cache stamps of `aist_paper` do not change.

The old implementation is frozen in `tests/legacy_reference.py` and serves as the oracle: the compatibility
functions must give identical arrays on toy data. Nothing is removed until `aist_paper`'s wrappers have switched and
all 49 figures rebuild identically.

## 5. Tests

1. **Balance:** unique coefficients; a dependent reservoir set (O2 + H2 + H2O) raises; an element not spanned
   raises with its name; Paper I coefficients identical to `balance_reaction`.
2. **Reservoirs:**
   - a missing multiplicity raises;
   - H2 gets its own (singlet) state, with no O2 triplet;
   - μ matches ASE `IdealGasThermo` for O2, H2O and H2;
   - the `dmu` axis equals the `p` axis at the corresponding pressure.
3. **Generality:**
   - Pt + O2 runs and equals the Ir run with the element relabelled;
   - Ir + H2 gives ⟨n_H/n_Ir⟩ ≠ 0 and n_Ir counts only Ir;
   - an Ir/O/H toy set with O2 + H2O has unique balances and a fixed-T map with axes (p_O2, p_H2O). Changing the
     reservoir basis (O2 + H2 with μ_H2 = μ_H2O − μ_O2/2) gives the same populations.
4. **Continuity:** `compute_gc`, `run_gc_variants`, the gas and support patches and the subset, gap and vibrational
   helpers are bit-identical to the frozen legacy code on toy ensembles.
5. **Paper I reproduction** (`tests/test_paper_reproduction.py`, skipped unless `AIST_PAPER_DIR` is set; reads
   data files only, imports nothing from the paper repository): for both functionals on the 150 × 150 grid, with the
   IrO2/IrO3 gas species and the 0.65 eV/Ir + 0.1 eV/O support shift, `Pgi`, `G_tensor` and `phase_index` are equal
   to the cached `grandcanonical_results.npz` / `grandcanonical_PBE0.npz`; the same for the variants NPZs.

## 6. Order of work on the branch

1. Core: `aist/species.py` (Reference, Reservoir, ThermoTerms, GasSpecies), `aist/balance.py`,
   `aist/conditions.py`, `aist/gce.py` (GrandCanonicalEnsemble, EnsembleResult, SupportShift). Tests 1–3.
2. Continuity: the legacy names over the core; tests 4–5. Then a message to the FABLE session for the paper-side
   identity check.
3. Agnostic helpers:
   - `subset`, `gaps` and `vibrational` keyed by element and reference count, with the old Ir/O names as aliases;
   - the motif pipeline driven by a `MotifConfig` (species, SOAP cutoffs, k per species, seed), no module globals.
     The bridging / terminal / molecular naming becomes an optional `naming="oxide"` scheme. `aist.motifs` is the new
     home and `aist.ace` stays as an alias. "ACE" here is not the Atomic Cluster Expansion.
4. Release readiness:
   - `[project]` metadata in `pyproject.toml` with a version, plus an optional `motifs` extra for the SOAP/UMAP
     stack **(proposal)**;
   - the licence file once Andreu chooses (MIT proposed);
   - a minimal CI (install + pytest);
   - the README rewritten around a small non-IryOx example. No IryOx data in the package.

## 7. Open points for Andreu

1. Object model and names as in section 3 (proposal).
2. Default normalisation per reference atom: decided. Default terms: electronic only, decided.
3. Light install with an optional `motifs` extra (proposal).
4. Licence: MIT (proposal).
5. Add-on to ASE rather than a merge (recommended).
6. Package name on PyPI: `aist` is taken (an unrelated "AI Service Tools", v0.2.3). The import name can stay `aist`;
   the distribution needs another name, e.g. `aist-thermo` (free as of 2026-10-28).
7. Git identity: every commit in both repositories, and the global git config, have the author email wrapped in
   guillemets (`«…@gmail.com»`). GitHub cannot link such commits to an account. Fix the global config and rewrite
   the email when the repository is recreated.
