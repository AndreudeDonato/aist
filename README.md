# aist

Ab initio statistical thermodynamics of cluster ensembles, on ASE `Atoms` objects.

`aist` takes a set of structures computed with any calculator and treats them as one grand-canonical ensemble
in contact with gas reservoirs. The set can hold many isomers of many stoichiometries, for example metal-oxide
clusters M_yO_x. Each structure needs an energy, and optionally harmonic frequencies and a spin multiplicity.
`aist` returns the Boltzmann population of every structure on a grid of temperature and reservoir pressures (or
chemical potentials). From the populations you get ensemble observables:
- the average composition;
- the most stable structure and its population;
- the configurational entropy;
- the population-weighted mean of any per-structure property (charges, gaps, motif labels, spectra).

The reservoir chemical potentials come from the full ideal-gas partition functions. The per-structure free
energies come from harmonic vibrations with ZPE, electronic (spin) degeneracy and, optionally, rotation and
translation. Both are checked against ASE's `IdealGasThermo` and `HarmonicThermo`.

## Install

From a clone of this repository:

```bash
pip install -e .                        # import name: aist
pip install -e ".[motifs,test]"         # + SOAP/UMAP/KMeans motif classification (aist.ace) and the tests
```

Releases will be published on PyPI as `aist-thermo` (`pip install aist-thermo`).

## Quick start

```python
import numpy as np
from ase.io import read
from aist import GrandCanonicalEnsemble, Reference, Reservoir, ThermoTerms

clusters = read("clusters.traj", ":")    # energies + info["frequencies"], info["zpe"], info["total_spin"]
ensemble = GrandCanonicalEnsemble(
    clusters,
    references=[Reference(read("Pt_atom.xyz"))],                  # μ_Pt fixed: energy per atom
    reservoirs=[Reservoir(read("O2.xyz"), multiplicity=3)],       # μ_O2(T, p): full ideal-gas partition function
    terms=ThermoTerms(vibrations=True, zpe=True),                 # your choice; default: electronic only
)
result = ensemble.evaluate(T=np.linspace(100, 1000, 150), p={"O2": np.logspace(-10, 10, 150)})

result.populations          # (n_structures, n_p, n_T)
result.composition("O")     # ⟨n_O / n_Pt⟩ map
result.most_stable()        # index of the lowest-ΔG structure at every (p, T)
result.average(values)      # population-weighted mean of any per-structure property
result.subset(counts={"Pt": 6}).composition("O")   # one cluster size, populations renormalised
```

`examples/cu_oxide_emt.py` runs the whole chain on objects built in ASE, in about 20 s:
1. relax random Cu₃O_x and Cu₄O_x structures with ASE's EMT potential;
2. compute their vibrations with `ase.vibrations`;
3. weight them in O₂ on a (T, p) grid.

EMT is a toy potential. For science, attach DFT data.

## What is computed

For structure *i* with element counts n_ie, references *r* (elements held at fixed μ, e.g. the isolated metal atom)
and reservoir molecules *k*:

    ΔG_i(T, {p_k}) = (1/N_i) [ G_i(T) − Σ_r a_ir μ_r − Σ_k ν_ik μ_k(T, p_k) ]
    P_i = g_i exp(−ΔG_i / kT) / Σ_j g_j exp(−ΔG_j / kT)

- **Balance.** The coefficients a_ir and ν_ik balance each structure exactly, in rational arithmetic. A set of
  reservoirs that is linearly dependent, or that cannot supply an element, stops with an error. For example,
  O₂ + H₂ + H₂O together is dependent.
- **Free energy of a structure.** G_i = E_i − kT ln(2S+1) plus the terms the user switches on: ZPE, harmonic
  vibrations, rotation, translation (`ThermoTerms`). Which approximation fits is the user's call; by default only the
  electronic contribution (energy and spin degeneracy) enters.
- **Normalisation.** N_i is the number of reference atoms (`normalize="reference"`, the default: a free energy
  per metal atom), 1 (`"cluster"`), or the count of a chosen element.
- **Degeneracy.** g_i is the chirality degeneracy from the point group (2 for chiral clusters, which counts both
  enantiomers).

## Conditions, reservoirs and variants

- **Conditions as axes.** Each reservoir is set either by a pressure (`p={"O2": ...}`, in Pa) or by a chemical
  potential relative to its DFT energy (`dmu={"O2": ...}`, in eV). Array-valued conditions become axes and
  scalars stay fixed. For example, a map at fixed temperature over two reservoirs:

  ```python
  ensemble = GrandCanonicalEnsemble(structures, references=[Reference(metal)],
                                    reservoirs=[Reservoir(o2, multiplicity=3), Reservoir(h2o, multiplicity=1)])
  result = ensemble.evaluate(T=500.0, p={"O2": np.logspace(-10, 5, 100), "H2O": np.logspace(-10, 5, 100)})
  ```
- **Every contribution is a switch.**
  - `ThermoTerms(vibrations=, zpe=, electronic=, rotation=, translation=)` sets the terms of the structures.
  - Each reservoir takes its own `terms=` (default all on).
  - `GasSpecies` has `vibrations=`, `rotation=` and `translation=`.
  - The electronic term of a reservoir needs its spin multiplicity: pass `multiplicity=` or `electronic_states=`,
    store `info["total_spin"]["J"]`, or switch the term off.
- **Variants.** `ensemble.variant(vibrations=False, zpe=False)` switches the same terms on structures and
  reservoirs, giving the full, no-vibration, vibrations-only and ZPE-only comparisons. Cached partition functions
  are shared.
- **Other data sources.** `ensemble.with_structures(new_list)` keeps the rows, balance and degeneracies but takes
  new energies or frequencies, for example from a machine-learned potential.
- **Corrections.**
  - `result.shifted(delta)` re-weights after a free-energy correction.
  - `SupportShift(ev_per_reference, {"O": ev})` is a flat adhesion proxy for supported clusters.
  - `GasSpecies(atoms, pressure)` adds volatile products (e.g. MO₂(g)) to the pool at a fixed partial pressure.

## Input conventions

Each `Atoms` object carries:
- **Energy (eV):** from its calculator (e.g. a `SinglePointCalculator`) or from `info["energy"]`.
- **`info["frequencies"]`:** all 3N harmonic frequencies in cm⁻¹, ascending, with the 5 or 6 rigid-body modes
  first. Imaginary modes are given as negative values.
- **`info["zpe"]`:** zero-point energy (eV).
- **`info["total_spin"]["J"]`:** spin multiplicity 2S+1. A structure without it is treated as a singlet, with a
  warning.

## Motif classification (`aist.ace`, needs `[motifs]`)

The motif classification works per atom:
1. SOAP descriptors (DScribe);
2. PCA, then UMAP;
3. KMeans clustering of atoms and structures.

An optional scheme names the oxygen motifs by bonding: bridging, terminal and molecular. The labels can be joined
to the ensemble populations. It runs with `aist.ace.pipeline.main(data_dir=..., results_dir=...)`.

## The 0.1 file-based API

`aist.core.grandcanonical.compute_gc`, `run_gc` and `aist.core.gc_variants.run_gc_variants` keep the 0.1
interface:
- trajectory files in, NPZ dictionaries out;
- one single-element reference and one single-element reservoir molecule.

They now run on the core above and give the same arrays as 0.1, bit for bit. The helpers in `aist.core`
(subsets, gaps, support, reservoir patches, vibrational metrics) work on those dictionaries.

## Tests

```bash
python -m pytest
```

The suite checks:
- the thermodynamics against ASE and closed forms;
- generality: other metals, H₂ and H₂O reservoirs, several reservoirs at once;
- bit-for-bit continuity with the 0.1 code, kept frozen in `tests/legacy/`.

`tests/test_paper_reproduction.py` also checks the IryOx paper ensembles bit for bit when `AIST_PAPER_DIR`
points at the paper repository.

## Citing

Developed for the IryOx papers on gas-phase Ir_yO_x clusters (Andreu A. de Donato, Universitat de Barcelona; in
preparation). Methods notes: `docs/`.

## License

MIT (see `LICENSE`).
