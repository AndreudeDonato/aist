# aist

Free-energy contributions of ASE `Atoms` objects, and building blocks for ensembles of them.

The core of `aist` is a small set of functions. Each takes an `Atoms` object computed with any calculator and
returns its free-energy terms at the temperatures and pressures you give:
- electronic energy and spin degeneracy;
- zero-point energy;
- harmonic vibrations;
- rotation;
- translation.

You choose which terms enter; no approximation is right or wrong.

How free energies are combined into an ensemble is also your choice. `aist` provides general pieces for that:
- Boltzmann populations of any set of structures;
- exact balancing of a structure against the reservoirs you pick;
- ideal-gas chemical potentials.

It also provides one complete construction, a grand-canonical ensemble over isomers and stoichiometries on a grid
of temperatures and pressures. `examples/` shows full setups to copy and adapt.

## Install

From a clone of this repository:

```bash
pip install -e .                        # import name: aist
pip install -e ".[motifs,test]"         # + SOAP/UMAP/KMeans motif classification (aist.ace) and the tests
```

Releases will be published on PyPI as `aist-thermo` (`pip install aist-thermo`).

## The core: contributions of one `Atoms` object

```python
import numpy as np
from aist import ThermoTerms, contributions, free_energy

T = np.linspace(100, 1000, 10)                                             # K, any shape
G = free_energy(atoms, T, terms=ThermoTerms(vibrations=True, zpe=True))    # E − kT ln(2S+1) + ZPE + F_vib
mu = free_energy(o2, T, 1e5, ThermoTerms.gas(), multiplicity=3)           # ideal-gas molecule at 1 bar
parts = contributions(o2, T, 1e5, ThermoTerms.gas(), multiplicity=3)      # each term separately (eV)
```

| Term (`ThermoTerms`) | What it is |
|---|---|
| always | energy E: the attached calculator's, or `info["energy"]` |
| `electronic` (default on) | −kT ln(2S+1), or the full sum over `states=[(energy, degeneracy), ...]` |
| `zpe` | `info["zpe"]`, or ½ Σ hcν over the real internal modes |
| `vibrations` | harmonic U − TS of the internal modes |
| `rotation` | rigid rotor; symmetry number from the point group unless given |
| `translation` | ideal gas at pressure p, including pV (H − TS = −kT ln q) |

`ThermoTerms.gas()` switches every term on. The individual functions (`electronic`, `zero_point_energy`,
`vibrational`, `rotational`, `translational`) return energies and entropies, and match ASE's
`IdealGasThermo` and `HarmonicThermo` (see `tests/`).

## Ensembles: your construction

A canonical ensemble of isomers takes three lines:

```python
from aist import boltzmann_populations
from aist.ensembles import chirality_degeneracy

G = np.array([free_energy(a, T, terms=ThermoTerms(vibrations=True, zpe=True)) for a in isomers])
P = boltzmann_populations(G, T, [chirality_degeneracy(a) for a in isomers])
```

For structures that exchange atoms with gas reservoirs, `GrandCanonicalEnsemble` balances every structure exactly
against fixed references and reservoir molecules, then weights it on a grid of conditions:

    ΔG_i = (1/N_i) [ G_i(T) − Σ_r a_ir μ_r − Σ_k ν_ik μ_k(T, p_k) ],    P_i ∝ g_i exp(−ΔG_i / kT)

```python
from aist import GrandCanonicalEnsemble, Reference, Reservoir

ensemble = GrandCanonicalEnsemble(
    clusters,                                                     # any number of isomers and stoichiometries
    references=[Reference(metal_atom)],                           # μ fixed: energy per atom
    reservoirs=[Reservoir(o2, multiplicity=3), Reservoir(h2o, multiplicity=1)],   # μ(T, p): ideal gas
    terms=ThermoTerms(vibrations=True, zpe=True),                 # one for all rows, or one per row
)
result = ensemble.evaluate(T=500.0, p={"O2": np.logspace(-10, 5, 100), "H2O": np.logspace(-10, 5, 100)})
result.populations                    # (rows, 100, 100): a map at fixed T over two pressures
result.composition("O")               # ⟨n_O / n_metal⟩
result.most_stable()                  # lowest-ΔG row at every point
result.subset(counts={"Pt": 6})       # one cluster size, populations renormalised
result.shifted(delta)                 # re-weighted after any correction to ΔG (e.g. a support proxy)
```

- **Conditions.** Each reservoir is set by a pressure (`p=`, Pa) or a Δμ relative to its DFT energy (`dmu=`, eV).
  Array-valued T, p or Δμ become axes; scalars stay fixed.
- **Balance.** The balance uses exact rational arithmetic. A dependent set of reservoirs (e.g. O₂ + H₂ + H₂O), or
  an element that nothing supplies, raises `BalanceError`.
- **Normalisation.** N_i is the number of reference atoms (`normalize="reference"`, the default), 1 (`"cluster"`),
  or the count of a chosen element.
- **Degeneracy.** g_i is the chirality degeneracy (`degeneracy="chirality"`), `None`, or your own array.
- **Gas-phase products** (e.g. MO₂(g)) enter as rows with their own terms, such as translation and rotation at
  their partial pressure.
- **Variants.** `ensemble.variant(vibrations=False, zpe=False)` switches the same terms on every row and
  reservoir.
- **Other data sources.** `ensemble.with_structures(other)` keeps the rows but takes new data, e.g. frequencies
  from a machine-learned potential.

## Examples

| Script | What it shows |
|---|---|
| `examples/water_contributions.py` | the core: each term of a molecule's free energy, attached DFT-style data |
| `examples/isomers_canonical.py` | a canonical ensemble built by hand: Cu₄Ag₂ isomers (EMT) vs T |
| `examples/cu_oxide_emt.py` | a grand-canonical ensemble: Cu₃O_x and Cu₄O_x clusters (EMT) in O₂ on a (T, p) grid |

EMT is a toy potential that ships with ASE. For science, attach DFT energies, frequencies and spin
multiplicities.

## Input conventions

- **Energy (eV):** the attached calculator's (e.g. a `SinglePointCalculator`), or `info["energy"]`.
- **`info["frequencies"]`:** all 3N harmonic frequencies in cm⁻¹, ascending, with the 5 or 6 rigid-body modes
  first. Imaginary modes are negative.
- **`info["zpe"]`:** zero-point energy (eV); optional.
- **`info["total_spin"]["J"]`:** spin multiplicity 2S+1. If it is missing and `multiplicity=` is not passed, the
  electronic term uses degeneracy 1 and warns.

## Motif classification (`aist.ace`, needs `[motifs]`)

Per-atom SOAP descriptors (DScribe), then PCA, UMAP and KMeans clustering of atoms and structures. Run it with
`aist.ace.pipeline.main(data_dir=..., results_dir=...)`.

## Tests

```bash
python -m pytest
```

The suite checks the contributions against ASE's thermochemistry and closed forms, and the ensembles' generality
(other metals, other reservoirs, two reservoirs at once, exact balance). `tests/test_paper_reproduction.py`
reproduces the IryOx paper ensembles bit for bit when `AIST_PAPER_DIR` points at the paper repository.

## Citing

Developed for the IryOx papers on gas-phase Ir_yO_x clusters (Andreu A. de Donato, Universitat de Barcelona; in
preparation).

## License

MIT (see `LICENSE`).
