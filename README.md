# aist

Ab initio statistical thermodynamics for clusters, on ASE `Atoms` objects: grand-canonical ensembles over isomers
and stoichiometries, harmonic vibrational and rotational thermodynamics, and SOAP / UMAP / KMeans motif
classification of atoms and structures.

Developed for the IryOx papers (Andreu A. de Donato, IQTCUB, Universitat de Barcelona). The manuscripts, their
figure pipeline and the curated datasets live in a separate repository (`aist_paper`), which imports this package.
Nothing in this package reads project data.

## Install

```bash
pip install -e .                 # editable install
python3 -m pytest tests/         # checks the thermodynamics core against ASE
```

## What is in the package

- `aist.thermo`, `aist.composedthermo` — partition functions and G = E + ZPE + U(T) − T·S per species
  (translational, rotational, vibrational, electronic), read from `Atoms.info` metadata.
- `aist.core.grandcanonical` — `GCConfig`, `compute_gc`, `run_gc`: formation free energies per reference atom on a
  (T, p) grid, Boltzmann populations with point-group degeneracies, ensemble projections of per-structure properties.
- `aist.core.gc_variants` — full / novib / vib-only / ZPE-only ensembles on one grid.
- `aist.core.vibrational` — ZPE, characteristic temperatures, energy-weighted Θ_eff maps.
- `aist.core.minima` — lowest-energy structure per stoichiometry.
- `aist.ace` — SOAP descriptors (dscribe) → PCA → UMAP → KMeans atom motifs and structure clusters, O motifs named
  by bonding (bridging, terminal, molecular): `aist.ace.pipeline.main(results_dir=..., data_dir=...)`.
- `aist.inertia`, `aist.steinhardt`, `aist.stoichiometry` (`balance_reaction`).

## Input format

ASE trajectories whose frames carry a single-point energy and, in `atoms.info`: `frequencies` (cm⁻¹, all 3N
modes), `zpe` (eV), `total_spin` (`{"J": multiplicity}`), `hirshfeld` (charges, spins, volumes) and `HOMO-LUMO`
(eV). Reference species (metal atom, O₂) as extxyz files with their energies.

## Methods

`docs/methodology_clustering_and_thermodynamics.md`.
