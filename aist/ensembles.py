"""Building blocks for ensembles of structures, and one generalised grand-canonical construction.

How free energies are combined into an ensemble is the user's choice. The general pieces are:
- ``boltzmann_populations`` for any set of free energies;
- ``aist.balance`` for exact reaction coefficients;
- ``Reference`` and ``Reservoir`` for fixed and gas-phase chemical potentials.

``GrandCanonicalEnsemble`` puts them together the way the IryOx papers do, for any elements:

    ΔG_i = (1/N_i) [ G_i(T) − Σ_r a_ir μ_r − Σ_k ν_ik μ_k(T, p_k) ],    P_i ∝ g_i exp(−ΔG_i / kT)

with N_i the number of reference atoms (``normalize="reference"``), 1 (``"cluster"``) or the count of a chosen
element, and g_i the chirality degeneracy.
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np
from ase import Atoms
from scipy.constants import Boltzmann, electron_volt

from aist.balance import balance_coefficients
from aist.conditions import Grid, build_grid
from aist.contributions import ThermoTerms, _parts, free_energy
from aist.thermo import POINT_GROUP_DEGEN
from aist.utils import get_energy

KB_EV = Boltzmann / electron_volt

__all__ = ["boltzmann_populations", "chirality_degeneracy", "Reference", "Reservoir", "GrandCanonicalEnsemble",
           "EnsembleResult"]


def boltzmann_populations(G, T, degeneracy=None) -> np.ndarray:
    """P_i = g_i exp(−G_i/kT) / Σ_j g_j exp(−G_j/kT) over axis 0 (structures); ``T`` broadcasts against G[0]."""
    G = np.asarray(G)
    weights = np.exp(-(1.0 / (KB_EV * np.asarray(T, dtype=float))) * (G - np.min(G, axis=0, keepdims=True)))
    if degeneracy is not None:
        weights *= np.asarray(degeneracy).reshape((-1,) + (1,) * (G.ndim - 1))
    return weights / np.sum(weights, axis=0)


def chirality_degeneracy(atoms: Atoms) -> int:
    """2 for a chiral point group (both enantiomers are ensemble members), else 1."""
    from pymatgen.core.structure import Molecule
    from pymatgen.symmetry.analyzer import PointGroupAnalyzer

    mol = Molecule(species=atoms.get_chemical_symbols(), coords=atoms.get_positions())
    return POINT_GROUP_DEGEN.get(PointGroupAnalyzer(mol).sch_symbol, 1)


class Reference:
    """Fixed chemical potential per atom of one element: an elemental ``Atoms`` (E per atom) or ``energy=``."""

    def __init__(self, atoms: Atoms | None = None, *, energy: float | None = None, element: str | None = None):
        if atoms is not None:
            symbols = set(atoms.get_chemical_symbols())
            if len(symbols) != 1:
                raise ValueError(f"A reference must contain one element, got {atoms.get_chemical_formula()}; "
                                 "use Reference(energy=..., element=...) otherwise")
            element = element or symbols.pop()
            energy = get_energy(atoms) / len(atoms)
        if energy is None or element is None:
            raise ValueError("Reference needs an elemental Atoms object, or both energy= and element=")
        self.element, self.energy_per_atom = element, energy
        self.name, self.composition = element, {element: 1}


class Reservoir:
    """A molecule exchanging atoms with the structures: μ(T, p) = its free energy with ``terms`` (default: all)."""

    def __init__(self, atoms: Atoms, *, name: str | None = None, multiplicity: float | None = None,
                 terms: ThermoTerms | None = None, symmetry_number: int | None = None):
        self.atoms, self.multiplicity, self.symmetry_number = atoms, multiplicity, symmetry_number
        self.name = name or atoms.get_chemical_formula()
        self.terms = terms or ThermoTerms.gas()
        self.energy = get_energy(atoms)
        self.composition = {e: atoms.get_chemical_symbols().count(e) for e in set(atoms.get_chemical_symbols())}

    def chemical_potential(self, T, p):
        """μ (eV) on broadcast-compatible ``T`` (K) and ``p`` (Pa)."""
        return free_energy(self.atoms, T, p, self.terms, multiplicity=self.multiplicity,
                           symmetry_number=self.symmetry_number)

    def with_terms(self, **changes) -> "Reservoir":
        other = object.__new__(Reservoir)
        other.__dict__.update(self.__dict__)
        other.terms = self.terms.replace(**changes)
        return other


class GrandCanonicalEnsemble:
    """Structures exchanging atoms with reservoir molecules, weighted on a grid of conditions.

    ``terms`` is one ``ThermoTerms`` for all structures or a list with one per structure (e.g. gas-phase
    molecules in the pool with their translation and rotation at their partial pressure). ``degeneracy`` is
    ``"chirality"``, ``None`` (all 1) or an array.
    """

    def __init__(self, structures, *, references, reservoirs, terms: ThermoTerms = ThermoTerms(),
                 normalize: str = "reference", degeneracy="chirality"):
        self.structures = list(structures)
        n = len(self.structures)
        self.terms = list(terms) if isinstance(terms, (list, tuple)) else [terms] * n
        if len(self.terms) != n:
            raise ValueError(f"terms: one ThermoTerms, or one per structure ({n}), got {len(self.terms)}")
        self.references, self.reservoirs = list(references), list(reservoirs)
        if not self.references:
            raise ValueError("Give at least one Reference (e.g. the isolated metal atom)")
        names = [s.name for s in self.references + self.reservoirs]
        if len(set(names)) != len(names):
            raise ValueError(f"Reference and reservoir names must be unique, got {names}")
        self.normalize = normalize
        self.elements, self.counts, X = balance_coefficients(
            self.structures, [(s.name, s.composition) for s in self.references + self.reservoirs])
        self.formulas = np.array([a.get_chemical_formula() for a in self.structures], dtype=object)
        if normalize == "reference":
            N = sum(self.count(r.element) for r in self.references)
        elif normalize == "cluster":
            N = np.ones(n, dtype=int)
        else:
            N = self.count(normalize)
        if np.any(N <= 0):
            i = int(np.nonzero(N <= 0)[0][0])
            raise ValueError(f"Row {i} ({self.formulas[i]}) has no atom to normalise by (normalize={normalize!r})")
        self.normalization = np.asarray(N, dtype=float)
        self.coefficients = {name: np.array([float(x) for x in X[:, j]]) for j, name in enumerate(names)}
        # coefficients of G_i, μ_r and μ_k in ΔG_i, exact rationals rounded once
        n_ref = len(self.references)
        self._p = np.array([float(Fraction(1, int(m))) for m in N])
        self._c = np.array([[float(-x / int(m)) for x in row] for row, m in zip(X, N)]).reshape(n, len(names))
        self._n_ref = n_ref
        if isinstance(degeneracy, str):
            if degeneracy != "chirality":
                raise ValueError("degeneracy must be 'chirality', None or an array")
            degeneracy = [chirality_degeneracy(a) for a in self.structures]
        self.degeneracy = np.ones(n) if degeneracy is None else np.asarray(degeneracy, dtype=float)

    def count(self, element: str) -> np.ndarray:
        """Atoms of ``element`` in every structure."""
        if element not in self.elements:
            return np.zeros(len(self.structures), dtype=int)
        return self.counts[:, self.elements.index(element)]

    def evaluate(self, T, p: dict | None = None, dmu: dict | None = None, *, axes=None,
                 dtype=np.float32) -> "EnsembleResult":
        """ΔG and populations on the grid of conditions.

        ``T`` in K; for every reservoir either ``p[name]`` (Pa) or ``dmu[name]`` (eV, relative to its DFT energy).
        Scalars are fixed, 1D arrays become axes (order ``axes``; default the reservoirs, then T). ΔG is stored in
        ``dtype``.
        """
        grid = build_grid(T, p, dmu, reservoir_names=[r.name for r in self.reservoirs], axes=axes)
        mus = [r.chemical_potential(grid.T, grid.full[r.name]) if grid.modes[r.name] == "p"
               else r.energy + grid.full[r.name] for r in self.reservoirs]
        G = np.empty((len(self.structures),) + grid.shape, dtype=dtype)
        for i, atoms in enumerate(self.structures):
            g = self._p[i] * grid.along_T(free_energy(atoms, grid.T_values, terms=self.terms[i]))
            for r, ref in enumerate(self.references):
                g = g + self._c[i, r] * ref.energy_per_atom
            for k, mu in enumerate(mus):
                g = g + self._c[i, self._n_ref + k] * mu
            G[i] = g
        populations = boltzmann_populations(G, grid.T, self.degeneracy).astype(np.float32)
        return EnsembleResult(self, grid, G, populations, np.argmin(G, axis=0))

    def variant(self, **changes) -> "GrandCanonicalEnsemble":
        """The same ensemble with terms switched on every structure and reservoir (e.g. no vibrations)."""
        other = object.__new__(GrandCanonicalEnsemble)
        other.__dict__.update(self.__dict__)
        other.terms = [t.replace(**changes) for t in self.terms]
        other.reservoirs = [r.with_terms(**changes) for r in self.reservoirs]
        return other

    def with_structures(self, structures) -> "GrandCanonicalEnsemble":
        """The same rows with replaced data (e.g. frequencies from another method); formulas must match."""
        structures = list(structures)
        if [a.get_chemical_formula() for a in structures] != list(self.formulas):
            raise ValueError("with_structures needs the same rows, in the same order, with the same formulas")
        other = object.__new__(GrandCanonicalEnsemble)
        other.__dict__.update(self.__dict__)
        other.structures = structures
        return other


class EnsembleResult:
    """ΔG (rows × grid), populations and lowest-ΔG index of an ensemble on a grid of conditions."""

    def __init__(self, ensemble: GrandCanonicalEnsemble, grid: Grid, G, populations, phase_index, rows=None):
        self.ensemble, self.grid, self.G = ensemble, grid, G
        self.populations, self.phase_index = populations, phase_index
        self.rows = np.arange(len(ensemble.structures)) if rows is None else np.asarray(rows, dtype=int)

    @property
    def axes(self) -> dict:
        return dict(zip(self.grid.names, self.grid.values))

    @property
    def formulas(self) -> np.ndarray:
        return self.ensemble.formulas[self.rows]

    @property
    def degeneracy(self) -> np.ndarray:
        return self.ensemble.degeneracy[self.rows]

    def count(self, element: str) -> np.ndarray:
        return self.ensemble.count(element)[self.rows]

    def index(self, **coordinates) -> tuple:
        """Nearest grid index, e.g. ``index(T=500, O2=2.1e4)`` (log scale for pressures)."""
        return self.grid.index(**coordinates)

    def _per_row(self, values) -> np.ndarray:
        values = np.asarray(values)
        if values.shape[:1] != self.rows.shape:
            raise ValueError(f"Expected one value per row ({len(self.rows)}), got shape {values.shape}")
        return values.reshape((-1,) + (1,) * len(self.grid.shape)) if values.ndim == 1 else values

    def shifted(self, delta, *, rows=None) -> "EnsembleResult":
        """Re-weighted after adding ``delta`` to ΔG: one value per row, per row and grid point, or, with ``rows``,
        only for those rows. The shift is applied in float64 to the stored ΔG; ΔG is stored again in its dtype."""
        g = np.asarray(self.G, dtype=float)
        if rows is None:
            g = g + self._per_row(delta)
        else:
            for row, d in zip(np.searchsorted(self.rows, np.asarray(rows, dtype=int)), delta):
                g[row] += d
        populations = boltzmann_populations(g, self.grid.T, self.degeneracy).astype(np.float32)
        return EnsembleResult(self.ensemble, self.grid, g.astype(self.G.dtype), populations,
                              np.argmin(g, axis=0), rows=self.rows)

    def subset(self, mask=None, *, counts: dict | None = None) -> "EnsembleResult":
        """Rows selected by ``mask`` and/or exact element ``counts`` (e.g. ``{"Ir": 6}``), populations renormalised."""
        keep = np.ones(len(self.rows), dtype=bool) if mask is None else np.asarray(mask, dtype=bool).copy()
        for element, n in (counts or {}).items():
            keep &= self.count(element) == n
        if not np.any(keep):
            raise ValueError("The subset is empty")
        P = np.asarray(self.populations[keep], dtype=np.float32)
        total = P.sum(axis=0, keepdims=True)
        np.divide(P, total, out=P, where=total > 0)
        G = self.G[keep]
        return EnsembleResult(self.ensemble, self.grid, G, P, np.argmin(G, axis=0), rows=self.rows[keep])

    def composition(self, element: str) -> np.ndarray:
        """⟨n_element / N⟩: atoms of ``element`` per normalisation unit (per metal atom with the default)."""
        return self.average(self.count(element) / self.ensemble.normalization[self.rows])

    def average(self, values) -> np.ndarray:
        """Population-weighted mean of a per-row property (shape ``(rows,)`` or ``(rows, *grid)``)."""
        return np.sum(self.populations * self._per_row(np.asarray(values, dtype=float)), axis=0)

    def most_stable(self) -> np.ndarray:
        """Index (into ``rows``) of the lowest-ΔG structure at every grid point."""
        return self.phase_index

    def contributions(self, names=None) -> dict:
        """Each row's ΔG split into parts on the grid: ``dft`` (bare energies), ``electronic``, ``vibrational``
        (with ZPE), ``rotational`` and ``translational``; arrays of shape ``(rows, *grid)`` that add up to ΔG.

        Each part collects the structure's term and the matching terms of the references and reservoirs, with
        the balance coefficients. ``names`` restricts the output to some parts (each costs one rows × grid
        array). Needs every reservoir given by pressure.
        """
        names = list(names or _PARTS)
        out = {name: np.empty((len(self.rows),) + self.grid.shape) for name in names}
        for j, parts in enumerate(self._row_parts()):
            for name in names:
                out[name][j] = parts[name]
        return out

    def decomposition(self) -> dict:
        """Ensemble averages of the parts of ΔG (``contributions``), keys ``E_dft_avg``, ``E_elec_avg``,
        ``E_vib_avg``, ``E_rot_avg`` and ``E_trans_avg``."""
        out = {key: np.zeros(self.grid.shape) for key in _PARTS.values()}
        for j, parts in enumerate(self._row_parts()):
            w = self.populations[j].astype(float)
            for name, key in _PARTS.items():
                out[key] += w * parts[name]
        return out

    def _row_parts(self):
        """Yield, row by row, the parts of ΔG on the grid (one implementation for both methods above)."""
        ens, grid = self.ensemble, self.grid
        if any(grid.modes[r.name] != "p" for r in ens.reservoirs):
            raise ValueError("contributions need every reservoir given by pressure")
        Tg, T1 = grid.T, grid.T_values
        res = [_parts(r.atoms, Tg, grid.full[r.name], r.terms, r.multiplicity, None, r.symmetry_number)
               for r in ens.reservoirs]
        k0 = ens._n_ref

        def g(c, e, s, T):
            return c[e] - T * c[s]

        for i in self.rows:
            c = _parts(ens.structures[i], T1, None, ens.terms[i], None, None, None)
            parts = {}
            dft = ens._p[i] * c["energy"]
            for r, ref in enumerate(ens.references):
                dft = dft + ens._c[i, r] * ref.energy_per_atom
            for k, rc in enumerate(res):
                dft = dft + ens._c[i, k0 + k] * rc["energy"]
            parts["dft"] = np.broadcast_to(dft, grid.shape)
            for name, (e, s) in (("electronic", ("eelec", "selec")), ("rotational", ("er", "sr")),
                                 ("translational", ("et", "st"))):
                term = ens._p[i] * grid.along_T(g(c, e, s, T1))
                for k, rc in enumerate(res):
                    term = term + ens._c[i, k0 + k] * g(rc, e, s, Tg)
                parts[name] = term
            vib = ens._p[i] * grid.along_T(c["ev"] + c["zpe"] - T1 * c["sv"])
            for k, rc in enumerate(res):
                vib = vib + ens._c[i, k0 + k] * (rc["ev"] + rc["zpe"] - Tg * rc["sv"])
            parts["vibrational"] = vib
            yield parts


_PARTS = {"dft": "E_dft_avg", "electronic": "E_elec_avg", "vibrational": "E_vib_avg", "rotational": "E_rot_avg",
          "translational": "E_trans_avg"}
