"""Grand-canonical ensembles of structures in contact with reservoir molecules.

    ΔG_i = (1/N_i) [ G_i(T) − Σ_r a_ir μ_r − Σ_k ν_ik μ_k(T, p_k) ],    P_i = g_i exp(−ΔG_i / kT) / Ξ

``a_ir`` and ``ν_ik`` balance structure *i* exactly against the references *r* (fixed μ per atom) and the
reservoir molecules *k*; ``N_i`` is the number of reference atoms (``normalize="reference"``, the IryOx papers'
per-metal-atom free energy), 1 (``"cluster"``) or the count of a chosen element; ``g_i`` is the chirality
degeneracy. Any array-valued condition (T, a pressure or a Δμ) is a grid axis.

The tensors are stored in float32 by default and populations are computed with exactly the arithmetic of the
0.1 code (``compute_gc``, ``reensemble_from_g``), so the Paper I ensembles are reproduced bit for bit.
"""
from __future__ import annotations

import warnings
from fractions import Fraction

import numpy as np
from scipy.constants import Boltzmann, electron_volt

from aist.balance import balance_coefficients
from aist.composedthermo import compute_thermo_contributions
from aist.conditions import Grid, build_grid
from aist.core.ensemble import renormalize_populations
from aist.species import GasSpecies, Reference, Reservoir, SpeciesThermo, ThermoTerms, spin_multiplicity
from aist.thermo import POINT_GROUP_DEGEN

KB_EV = Boltzmann / electron_volt

__all__ = ["GrandCanonicalEnsemble", "EnsembleResult", "SupportShift", "chirality_degeneracy"]


def chirality_degeneracy(atoms) -> int:
    """2 for a chiral point group (both enantiomers are ensemble members), else 1."""
    from pymatgen.core.structure import Molecule
    from pymatgen.symmetry.analyzer import PointGroupAnalyzer

    mol = Molecule(species=atoms.get_chemical_symbols(), coords=atoms.get_positions())
    return POINT_GROUP_DEGEN.get(PointGroupAnalyzer(mol).sch_symbol, 1)


def _gas_delta(atoms, T, p_gas, elec, *, include_v, include_zpe, rotation=False, translation=False,
               vibrations=False) -> np.ndarray:
    """Translation and/or rotation at ``p_gas``, or (``vibrations``) vibrations + ZPE, of one gas molecule (float32)."""
    pgas = np.full_like(T, p_gas)
    if not vibrations:
        on = dict(include_v=include_v, include_zpe=include_zpe, include_r=rotation, include_t=translation)
        off = dict(include_v=include_v, include_zpe=include_zpe, include_r=False, include_t=False)
    else:
        on = dict(include_v=True, include_zpe=True, include_r=False, include_t=False)
        off = dict(include_v=False, include_zpe=False, include_r=False, include_t=False)
    g_on, _ = compute_thermo_contributions(atoms, T, pgas, elec_states=elec, **on)
    g_off, _ = compute_thermo_contributions(atoms, T, pgas, elec_states=elec, **off)
    return np.asarray(g_on - g_off, dtype=np.float32)


class SupportShift:
    """Flat adhesion proxy for clusters standing in for supported ones (gas-phase species untouched).

    Every cluster is stabilised by ``ev_per_reference`` per reference atom plus ``ev_per_element[e]`` per atom of
    element ``e``; on the per-reference-atom free energy that is ``−(a + Σ_e b_e n_e / N)``.
    """

    def __init__(self, ev_per_reference: float, ev_per_element: dict | None = None):
        self.ev_per_reference = ev_per_reference
        self.ev_per_element = dict(ev_per_element or {})

    def delta(self, ensemble: "GrandCanonicalEnsemble") -> np.ndarray:
        n_ref = ensemble.reference_counts.astype(float)
        N = ensemble.normalization
        corr = np.zeros(len(ensemble.structures))
        for i in range(len(corr)):
            if ensemble.is_gas[i]:
                continue
            if ensemble.normalize == "reference":
                c = self.ev_per_reference
                for e, b in self.ev_per_element.items():
                    c = c + b * (float(ensemble.count(e)[i]) / N[i])
            else:
                c = self.ev_per_reference * n_ref[i]
                for e, b in self.ev_per_element.items():
                    c = c + b * float(ensemble.count(e)[i])
                c = c / N[i]
            corr[i] = c
        return -corr


class GrandCanonicalEnsemble:
    """A pool of structures (lists of ASE ``Atoms``) exchanging atoms with reservoir molecules.

    Parameters
    ----------
    structures : Atoms objects with an energy (calculator or ``info['energy']``) and, as needed,
        ``info['frequencies']`` (cm⁻¹, all 3N modes), ``info['zpe']`` (eV) and ``info['total_spin']['J']``.
    references : ``Reference`` objects, one per element held at a fixed chemical potential (e.g. the metal atom).
    reservoirs : ``Reservoir`` molecules; together with the references they must balance every structure uniquely.
    terms : ``ThermoTerms`` for the structures. Default: the electronic contribution only (energy and spin
        degeneracy); add vibrations, ZPE, rotation or translation by switching them on.
    normalize : ``"reference"`` (default: per reference atom), ``"cluster"``, or an element symbol.
    degeneracy : ``"chirality"`` (point-group based), ``None`` (all 1) or an array.
    gas : ``GasSpecies`` appended to the pool (volatile products at a fixed partial pressure).
    """

    def __init__(self, structures, *, references, reservoirs, terms: ThermoTerms | None = None,
                 normalize: str = "reference", degeneracy="chirality", gas=()):
        structures = list(structures)
        self.gas = [g if isinstance(g, GasSpecies) else GasSpecies(*g) for g in gas]
        self.structures = structures + [g.atoms for g in self.gas]
        self.is_gas = np.zeros(len(self.structures), dtype=bool)
        self.is_gas[len(structures):] = True
        self.references = list(references)
        self.reservoirs = list(reservoirs)
        if not self.references:
            raise ValueError("Give at least one Reference (e.g. the isolated metal atom)")
        names = [r.name for r in self.references] + [r.name for r in self.reservoirs]
        if len(set(names)) != len(names):
            raise ValueError(f"Reference and reservoir names must be unique, got {names}")
        self.terms = terms or ThermoTerms()
        self.normalize = normalize
        species = [(r.name, r.composition) for r in self.references + self.reservoirs]
        self.elements, self.counts, X = balance_coefficients(self.structures, species)
        self.formulas = np.array([a.get_chemical_formula() for a in self.structures], dtype=object)
        n_ref = len(self.references)
        self.reference_counts = np.array(
            [sum(int(self.count(r.element)[i]) for r in self.references) for i in range(len(self.structures))]
        )
        N = self._normalization_counts()
        self.normalization = N.astype(float)
        self.coefficients = {name: np.array([float(x) for x in X[:, j]]) for j, name in enumerate(names)}
        # exact coefficients of G_i, μ_r and μ_k in ΔG_i, rounded once
        self._p = np.array([float(Fraction(1, int(n))) for n in N])
        self._c_ref = np.array([[float(-X[i, r] / int(N[i])) for r in range(n_ref)] for i in range(len(N))])
        self._c_res = np.array(
            [[float(-X[i, n_ref + k] / int(N[i])) for k in range(len(self.reservoirs))] for i in range(len(N))]
        ).reshape(len(N), len(self.reservoirs))
        self.degeneracy = self._degeneracy(degeneracy)
        self._thermo: list = [None] * len(self.structures)
        if self.terms.electronic:
            missing = [i for i in range(len(structures)) if spin_multiplicity(self.structures[i]) is None]
            if missing:
                warnings.warn(
                    f"{len(missing)} structure(s) have no info['total_spin']['J'] and are treated as singlets "
                    f"(first: row {missing[0]}, {self.formulas[missing[0]]})",
                    stacklevel=2,
                )

    # ------------------------------------------------------------------ composition
    def count(self, element: str) -> np.ndarray:
        """Number of atoms of ``element`` in every structure (0 where absent)."""
        if element not in self.elements:
            return np.zeros(len(self.structures), dtype=int)
        return self.counts[:, self.elements.index(element)]

    def _normalization_counts(self) -> np.ndarray:
        if self.normalize == "reference":
            N = self.reference_counts
        elif self.normalize == "cluster":
            N = np.ones(len(self.structures), dtype=int)
        elif isinstance(self.normalize, str):
            N = self.count(self.normalize)
        else:
            raise ValueError("normalize must be 'reference', 'cluster' or an element symbol")
        if np.any(N <= 0):
            i = int(np.nonzero(N <= 0)[0][0])
            raise ValueError(f"Row {i} ({self.structures[i].get_chemical_formula()}) has no atom to normalise by "
                             f"(normalize={self.normalize!r})")
        return np.asarray(N, dtype=int)

    def _degeneracy(self, degeneracy) -> np.ndarray:
        n = len(self.structures)
        if degeneracy is None:
            return np.ones(n)
        if isinstance(degeneracy, str):
            if degeneracy != "chirality":
                raise ValueError("degeneracy must be 'chirality', None or an array")
            return np.array([chirality_degeneracy(a) for a in self.structures], dtype=float)
        deg = np.asarray(degeneracy, dtype=float)
        if deg.shape != (n,):
            raise ValueError(f"degeneracy array must have shape ({n},)")
        return deg

    # ------------------------------------------------------------------ thermodynamics
    def thermo(self, i: int) -> SpeciesThermo:
        """Cached free-energy terms of row ``i``."""
        if self._thermo[i] is None:
            atoms = self.structures[i]
            n_clusters = len(self.structures) - len(self.gas)
            J = self.gas[i - n_clusters].multiplicity if self.is_gas[i] else None
            if J is None:
                J = spin_multiplicity(atoms)
            elec = [(0.0, J)] if J is not None else None
            self._thermo[i] = SpeciesThermo(atoms, electronic_states=elec)
        return self._thermo[i]

    def _electronic_states(self, i: int):
        return self.thermo(i).electronic_states

    def chemical_potentials(self, grid: Grid) -> list[np.ndarray]:
        """μ_k of every reservoir on the full grid (eV)."""
        mus = []
        for res in self.reservoirs:
            if grid.modes[res.name] == "p":
                mus.append(res.chemical_potential(grid.T, grid.full[res.name]))
            else:
                mus.append(res.energy + grid.full[res.name])
        return mus

    def formation_free_energy(self, i: int, grid: Grid, mus) -> np.ndarray:
        """ΔG_i on the grid (float64)."""
        Ep = grid.along_T(self.thermo(i).free_energy(grid.T_values, self.terms))
        g = self._p[i] * Ep
        for r, ref in enumerate(self.references):
            g = g + self._c_ref[i, r] * ref.energy_per_atom
        for k, mu in enumerate(mus):
            g = g + self._c_res[i, k] * mu
        return g

    def evaluate(self, T, p: dict | None = None, dmu: dict | None = None, *, axes=None,
                 dtype=np.float32) -> "EnsembleResult":
        """Free energies and populations on the conditions grid.

        ``T`` in K; ``p[name]`` in Pa or ``dmu[name]`` in eV (relative to the molecule's DFT energy) for every
        reservoir; scalars are fixed, 1D arrays become axes (order: ``axes``, default reservoirs then T).
        """
        grid = build_grid(T, p, dmu, reservoir_names=[r.name for r in self.reservoirs], axes=axes)
        mus = self.chemical_potentials(grid)
        G = np.empty((len(self.structures),) + grid.shape, dtype=dtype)
        for i in range(len(self.structures)):
            G[i] = self.formation_free_energy(i, grid, mus)
        result = EnsembleResult._from_free_energies(self, grid, G)
        if self.gas:
            result = self._with_gas_terms(result)
        return result

    def _with_gas_terms(self, result: "EnsembleResult") -> "EnsembleResult":
        """Gas species: strip vibrations if asked, then add translation + rotation at their partial pressure."""
        rows = np.nonzero(self.is_gas)[0]
        T = result.grid.T
        strip = [(row, gs) for row, gs in zip(rows, self.gas) if not gs.vibrations]
        if strip:
            deltas = np.stack([
                -(_gas_delta(gs.atoms, T, gs.pressure, self._electronic_states(row), include_v=True,
                             include_zpe=True, vibrations=True) * self._p[row])
                for row, gs in strip
            ])
            result = result.shifted(deltas, rows=[row for row, _ in strip])
        moving = [(row, gs) for row, gs in zip(rows, self.gas) if gs.rotation or gs.translation]
        if not moving:
            return result
        deltas = np.stack([
            _gas_delta(gs.atoms, T, gs.pressure, self._electronic_states(row), include_v=self.terms.vibrations,
                       include_zpe=self.terms.zpe, rotation=gs.rotation, translation=gs.translation) * self._p[row]
            for row, gs in moving
        ])
        return result.shifted(deltas, rows=[row for row, _ in moving])

    # ------------------------------------------------------------------ variants
    def variant(self, **changes) -> "GrandCanonicalEnsemble":
        """The same ensemble with terms switched on structures and reservoirs alike (full / novib / ZPE only).

        Cached partition functions are shared, so a variant costs only the reassembly of G.
        """
        other = object.__new__(GrandCanonicalEnsemble)
        other.__dict__.update(self.__dict__)
        other.terms = self.terms.replace(**changes)
        other.reservoirs = [r.with_terms(**changes) for r in self.reservoirs]
        return other

    def with_structures(self, structures) -> "GrandCanonicalEnsemble":
        """Same pool with replaced data (e.g. frequencies or energies from another method), same rows and formulas.

        Compositions, balance and degeneracies are kept; free-energy terms are recomputed from the new objects.
        """
        structures = list(structures)
        n_clusters = len(self.structures) - len(self.gas)
        if len(structures) != n_clusters:
            raise ValueError(f"Expected {n_clusters} structures, got {len(structures)}")
        formulas = [a.get_chemical_formula() for a in structures]
        bad = [i for i, f in enumerate(formulas) if f != self.formulas[i]]
        if bad:
            raise ValueError(f"Row {bad[0]} changed formula ({self.formulas[bad[0]]} -> {formulas[bad[0]]})")
        other = object.__new__(GrandCanonicalEnsemble)
        other.__dict__.update(self.__dict__)
        other.structures = structures + [g.atoms for g in self.gas]
        other._thermo = [None] * len(other.structures)
        return other

    # ------------------------------------------------------------------ ASE-style conveniences
    def get_free_energies(self, T, p=None, dmu=None, **kwargs) -> np.ndarray:
        """ΔG of every structure on the conditions (eV per normalisation unit)."""
        return self.evaluate(T, p, dmu, **kwargs).G

    def get_populations(self, T, p=None, dmu=None, **kwargs) -> np.ndarray:
        """Boltzmann populations of every structure on the conditions."""
        return self.evaluate(T, p, dmu, **kwargs).populations

    def get_composition(self, element: str, T, p=None, dmu=None, **kwargs) -> np.ndarray:
        """Ensemble ⟨n_element / N⟩ on the conditions."""
        return self.evaluate(T, p, dmu, **kwargs).composition(element)

    def get_average(self, values, T, p=None, dmu=None, **kwargs) -> np.ndarray:
        """Population-weighted mean of a per-structure property on the conditions."""
        return self.evaluate(T, p, dmu, **kwargs).average(values)


class EnsembleResult:
    """Free energies ``G`` (rows × grid), populations and lowest-G index of an ensemble on a grid."""

    def __init__(self, ensemble: GrandCanonicalEnsemble, grid: Grid, G, populations, phase_index, rows=None):
        self.ensemble = ensemble
        self.grid = grid
        self.G = G
        self.populations = populations
        self.phase_index = phase_index
        self.rows = np.arange(len(ensemble.structures)) if rows is None else np.asarray(rows, dtype=int)

    @classmethod
    def _from_free_energies(cls, ensemble, grid, G):
        """Populations as ``compute_gc`` computes them (from the stored tensor)."""
        beta = 1.0 / (KB_EV * grid.T)
        G_min = np.min(G, axis=0, keepdims=True)
        weights = np.exp(-beta * (G - G_min))
        weights *= ensemble.degeneracy.reshape((-1,) + (1,) * len(grid.shape))
        Xi = np.sum(weights, axis=0)
        populations = (weights / Xi).astype(np.float32)
        return cls(ensemble, grid, G, populations, np.argmin(G, axis=0))

    # ------------------------------------------------------------------ views
    @property
    def axes(self) -> dict:
        return dict(zip(self.grid.names, self.grid.values))

    @property
    def shape(self) -> tuple:
        return self.grid.shape

    @property
    def degeneracy(self) -> np.ndarray:
        return self.ensemble.degeneracy[self.rows]

    @property
    def formulas(self) -> np.ndarray:
        return self.ensemble.formulas[self.rows]

    def count(self, element: str) -> np.ndarray:
        return self.ensemble.count(element)[self.rows]

    def _per_row(self, values) -> np.ndarray:
        values = np.asarray(values)
        if values.shape[0] != len(self.rows):
            raise ValueError(f"Expected one value per row ({len(self.rows)}), got {values.shape[0]}")
        if values.ndim == 1:
            return values.reshape((-1,) + (1,) * len(self.grid.shape))
        return values

    def index(self, **coordinates) -> tuple:
        """Nearest grid index, e.g. ``res.index(T=500, O2=2.1e4)``."""
        return self.grid.index(**coordinates)

    # ------------------------------------------------------------------ re-weighting
    def shifted(self, delta, *, rows=None) -> "EnsembleResult":
        """Re-ensembled after adding ``delta`` to ΔG (per row, per row and grid point, or a ``SupportShift``).

        As in ``reensemble_from_g``: the shift is applied in float64 to the stored tensor, populations come from
        that float64 tensor and the stored tensor is rounded again to its dtype.
        """
        g = np.asarray(self.G, dtype=float)
        if hasattr(delta, "delta"):
            delta = delta.delta(self.ensemble)[self.rows]
        if rows is None:
            g = g + self._per_row(delta)
        else:
            local = np.searchsorted(self.rows, np.asarray(rows, dtype=int))
            for j, row in enumerate(local):
                g[row] += delta[j]
        beta = 1.0 / (KB_EV * self.grid.T)
        weights = np.exp(-beta * (g - np.min(g, axis=0, keepdims=True)))
        weights *= self.degeneracy.reshape((-1,) + (1,) * len(self.grid.shape))
        xi = np.sum(weights, axis=0)
        populations = (weights / xi).astype(np.float32)
        return EnsembleResult(self.ensemble, self.grid, g.astype(self.G.dtype), populations,
                              np.argmin(g, axis=0), rows=self.rows)

    def subset(self, mask=None, *, counts: dict | None = None, gas: bool = False) -> "EnsembleResult":
        """Rows selected by ``mask`` and/or exact element ``counts`` (e.g. ``{"Ir": 6}``), populations renormalised.

        ``gas=True`` keeps the gas-species rows in every subset.
        """
        keep = np.ones(len(self.rows), dtype=bool) if mask is None else np.asarray(mask, dtype=bool).copy()
        if counts:
            for element, n in counts.items():
                keep &= self.count(element) == n
        if gas:
            keep |= self.ensemble.is_gas[self.rows]
        if not np.any(keep):
            raise ValueError("The subset is empty")
        populations = renormalize_populations(self.populations, keep)
        G = self.G[keep]
        return EnsembleResult(self.ensemble, self.grid, G, populations, np.argmin(G, axis=0), rows=self.rows[keep])

    # ------------------------------------------------------------------ observables
    def composition(self, element: str) -> np.ndarray:
        """⟨n_element / N⟩: atoms of ``element`` per normalisation unit (per metal atom in the papers)."""
        ratio = self.count(element) / self.ensemble.normalization[self.rows]
        return np.sum(self.populations * self._per_row(ratio), axis=0)

    def average(self, values) -> np.ndarray:
        """Population-weighted mean of a per-row property (shape ``(rows,)`` or ``(rows, *grid)``)."""
        return np.sum(self.populations * self._per_row(np.asarray(values, dtype=float)), axis=0)

    def most_stable(self) -> np.ndarray:
        """Index (into ``rows``) of the lowest-ΔG structure at every grid point."""
        return self.phase_index

    def most_stable_population(self) -> np.ndarray:
        """Population of the lowest-ΔG structure at every grid point."""
        return np.take_along_axis(self.populations, self.phase_index[None], axis=0)[0]

    def configurational_entropy(self) -> np.ndarray:
        """−k Σ P ln P (eV/K)."""
        return -KB_EV * np.sum(self.populations * np.log(self.populations + 1e-20), axis=0)

    def mean_free_energy(self) -> np.ndarray:
        """Population-weighted ΔG."""
        return np.sum(self.populations * self.G, axis=0)

    def decomposition(self) -> dict:
        """Ensemble-averaged parts of ΔG: bare DFT, electronic, vibrational (with ZPE), rotational, translational.

        Uses the current populations and needs every reservoir given by pressure.
        """
        ens, grid = self.ensemble, self.grid
        if any(grid.modes[r.name] != "p" for r in ens.reservoirs):
            raise ValueError("decomposition() needs every reservoir given by pressure")
        Tg = grid.T
        res_c = [r.thermo.contributions(Tg, r.terms, grid.full[r.name]) for r in ens.reservoirs]
        parts = {key: np.zeros(grid.shape) for key in ("E_dft_avg", "E_elec_avg", "E_vib_avg", "E_rot_avg",
                                                         "E_trans_avg")}

        def g_from(c, e, s, T):
            return c.get(e, 0.0) - T * c.get(s, 0.0)

        for j, i in enumerate(self.rows):
            w = self.populations[j].astype(float)
            c = ens.thermo(i).contributions(grid.T_values, ens.terms)
            T1 = grid.T_values
            dft = ens._p[i] * ens.thermo(i).energy
            for r, ref in enumerate(ens.references):
                dft = dft + ens._c_ref[i, r] * ref.energy_per_atom
            for k, rc in enumerate(res_c):
                dft = dft + ens._c_res[i, k] * ens.reservoirs[k].energy
            parts["E_dft_avg"] += w * dft
            for key, (e, s) in (("E_elec_avg", ("eelec", "selec")), ("E_rot_avg", ("er", "sr")),
                                ("E_trans_avg", ("et", "st"))):
                term = ens._p[i] * grid.along_T(g_from(c, e, s, T1))
                for k, rc in enumerate(res_c):
                    term = term + ens._c_res[i, k] * g_from(rc, e, s, Tg)
                parts[key] += w * term
            vib = ens._p[i] * grid.along_T(c.get("ev", 0.0) + c.get("zpe", 0.0) - T1 * c.get("sv", 0.0))
            for k, rc in enumerate(res_c):
                vib = vib + ens._c_res[i, k] * (rc.get("ev", 0.0) + rc.get("zpe", 0.0) - Tg * rc.get("sv", 0.0))
            parts["E_vib_avg"] += w * vib
        return parts
