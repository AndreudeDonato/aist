"""Species of a grand-canonical ensemble: term flags, references, reservoir molecules and gas-phase products.

Every free energy here is built from the partition functions of ``aist.thermo`` with the same arithmetic as
``aist.composedthermo.compute_thermo_contributions``, so results are bit-identical to the 0.1 code. The terms of
one ``Atoms`` object are computed lazily and cached per temperature array, which keeps variants (vibrations off,
ZPE only, a replaced reservoir) cheap.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, replace

import numpy as np
from ase import Atoms

from aist.composedthermo import process_frequencies
from aist.thermo import ThermoFromAtoms
from aist.utils import get_energy


@dataclass(frozen=True)
class ThermoTerms:
    """Which free-energy terms enter G, chosen by the user; none is right or wrong.

    By default only the electronic contribution enters: the energy E plus the electronic (spin) degeneracy
    −kT ln(2S+1). Each other term is added by switching it on: ``vibrations`` (harmonic U − TS), ``zpe``,
    ``rotation`` (rigid rotor) and ``translation`` (ideal gas at ``pressure``, in Pa, which then has to be given).
    Reservoir molecules default to ``ThermoTerms.gas()``, every term on, which is their ideal-gas chemical
    potential.
    """

    vibrations: bool = False
    zpe: bool = False
    electronic: bool = True
    rotation: bool = False
    translation: bool = False
    pressure: float | None = None  # Pa; the structures' partial pressure, needed only when ``translation`` is on

    @classmethod
    def gas(cls, **overrides) -> "ThermoTerms":
        """All terms on: the ideal-gas chemical potential of a molecule."""
        flags = dict(vibrations=True, zpe=True, electronic=True, rotation=True, translation=True)
        flags.update(overrides)
        return cls(**flags)

    def replace(self, **changes) -> "ThermoTerms":
        return replace(self, **changes)


def spin_multiplicity(atoms: Atoms) -> float | None:
    """``2S+1`` from ``atoms.info['total_spin']['J']`` (a dict or an ase ``_JSON`` string), else ``None``."""
    spin = atoms.info.get("total_spin")
    if isinstance(spin, str):
        spin = json.loads(spin.replace("_JSON ", "", 1).strip())
    if isinstance(spin, dict) and spin.get("J") is not None:
        return spin["J"]
    return None


class SpeciesThermo:
    """Lazily computed thermodynamic terms of one ``Atoms`` object on temperature (and pressure) arrays.

    ``electronic_states`` is a list of ``(energy_eV, degeneracy)``; ``None`` leaves the electronic term out.
    """

    def __init__(self, atoms: Atoms, *, electronic_states=None, symmetry_number=None):
        self.atoms = atoms
        self.thermo = ThermoFromAtoms(atoms)
        self.energy = get_energy(atoms)
        zpe = atoms.info.get("zpe", 0)
        self.zpe = 0 if zpe is None else zpe
        self.frequencies = process_frequencies(self.thermo, atoms.info.get("frequencies"))
        self.electronic_states = None if electronic_states is None else list(electronic_states)
        self.symmetry_number = symmetry_number
        self._cache: dict = {}

    @staticmethod
    def _key(name, *arrays):
        return (name,) + tuple((np.asarray(a, dtype=float).tobytes(), np.shape(a)) for a in arrays)

    def _cached(self, key, compute):
        if key not in self._cache:
            self._cache[key] = compute()
        return self._cache[key]

    def vibrational(self, T):
        if len(self.frequencies) == 0:
            return 0, 0
        return self._cached(self._key("vib", T), lambda: self.thermo.get_vibrational(self.frequencies, T)[:2])

    def rotational(self, T):
        if len(self.atoms) < 2:
            return 0, 0
        return self._cached(
            self._key("rot", T), lambda: self.thermo.get_rotational(T, sigma=self.symmetry_number)[:2]
        )

    def translational(self, T, p):
        def compute():
            et, st, _ = self.thermo.get_translational(T, p)
            # ideal-gas enthalpy: H_trans = U_trans + pV = 3/2 kT + kT, so G = H - TS = -kT ln q
            et = et + self.thermo.tp.kb * self.thermo.tp.J_to_eV * np.asarray(T, dtype=float)
            return et, st

        return self._cached(self._key("trans", T, p), compute)

    def electronic(self, T):
        if self.electronic_states is None:
            return 0, 0
        return self._cached(
            self._key("elec", T), lambda: self.thermo.get_electronic(self.electronic_states, T)[:2]
        )

    def contributions(self, T, terms: ThermoTerms, p=None) -> dict:
        """The terms that ``terms`` switches on, as in ``compute_thermo_contributions`` (absent terms are 0)."""
        if p is None:
            p = terms.pressure
        if terms.translation and p is None:
            raise ValueError("The translational term needs a pressure: ThermoTerms(translation=True, pressure=...) (Pa)")
        et, st = self.translational(T, p) if terms.translation else (0, 0)
        ev, sv = self.vibrational(T) if terms.vibrations else (0, 0)
        er, sr = self.rotational(T) if terms.rotation else (0, 0)
        eelec, selec = self.electronic(T) if terms.electronic else (0, 0)
        return {
            "et": et, "ev": ev, "er": er, "eelec": eelec,
            "st": st, "sv": sv, "sr": sr, "selec": selec,
            "zpe": self.zpe if terms.zpe else 0,
        }

    def free_energy(self, T, terms: ThermoTerms, p=None):
        """``E + ZPE + U(T) - T S`` with the terms that ``terms`` switches on (eV)."""
        c = self.contributions(T, terms, p)
        total_energy = self.energy + c["zpe"] + c["et"] + c["ev"] + c["er"] + c["eelec"]
        total_entropy = c["st"] + c["sv"] + c["sr"] + c["selec"]
        return total_energy - T * total_entropy


class Reference:
    """Fixed chemical potential per atom of one element (e.g. the isolated metal atom, or a bulk per atom).

    Give an elemental ``Atoms`` object (μ = E / number of atoms) or ``energy=`` (eV per atom) with ``element=``.
    """

    def __init__(self, atoms: Atoms | None = None, *, energy: float | None = None, element: str | None = None):
        if atoms is not None:
            symbols = set(atoms.get_chemical_symbols())
            if len(symbols) != 1:
                raise ValueError(
                    f"A reference must contain one element, got {atoms.get_chemical_formula()}; "
                    "use Reference(energy=..., element=...) for anything else"
                )
            found = symbols.pop()
            if element is not None and element != found:
                raise ValueError(f"Reference element {element!r} does not match the atoms ({found})")
            element = found
            energy = get_energy(atoms) / len(atoms)
        if energy is None or element is None:
            raise ValueError("Reference needs an elemental Atoms object, or both energy= and element=")
        self.element = element
        self.energy_per_atom = energy
        self.composition = {element: 1}
        self.name = element

    def __repr__(self) -> str:
        return f"Reference({self.element}, {self.energy_per_atom:.6f} eV/atom)"


class Reservoir:
    """A gas molecule that exchanges atoms with the structures; μ(T, p) from its ideal-gas partition function.

    Every term is a choice (``terms``; default all on: translation, rotation, vibrations, ZPE, electronic).
    The electronic term needs the spin multiplicity: give ``multiplicity`` (2S+1; O2 is 3, H2 and H2O are 1),
    ``electronic_states`` (``(energy_eV, degeneracy)`` pairs) or store ``info['total_spin']['J']``, or leave the
    term out with ``terms=ThermoTerms.gas(electronic=False)``.
    """

    def __init__(self, atoms: Atoms, *, name: str | None = None, multiplicity: float | None = None,
                 electronic_states=None, symmetry_number: int | None = None, terms: ThermoTerms | None = None):
        self.atoms = atoms
        self.name = name or atoms.get_chemical_formula()
        self.terms = terms or ThermoTerms.gas()
        if electronic_states is None:
            if multiplicity is None:
                multiplicity = spin_multiplicity(atoms)
            if multiplicity is not None:
                electronic_states = [(0.0, multiplicity)]
        self.electronic_states = None if electronic_states is None else list(electronic_states)
        self._check(self.terms)
        self.composition = dict(Counter(atoms.get_chemical_symbols()))
        self.thermo = SpeciesThermo(atoms, electronic_states=self.electronic_states, symmetry_number=symmetry_number)
        self.energy = self.thermo.energy

    def _check(self, terms: ThermoTerms) -> None:
        if terms.electronic and self.electronic_states is None:
            raise ValueError(
                f"Reservoir {self.name}: the electronic term is on but the spin multiplicity is unknown. Pass "
                "multiplicity= (2S+1; O2 is 3, H2 and H2O are 1), electronic_states= or store "
                "atoms.info['total_spin']['J'], or leave the term out with terms=ThermoTerms.gas(electronic=False)."
            )

    def with_terms(self, **changes) -> "Reservoir":
        """The same molecule with some terms switched (shares the cached partition functions)."""
        other = object.__new__(Reservoir)
        other.__dict__.update(self.__dict__)
        other.terms = self.terms.replace(**changes)
        other._check(other.terms)
        return other

    def chemical_potential(self, T, p, terms: ThermoTerms | None = None):
        """μ(T, p) in eV on broadcast-compatible ``T`` (K) and ``p`` (Pa) arrays, with the terms switched on."""
        terms = terms or self.terms
        self._check(terms)
        return self.thermo.free_energy(T, terms, p)

    def __repr__(self) -> str:
        return f"Reservoir({self.name}, terms={self.terms}, g_el={self.electronic_states})"


@dataclass
class GasSpecies:
    """A volatile product (e.g. IrO2(g)) that competes with the structures at a fixed partial pressure.

    It enters the pool with the structure terms plus, by choice, its own translation and rotation at
    ``pressure`` (Pa); ``vibrations=False`` strips its vibrations and ZPE. The multiplicity comes from
    ``info['total_spin']`` if not given.
    """

    atoms: Atoms
    pressure: float
    multiplicity: float | None = None
    vibrations: bool = True
    rotation: bool = True
    translation: bool = True
