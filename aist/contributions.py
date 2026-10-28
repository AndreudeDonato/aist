"""Free-energy contributions of one ASE ``Atoms`` object: the core of aist.

What a function needs is read from the ``Atoms`` object:
- the energy, from its calculator or ``info['energy']`` (eV);
- ``info['frequencies']``: all 3N harmonic frequencies in cm⁻¹, ascending, rigid-body modes first;
- ``info['zpe']`` (eV);
- ``info['total_spin']['J']``: the spin multiplicity 2S+1.

Temperatures (K) and pressures (Pa) may be scalars or arrays of any shape; energies come back in eV and entropies
in eV/K. Which terms enter a free energy is the user's choice (``ThermoTerms``); none is right or wrong.
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, replace

import numpy as np
from ase import Atoms

from aist.thermo import ThermoFromAtoms
from aist.utils import get_energy

__all__ = [
    "ThermoTerms", "energy", "internal_frequencies", "zero_point_energy", "vibrational", "rotational",
    "translational", "electronic", "contributions", "free_energy",
]


@dataclass(frozen=True)
class ThermoTerms:
    """The terms of a free energy, each switched on or off by the user.

    By default only the electronic contribution enters: the energy plus the spin degeneracy −kT ln(2S+1).
    ``ThermoTerms.gas()`` switches every term on: the ideal-gas free energy of a molecule. ``pressure`` (Pa) is
    used by the translational term when no pressure is passed to the function.
    """

    vibrations: bool = False
    zpe: bool = False
    electronic: bool = True
    rotation: bool = False
    translation: bool = False
    pressure: float | None = None

    @classmethod
    def gas(cls, **overrides) -> "ThermoTerms":
        flags = dict(vibrations=True, zpe=True, electronic=True, rotation=True, translation=True)
        flags.update(overrides)
        return cls(**flags)

    def replace(self, **changes) -> "ThermoTerms":
        return replace(self, **changes)


def energy(atoms: Atoms) -> float:
    """Total energy (eV): ``info['energy']`` or the attached calculator's."""
    return get_energy(atoms)


def internal_frequencies(atoms: Atoms) -> np.ndarray:
    """Internal-mode frequencies (cm⁻¹): ``info['frequencies']`` without the 5 (linear) or 6 rigid-body modes."""
    freqs = atoms.info.get("frequencies")
    if freqs is None or not hasattr(freqs, "__len__"):
        return np.array([])
    freqs = np.asarray(freqs, dtype=float).ravel()
    if freqs.size == 0:
        return np.array([])
    return freqs[5:] if ThermoFromAtoms(atoms).is_linear() else freqs[6:]


def zero_point_energy(atoms: Atoms) -> float:
    """``info['zpe']`` (eV); without it, ½ Σ hcν over the real internal modes."""
    zpe = atoms.info.get("zpe")
    if zpe is not None:
        return zpe
    nu = internal_frequencies(atoms)
    tp = ThermoFromAtoms(atoms).tp
    return float(0.5 * np.sum(nu[nu > 0]) * 100.0 * tp.c * tp.h_planck * tp.J_to_eV)


def vibrational(atoms: Atoms, T):
    """Harmonic vibrational ``(U, S)`` of the internal modes, thermal part only (the ZPE is a separate term)."""
    nu = internal_frequencies(atoms)
    if len(nu) == 0:
        return 0, 0
    return ThermoFromAtoms(atoms).get_vibrational(nu, T)[:2]


def rotational(atoms: Atoms, T, *, symmetry_number: int | None = None):
    """Rigid-rotor ``(U, S)``; the symmetry number comes from the point group unless given. Zero for one atom."""
    if len(atoms) < 2:
        return 0, 0
    return ThermoFromAtoms(atoms).get_rotational(T, sigma=symmetry_number)[:2]


def translational(atoms: Atoms, T, p):
    """Ideal-gas translational ``(H, S)`` at pressure ``p`` (Pa); H = U + pV, so H − TS = −kT ln q."""
    thermo = ThermoFromAtoms(atoms)
    U, S, _ = thermo.get_translational(T, p)
    return U + thermo.tp.kb * thermo.tp.J_to_eV * np.asarray(T, dtype=float), S


_NO_SPIN = ("no spin multiplicity given or stored in info['total_spin']['J']: the electronic term uses "
            "degeneracy 1 (pass multiplicity= to set it)")


def _spin_multiplicity(atoms: Atoms):
    spin = atoms.info.get("total_spin")
    if isinstance(spin, str):
        spin = json.loads(spin.replace("_JSON ", "", 1).strip())
    if isinstance(spin, dict):
        return spin.get("J")
    return None


def electronic(atoms: Atoms, T, *, multiplicity: float | None = None, states=None):
    """Electronic ``(U, S)`` from ``states`` (``(energy_eV, degeneracy)`` pairs) or the spin multiplicity.

    The multiplicity is ``multiplicity`` or ``info['total_spin']['J']``; with neither the degeneracy is 1 (the
    term is zero) and a warning says so.
    """
    if states is None:
        if multiplicity is None:
            multiplicity = _spin_multiplicity(atoms)
        if multiplicity is None:
            warnings.warn(_NO_SPIN, stacklevel=2)
            return 0, 0
        states = [(0.0, multiplicity)]
    return ThermoFromAtoms(atoms).get_electronic(list(states), T)[:2]


def _parts(atoms, T, p, terms, multiplicity, states, symmetry_number) -> dict:
    """Energies and entropies of the switched-on terms (absent terms are 0), with the energy and the ZPE."""
    if p is None:
        p = terms.pressure
    if terms.translation and p is None:
        raise ValueError("The translational term needs a pressure (p=..., or ThermoTerms(pressure=...), in Pa)")
    et, st = translational(atoms, T, p) if terms.translation else (0, 0)
    ev, sv = vibrational(atoms, T) if terms.vibrations else (0, 0)
    er, sr = rotational(atoms, T, symmetry_number=symmetry_number) if terms.rotation else (0, 0)
    eelec, selec = (electronic(atoms, T, multiplicity=multiplicity, states=states) if terms.electronic
                    else (0, 0))
    return {
        "energy": energy(atoms), "zpe": zero_point_energy(atoms) if terms.zpe else 0,
        "et": et, "ev": ev, "er": er, "eelec": eelec, "st": st, "sv": sv, "sr": sr, "selec": selec,
    }


def contributions(atoms: Atoms, T, p=None, terms: ThermoTerms = ThermoTerms(), *, multiplicity=None,
                  states=None, symmetry_number=None) -> dict:
    """Every switched-on term as a free-energy contribution (eV) on ``T`` (and ``p``).

    Keys: ``energy``, ``zpe``, ``vibrational`` (U − TS), ``rotational``, ``translational`` and ``electronic``;
    ``free_energy`` returns their sum.
    """
    c = _parts(atoms, T, p, terms, multiplicity, states, symmetry_number)
    out = {"energy": c["energy"], "zpe": c["zpe"]}
    for name, (e, s) in (("vibrational", ("ev", "sv")), ("rotational", ("er", "sr")),
                         ("translational", ("et", "st")), ("electronic", ("eelec", "selec"))):
        out[name] = c[e] - T * c[s]
    return out


def free_energy(atoms: Atoms, T, p=None, terms: ThermoTerms = ThermoTerms(), *, multiplicity=None, states=None,
                symmetry_number=None):
    """Free energy E + ZPE + Σ U − T Σ S over the terms that ``terms`` switches on (eV)."""
    c = _parts(atoms, T, p, terms, multiplicity, states, symmetry_number)
    total_energy = c["energy"] + c["zpe"] + c["et"] + c["ev"] + c["er"] + c["eelec"]
    total_entropy = c["st"] + c["sv"] + c["sr"] + c["selec"]
    return total_energy - T * total_entropy
