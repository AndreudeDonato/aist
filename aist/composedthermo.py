"""0.1 interface to the free-energy contributions (``aist.contributions`` is the implementation)."""
import numpy as np

from aist.contributions import ThermoTerms, _parts, internal_frequencies


def process_frequencies(thermo, frequencies):
    """Remove translational and rotational modes based on linearity."""
    if frequencies is None or not hasattr(frequencies, "__len__"):
        return np.array([])
    frequencies = np.asarray(frequencies, dtype=float).ravel()
    if frequencies.size == 0:
        return np.array([])
    return frequencies[5:] if thermo.is_linear() else frequencies[6:]


def compute_thermo_contributions(structure, Tg, Pg, elec_states=None,
                                 include_v=True, include_zpe=True, include_r=True, include_t=True,
                                 symmetry_number=None):
    """
    Compute energy and entropy contributions for a given ASE Atoms object.

    ``elec_states`` (``(energy_eV, degeneracy)`` pairs) switches the electronic term on; ``symmetry_number``
    overrides the rotational symmetry number found from the point group. A missing ``info['zpe']`` is
    ½ Σ hcν of the internal modes (0.1 used 0).
    Returns:
        total_energy - Tg*total_entropy
        contributions dict: {'et', 'ev', 'er', 'eelec', 'st', 'sv', 'sr', 'selec', 'zpe'}
    """
    terms = ThermoTerms(vibrations=include_v, zpe=include_zpe, electronic=elec_states is not None,
                        rotation=include_r, translation=include_t)
    c = _parts(structure, Tg, Pg, terms, None, elec_states, symmetry_number)
    total_energy = c["energy"] + c["zpe"] + c["et"] + c["ev"] + c["er"] + c["eelec"]
    total_entropy = c["st"] + c["sv"] + c["sr"] + c["selec"]
    contributions = {k: c[k] for k in ("et", "ev", "er", "eelec", "st", "sv", "sr", "selec", "zpe")}
    return total_energy - Tg * total_entropy, contributions


__all__ = ["compute_thermo_contributions", "internal_frequencies", "process_frequencies"]
