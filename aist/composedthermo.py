import numpy as np
from aist.thermo import ThermoFromAtoms
from aist.utils import get_energy, get_info_value

def process_frequencies(thermo, frequencies):
    """Remove translational and rotational modes based on linearity."""
    if frequencies is None or len(frequencies) == 0:
        return np.array([])
    return frequencies[5:] if thermo.is_linear() else frequencies[6:]

def compute_thermo_contributions(structure, Tg, Pg, elec_states=None,
                                 include_v=True, include_zpe=True, include_r=True, include_t=True):
    """
    Compute energy and entropy contributions for a given ASE Atoms object.
    Returns:
        total_energy - Tg*total_entropy
        contributions dict: {'et', 'ev', 'er', 'eelec', 'st', 'sv', 'sr', 'selec', 'zpe'}
    """
    en = get_energy(structure)
    thermo = ThermoFromAtoms(structure)
    zpe = get_info_value(structure, "zpe") if include_zpe else 0
    frequencies = process_frequencies(thermo, get_info_value(structure, "frequencies")) if include_v else []

    et = st = ev = sv = er = sr = eelec = selec = 0

    if include_t:
        et, st, _ = thermo.get_translational(Tg, Pg)
    if include_v and len(frequencies) > 0:
        ev, sv, _ = thermo.get_vibrational(frequencies, Tg)
    if include_r and len(frequencies) > 0:
        er, sr, _ = thermo.get_rotational(Tg)
    if elec_states is not None:
        eelec, selec, _ = thermo.get_electronic(elec_states, Tg)

    total_energy = en + zpe + et + ev + er + eelec
    total_entropy = st + sv + sr + selec

    contributions = {
        "et": et, "ev": ev, "er": er, "eelec": eelec,
        "st": st, "sv": sv, "sr": sr, "selec": selec,
        "zpe": zpe
    }

    return total_energy - Tg * total_entropy, contributions
