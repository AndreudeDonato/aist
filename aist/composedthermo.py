import numpy as np
from aist.thermo import ThermoFromAtoms
from aist.utils import get_energy

def process_frequencies(thermo, frequencies):
    """Remove translational and rotational modes based on linearity."""
    if frequencies is None or not hasattr(frequencies, "__len__"):
        return np.array([])
    frequencies = np.asarray(frequencies, dtype=float).ravel()
    if frequencies.size == 0:
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
    zpe = structure.info.get("zpe", 0) if include_zpe else 0
    if include_zpe and zpe is None:
        zpe = 0
    freq_raw = structure.info.get("frequencies")
    frequencies = process_frequencies(thermo, freq_raw) if include_v else []

    et = st = ev = sv = er = sr = eelec = selec = 0

    if include_t:
        et, st, _ = thermo.get_translational(Tg, Pg)
        # ideal-gas enthalpy: H_trans = U_trans + pV = 3/2 kT + kT, so G = H - TS = -kT ln q
        et = et + thermo.tp.kb * thermo.tp.J_to_eV * np.asarray(Tg, dtype=float)
    if include_v and len(frequencies) > 0:
        ev, sv, _ = thermo.get_vibrational(frequencies, Tg)
    if include_r and len(structure) > 1:
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
