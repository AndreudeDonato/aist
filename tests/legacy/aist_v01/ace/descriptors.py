"""Per-atom ACE-style descriptors (SOAP from dscribe).

SOAP gives each atom a fixed-length vector built from a radial basis times
spherical harmonics — i.e. a radial distribution coupled with an angular
distribution around the atom, the same conceptual content as ACE descriptors
used for per-atom unsupervised clustering.
"""

from __future__ import annotations

import numpy as np
from dscribe.descriptors import SOAP


_SOAP_CACHE: dict[tuple, SOAP] = {}


def _soap_factory(species: tuple[str, ...], r_cut: float, n_max: int, l_max: int,
                  sigma: float, average: str = "off") -> SOAP:
    key = (tuple(sorted(species)), float(r_cut), int(n_max), int(l_max),
           float(sigma), average)
    soap = _SOAP_CACHE.get(key)
    if soap is None:
        soap = SOAP(
            species=list(key[0]),
            r_cut=float(r_cut),
            n_max=int(n_max),
            l_max=int(l_max),
            sigma=float(sigma),
            periodic=False,
            sparse=False,
            average=average,
        )
        _SOAP_CACHE[key] = soap
    return soap


def compute_per_atom_descriptors(
    structures,
    species_list,
    *,
    r_cut: float,
    n_max: int,
    l_max: int,
    sigma: float,
):
    """Compute SOAP descriptors per atom for every structure.

    Returns
    -------
    per_struct_descriptors : list of np.ndarray
        ``per_struct_descriptors[si]`` has shape ``(n_atoms_si, D)``.
    """
    soap = _soap_factory(tuple(species_list), r_cut, n_max, l_max, sigma)
    out = []
    for atoms in structures:
        d = soap.create(atoms)
        out.append(np.asarray(d, dtype=float))
    return out


def split_descriptors_by_species(structures, descriptors, species: str):
    """Return descriptors restricted to one species + provenance.

    Parameters
    ----------
    structures
        Iterable of ``ase.Atoms``.
    descriptors
        ``per_struct_descriptors`` from :func:`compute_per_atom_descriptors`.
    species
        ``"Ir"`` or ``"O"``.

    Returns
    -------
    X : np.ndarray  (n_atoms, D)
        Pooled descriptor block for that species.
    provenance : list of (frame_index, atom_index_in_structure)
        One entry per row of ``X``.
    """
    rows = []
    prov = []
    for si, (atoms, desc) in enumerate(zip(structures, descriptors)):
        symbols = atoms.get_chemical_symbols()
        for ai, sym in enumerate(symbols):
            if sym == species:
                rows.append(desc[ai])
                prov.append((si, ai))
    if rows:
        X = np.vstack(rows)
    else:
        X = np.zeros((0, descriptors[0].shape[1] if descriptors else 0))
    return X, prov
