from ase import Atoms

def get_energy(atoms: Atoms):
    """Get the energy from an ASE Atoms object.

    Tries .info["energy"] first; if not present, falls back to get_potential_energy().
    """
    try:
        return atoms.info["energy"]
    except KeyError:
        try:
            return atoms.get_potential_energy()
        except Exception:
            raise ValueError("Energy not found in .info['energy'] or via get_potential_energy()")
