from ase import Atoms
import re

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
        
def get_info_value(atoms, key, default=0):
    """Safely get a value from atoms.info, return default if key not found."""
    return atoms.info.get(key, default)
    
def to_latex_formula(formula):
        return re.sub(r'([A-Z][a-z]*)(\d*)',
                      lambda m: m.group(1) + (f'$_{{{m.group(2)}}}$' if
                                              m.group(2) else ''),
                      formula)
