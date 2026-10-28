import os
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
        
def to_latex_formula(formula):
        return re.sub(r'([A-Z][a-z]*)(\d*)',
                      lambda m: m.group(1) + (f'$_{{{m.group(2)}}}$' if
                                              m.group(2) else ''),
                      formula)


def ensure_output_dir(path: str):
    """Ensure the output directory for a file path exists."""
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
