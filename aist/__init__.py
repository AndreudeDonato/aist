"""aist: ab initio statistical thermodynamics of cluster ensembles on ASE ``Atoms`` objects.

    from aist import GrandCanonicalEnsemble, Reference, Reservoir

    ens = GrandCanonicalEnsemble(clusters, references=[Reference(metal_atom)],
                                 reservoirs=[Reservoir(o2, multiplicity=3)])
    res = ens.evaluate(T=temperatures, p={"O2": pressures})
    res.populations, res.composition("O"), res.most_stable()
"""

__version__ = "0.2.0"

from aist.balance import BalanceError  # noqa: E402
from aist.gce import EnsembleResult, GrandCanonicalEnsemble, SupportShift  # noqa: E402
from aist.species import GasSpecies, Reference, Reservoir, ThermoTerms  # noqa: E402

__all__ = [
    "BalanceError",
    "EnsembleResult",
    "GasSpecies",
    "GrandCanonicalEnsemble",
    "Reference",
    "Reservoir",
    "SupportShift",
    "ThermoTerms",
    "__version__",
]
