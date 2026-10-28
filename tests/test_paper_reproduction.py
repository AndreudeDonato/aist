"""Paper I reproduction: the new core gives the cached ensembles of the paper repository bit for bit.

Skipped unless ``AIST_PAPER_DIR`` points at a checkout of the paper repository with its figure caches built.
Only data and cache files are read; nothing is imported from that repository. For each functional the
cached ensemble rows (the ``*_frames.npy`` sidecar) are rebuilt with the paper settings (vibrations, ZPE and
the electronic term on structures, rotation and translation off, O2 a triplet, the isolated Ir atom as the
reference, a 0.65 eV/Ir + 0.1 eV/O support shift) on the cached 150 × 150 grid.
"""
import os
import sys

import numpy as np
import pytest
from ase.io import read

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aist import GrandCanonicalEnsemble, Reference, Reservoir, SupportShift, ThermoTerms  # noqa: E402

PAPER = os.environ.get("AIST_PAPER_DIR")
pytestmark = pytest.mark.skipif(not PAPER, reason="set AIST_PAPER_DIR to the paper repository to run")

CASES = (("PBE", "grandcanonical_results", "gc_variants"), ("PBE0", "grandcanonical_PBE0", "gc_variants_PBE0"))
BRANCHES = (("full", True, True), ("novib", False, False), ("vib_nozpe", True, False), ("zpe_only", False, True))


def _ensemble(label: str, cache: str):
    frames = np.load(os.path.join(PAPER, "figures", "cache", f"{cache}_frames.npy"))
    assert np.all(frames >= 0), "gas-phase rows in the cache: extend this test with GasSpecies"
    traj = read(os.path.join(PAPER, "data", f"IryOx{label}.traj"), index=":")
    return GrandCanonicalEnsemble(
        [traj[i] for i in frames],
        references=[Reference(read(os.path.join(PAPER, "data", f"Ir{label}.xyz")))],
        reservoirs=[Reservoir(read(os.path.join(PAPER, "data", f"O2_{label}.xyz")), multiplicity=3)],
        terms=ThermoTerms(vibrations=True, zpe=True, electronic=True, rotation=False, translation=False,
                          pressure=0.01),
    )


def _support(cached) -> SupportShift:
    # the cache stores the settings in float32; the shift was applied with the config's decimal values
    return SupportShift(round(float(cached["gc_support_ev_per_ir"]), 6),
                        {"O": round(float(cached["gc_support_ev_per_o"]), 6)})


@pytest.mark.parametrize("label,cache,_", CASES)
def test_main_ensemble_is_reproduced(label, cache, _):
    cached = np.load(os.path.join(PAPER, "figures", "cache", f"{cache}.npz"), allow_pickle=True)
    ens = _ensemble(label, cache)
    res = ens.evaluate(T=cached["Ts"], p={"O2": cached["Ps"]}).shifted(_support(cached))
    assert np.array_equal(res.G, cached["G_tensor"])
    assert np.array_equal(res.populations, cached["Pgi"])
    assert np.array_equal(res.phase_index, cached["phase_index"])
    assert np.array_equal(ens.degeneracy, cached["degeneracy"])
    assert np.array_equal(res.composition("O"), cached["x_per_non_o"])


@pytest.mark.parametrize("label,_,cache", CASES)
def test_variants_are_reproduced(label, _, cache):
    cached = np.load(os.path.join(PAPER, "figures", "cache", f"{cache}.npz"), allow_pickle=True)
    ens = _ensemble(label, cache)
    for name, vib, zpe in BRANCHES:
        res = ens.variant(vibrations=vib, zpe=zpe).evaluate(T=cached["Ts"], p={"O2": cached["Ps"]})
        res = res.shifted(_support(cached))
        assert np.array_equal(res.G, cached[f"{name}_G_tensor"]), name
        assert np.array_equal(res.populations, cached[f"{name}_Pgi"]), name
        assert np.array_equal(res.phase_index, cached[f"{name}_phase_index"]), name
        assert np.array_equal(res.composition("O"), cached[f"{name}_x_per_non_o"]), name
