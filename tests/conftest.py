import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from sim.phenology import DssatPhenology

PHENO = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "results", "phenology_table.npz")


@pytest.fixture(scope="session")
def ph():
    if not os.path.exists(PHENO):
        pytest.skip("phenology table absent")
    return DssatPhenology(PHENO)


@pytest.fixture(scope="session")
def ref_day(ph):
    """Reference sowing day: the middle of the precomputed grid."""
    return ph.sow_days[len(ph.sow_days) // 2]
