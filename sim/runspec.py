"""
Run coordinates and the out-of-domain reliability gate.

EVERY result row carries its full scenario coordinates (P2-D12). A number
without them is not reportable: the capacity model alone moves the headline
from +23.0% to +6.0%, and the plateau scenario decides whether visit merging is
a free operational saving or a trade-off against yield.
"""

from dataclasses import dataclass, asdict, field as _dcfield
from typing import Optional

# Fraction of scored field-years that may fall outside the observed TSI domain
# before a result is reported as unreliable rather than quoted. ASSUMED by
# judgement; reported alongside the figure it gates (Phase 2 section 6.8).
OOD_RELIABILITY_THRESHOLD = 0.05


@dataclass(frozen=True)
class RunCoords:
    """Complete coordinates of one run. Serialise with `asdict`."""
    # biology
    curve: str
    plateau: str
    kappa: float
    delta_thr: float
    asi0: float
    scoring_mode: str
    # population
    pair_scope: str
    pair_rho: float
    tau_min: float
    tau_max: float
    n_fields: int
    farm_seed: int
    # operations
    objective: str
    capacity_model: str
    capacity: float
    split_male: bool
    split_gap: Optional[int]
    # provenance
    phenology_table_sha: str = ""
    git_commit: str = ""
    code_path: str = "phase3"

    def as_row(self):
        return asdict(self)


@dataclass
class Reliability:
    n_field_years: int
    n_out_of_domain: int
    threshold: float = OOD_RELIABILITY_THRESHOLD

    @property
    def fraction(self):
        return (self.n_out_of_domain / self.n_field_years
                if self.n_field_years else 0.0)

    @property
    def reliable(self):
        return self.fraction <= self.threshold

    def verdict(self):
        return ("reliable" if self.reliable else
                f"UNRELIABLE: {self.fraction:.1%} of scored field-years fall "
                f"outside the observed TSI domain, above the {self.threshold:.0%} "
                f"threshold. Report as unreliable rather than quoting it.")

    def as_row(self):
        return dict(n_field_years=self.n_field_years,
                    n_out_of_domain=self.n_out_of_domain,
                    ood_fraction=self.fraction,
                    ood_threshold=self.threshold,
                    reliable=self.reliable)


def table_sha(path):
    """Content hash of the phenology table, so a regenerated table invalidates results."""
    import hashlib
    try:
        with open(path, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()[:16]
    except OSError:
        return ""


def git_commit(repo="."):
    import subprocess
    try:
        return subprocess.run(["git", "-C", repo, "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True,
                              timeout=10).stdout.strip()
    except Exception:
        return ""
