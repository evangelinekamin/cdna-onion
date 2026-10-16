"""Turn one grind session into a verified, scored result.

A run is valid only if the entry finished cleanly AND every reported offset is
a genuine match. One bad offset voids the run, because a user would have been
handed a wrong key. The rate counts each distinct verified hit once.
"""

import math
import random
from concurrent.futures import Executor
from dataclasses import dataclass, field

from .. import onion
from . import stats
from .runner import DONE, GrindRecord
from .verify import HitVerifier, parallel_failures

# Beyond this many standard deviations, the entry's own candidate count is not
# credible. Six sigma keeps honest runs from ever tripping it.
SELF_REPORT_SIGMA = 6.0
BAD_OFFSETS_KEPT = 20
SAMPLE_FIRST = 1000  # a broken matcher is caught here before millions are verified


@dataclass(frozen=True)
class RunScore:
    valid: bool
    reason: str
    outcome: str
    hits: int
    duplicates: int
    bad_offsets: tuple[int, ...]
    p_match: float
    wall_seconds: float
    rate: float
    rate_rel_err: float
    candidates: int | None
    self_report_z: float | None
    flags: tuple[str, ...] = field(default=())

    def as_record(self) -> dict:
        return {"valid": self.valid, "reason": self.reason, "outcome": self.outcome,
                "hits": self.hits, "duplicates": self.duplicates,
                "bad_offsets": [str(k) for k in self.bad_offsets],
                "p_match": self.p_match, "wall_seconds": self.wall_seconds, "rate": self.rate,
                "rate_rel_err": self.rate_rel_err,
                "candidates": None if self.candidates is None else str(self.candidates),
                "self_report_z": self.self_report_z, "flags": list(self.flags)}


def score(record: GrindRecord, a0: bytes, targets: tuple[onion.Target, ...],
          pool: Executor | None = None) -> RunScore:
    p_match = float(onion.match_probability(targets))
    unique = sorted(set(record.offsets))
    duplicates = len(record.offsets) - len(unique)
    verifier = HitVerifier(a0, targets)
    sample = random.Random().sample(unique, min(SAMPLE_FIRST, len(unique)))
    bad = verifier.failures(sample, BAD_OFFSETS_KEPT) or \
        parallel_failures(verifier, unique, pool, limit=BAD_OFFSETS_KEPT)
    flags = ["duplicate-offsets"] if duplicates else []

    z = None
    if record.candidates is not None:
        z = stats.self_report_z(len(unique), record.candidates, p_match)
        if abs(z) > SELF_REPORT_SIGMA:
            flags.append("self-report-inconsistent")

    def result(valid: bool, reason: str, rate: float = 0.0) -> RunScore:
        return RunScore(valid, reason, record.outcome, len(unique), duplicates,
                        tuple(sorted(bad)), p_match, record.wall_seconds, rate,
                        stats.poisson_rel_err(len(unique)), record.candidates, z, tuple(flags))

    if record.outcome != DONE:
        return result(False, f"{record.outcome}: {record.message}")
    if bad:
        more = "+" if len(bad) >= BAD_OFFSETS_KEPT else ""
        return result(False, f"{len(bad)}{more} reported offset(s) do not match any target")
    if record.wall_seconds <= 0 or not unique:
        return result(False, "no verified hits; nothing to measure")
    rate = stats.pow_rate(len(unique), p_match, record.wall_seconds)
    return result(True, "", rate if math.isfinite(rate) else 0.0)
