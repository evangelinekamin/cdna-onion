"""Fresh inputs for every run, with targets sized like a real search.

Benchmarks need enough hits for the statistics (relative error is
1/sqrt(hits)), but every extra target and every extra hit adds per-candidate
and per-hit work that a real 10-character search never pays. So plans use at
most 4 targets and a modest hit rate (see plan_for).
"""

import math
import secrets
from dataclasses import dataclass

from .. import onion, splitkey
from .runner import DONE
from .score import RunScore

MIN_CALIBRATION_HITS = 30
_rng = secrets.SystemRandom()


@dataclass(frozen=True)
class TargetPlan:
    chars: int
    count: int

    @property
    def p_match(self) -> float:
        return self.count / 32**self.chars


# ~1.2e-7 match probability: tens to thousands of hits per second from 1e8 to
# 3e10 candidates/s, so even a very fast entry cannot flood its own calibration.
# Slower entries get too few hits and are planned from their own count instead.
CALIBRATION_PLAN = TargetPlan(5, 4)


def planning_rate(calibration: RunScore) -> float:
    """Rough candidates/s from a calibration run, used only to size targets.

    Uses the verified proof-of-work rate when there are enough hits; an entry too
    slow to hit the conservative calibration targets is planned from its own
    count. Scored runs never use self-reports.
    """
    if calibration.outcome != DONE or calibration.bad_offsets:
        raise RuntimeError(f"calibration run failed: {calibration.reason}")
    if calibration.hits >= MIN_CALIBRATION_HITS:
        return calibration.rate
    if calibration.candidates and calibration.wall_seconds > 0:
        return calibration.candidates / calibration.wall_seconds
    raise RuntimeError("calibration produced no hits and no candidate count")


def plan_for(rate: float, hits_per_second: float, max_targets: int = 4) -> TargetPlan:
    """Targets giving roughly `hits_per_second` hits for an entry doing `rate` candidates/s.

    Benchmarks should resemble real searches: a few targets, and hits rare enough
    that per-hit work (a cold path that re-derives the point, say) stays a rounding
    error, because a real 10-character search almost never hits. So the target
    count is capped and the achievable match probability closest to the wish is
    chosen (within about 3x). That is plenty for the statistics, which only need
    enough hits.
    """
    if rate <= 0 or hits_per_second <= 0:
        raise ValueError("rate and hits_per_second must be positive")
    want = math.log(min(1.0, hits_per_second / rate))
    best: tuple[float, TargetPlan] | None = None
    for chars in range(1, onion.MAX_PREFIX_CHARS + 1):
        for count in range(1, min(max_targets, 32**chars) + 1):
            error = abs(math.log(count / 32**chars) - want)
            if best is None or error < best[0]:
                best = (error, TargetPlan(chars, count))
        if math.log(1 / 32**chars) < want - 4:
            break  # every longer prefix is already far too rare
    assert best is not None
    return best[1]


def random_targets(plan: TargetPlan) -> tuple[onion.Target, ...]:
    texts: set[str] = set()
    while len(texts) < plan.count:
        texts.add("".join(_rng.choice(onion.ALPHABET) for _ in range(plan.chars)))
    return onion.normalize_targets(texts)


@dataclass(frozen=True)
class RunInputs:
    a0: bytes                          # public start point; its scalar is discarded
    targets: tuple[onion.Target, ...]

    @property
    def target_texts(self) -> tuple[str, ...]:
        return tuple(t.text for t in self.targets)


def fresh_inputs(plan: TargetPlan) -> RunInputs:
    """A new random start point and new random targets: nothing precomputable."""
    return RunInputs(splitkey.start_point(splitkey.random_start_scalar()), random_targets(plan))


def invalid_start_point() -> bytes:
    """A canonical 32-byte encoding that is not a curve point (for the conformance test)."""
    y = 2
    while True:
        candidate = y.to_bytes(32, "little")
        try:
            splitkey.E.decode(candidate)
        except ValueError:
            return candidate
        y += 1
