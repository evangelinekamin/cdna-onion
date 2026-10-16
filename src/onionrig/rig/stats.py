"""Throughput statistics: proof-of-work rates, their uncertainty, and paired comparisons.

A run that finds H verified hits for targets matched with probability p in T
seconds did about H / p work, so its rate is H / (p * T). H is Poisson, so a
single run's relative error is 1 / sqrt(H). Across runs, thermal and clock
drift add variance; the standard error of the mean over runs captures both.
"""

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass

# Two-sided 95% Student t critical values by degrees of freedom.
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
        9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
        16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086, 25: 2.060, 30: 2.042,
        40: 2.021, 60: 2.000, 120: 1.980}


def t95(df: int) -> float:
    if df < 1:
        raise ValueError("need at least one degree of freedom")
    eligible = [d for d in _T95 if d <= df]
    return _T95[max(eligible)] if df <= 120 else 1.960


def pow_rate(hits: int, p_match: float, seconds: float) -> float:
    """Candidates per second implied by `hits` verified matches."""
    if p_match <= 0 or seconds <= 0:
        raise ValueError("p_match and seconds must be positive")
    return hits / (p_match * seconds)


def poisson_rel_err(hits: int) -> float:
    return 1 / math.sqrt(hits) if hits > 0 else math.inf


def self_report_z(hits: int, candidates: int, p_match: float) -> float:
    """How many standard deviations the hit count is from what the claimed work predicts.

    Large |z| means the entry's candidate count is wrong, or it skips candidates
    it claims to test.
    """
    expected = candidates * p_match
    if expected <= 0:
        return math.inf if hits else 0.0
    return (hits - expected) / math.sqrt(expected)


@dataclass(frozen=True)
class Summary:
    n: int
    mean: float
    sem: float
    ci_low: float
    ci_high: float

    @property
    def rel_sem(self) -> float:
        return self.sem / self.mean if self.mean else math.inf


def summarize(values: Sequence[float]) -> Summary:
    """Mean with a t-based 95% interval; never the min, which rewards lucky noise."""
    if not values:
        raise ValueError("nothing to summarize")
    mean = statistics.fmean(values)
    if len(values) == 1:
        return Summary(1, mean, math.inf, -math.inf, math.inf)
    sem = statistics.stdev(values) / math.sqrt(len(values))
    half = t95(len(values) - 1) * sem
    return Summary(len(values), mean, sem, mean - half, mean + half)


@dataclass(frozen=True)
class Ratio:
    """B relative to A: geometric mean of per-pair ratios with a 95% interval."""

    n: int
    ratio: float
    ci_low: float
    ci_high: float

    @property
    def verdict(self) -> str:
        if self.ci_low > 1:
            return "faster"
        if self.ci_high < 1:
            return "slower"
        return "indistinguishable"


def paired_ratio(a_rates: Sequence[float], b_rates: Sequence[float]) -> Ratio:
    """Compare paired runs (A_i, B_i) taken back to back on the same card.

    The interval comes from the spread of the log ratios. One pair measures
    neither run-to-run variation nor drift, so with fewer than two pairs the
    interval is unbounded and no verdict is possible.
    """
    if len(a_rates) != len(b_rates) or not a_rates:
        raise ValueError("need equally many A and B rates, at least one pair")
    logs = [math.log(b / a) for a, b in zip(a_rates, b_rates, strict=True)]
    center = statistics.fmean(logs)
    if len(logs) < 2:
        return Ratio(len(logs), math.exp(center), 0.0, math.inf)
    half = t95(len(logs) - 1) * statistics.stdev(logs) / math.sqrt(len(logs))
    return Ratio(len(logs), math.exp(center), math.exp(center - half), math.exp(center + half))


def abba_order(pairs: int) -> list[tuple[int, str]]:
    """Run order (pair index, side) that cancels linear drift: AB BA AB BA ..."""
    order: list[tuple[int, str]] = []
    for i in range(pairs):
        sides = ("A", "B") if i % 2 == 0 else ("B", "A")
        order += [(i, s) for s in sides]
    return order


def enough_runs(rates: Sequence[float], *, min_runs: int, max_runs: int,
                target_rel_sem: float) -> bool:
    """Adaptive repetition: stop once the mean is precise enough, or at the cap."""
    if len(rates) < min_runs:
        return False
    if len(rates) >= max_runs:
        return True
    return summarize(rates).rel_sem <= target_rel_sem
