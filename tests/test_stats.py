"""Throughput statistics."""

import math
import random

import pytest

from onionrig.rig import stats


def test_pow_rate_and_counting_error():
    assert stats.pow_rate(1000, 1 / 1024, 2.0) == pytest.approx(512_000)
    assert stats.poisson_rel_err(10_000) == pytest.approx(0.01)
    assert stats.poisson_rel_err(0) == math.inf
    with pytest.raises(ValueError):
        stats.pow_rate(1, 0, 1)


def test_self_report_z_flags_inflated_counts():
    assert abs(stats.self_report_z(1000, 1_024_000, 1 / 1024)) < 1e-9
    assert stats.self_report_z(1000, 102_400_000, 1 / 1024) < -50   # claimed 100x the work
    assert stats.self_report_z(1000, 512_000, 1 / 1024) > 20         # half-counted


def test_self_report_z_is_calibrated_on_honest_poisson_counts():
    rng = random.Random(3)
    p, candidates = 1 / 4096, 4_096_000
    zs = []
    for _ in range(400):
        hits = sum(rng.random() < p * 1000 for _ in range(candidates // 1000))  # binomial-ish
        zs.append(stats.self_report_z(hits, candidates, p))
    assert sum(abs(z) > 3 for z in zs) <= 6  # about 0.3% expected beyond 3 sigma


def test_t95_table():
    assert stats.t95(1) == pytest.approx(12.706)
    assert stats.t95(27) == pytest.approx(2.060)   # nearest tabulated df at or below
    assert stats.t95(10_000) == pytest.approx(1.960)
    with pytest.raises(ValueError):
        stats.t95(0)


def test_summarize():
    s = stats.summarize([10.0, 12.0, 11.0, 13.0])
    assert s.mean == pytest.approx(11.5)
    assert s.ci_low < 11.5 < s.ci_high
    single = stats.summarize([5.0])
    assert single.mean == 5.0 and single.sem == math.inf
    with pytest.raises(ValueError):
        stats.summarize([])


def test_paired_ratio_verdicts():
    a = [100.0, 101.0, 99.0, 100.5]
    assert stats.paired_ratio(a, [x * 1.20 for x in a]).verdict == "faster"
    assert stats.paired_ratio(a, [x * 0.80 for x in a]).verdict == "slower"
    noisy = stats.paired_ratio([100.0, 100.0, 100.0], [99.0, 101.5, 99.8])
    assert noisy.verdict == "indistinguishable"
    assert stats.paired_ratio(a, a).ratio == pytest.approx(1.0)


def test_a_single_pair_gives_no_verdict():
    # One pair measures neither run-to-run variation nor drift.
    single = stats.paired_ratio([100.0], [130.0])
    assert single.ratio == pytest.approx(1.3) and single.ci_high == math.inf
    assert single.verdict == "indistinguishable"


def test_abba_order_cancels_linear_drift():
    order = stats.abba_order(4)
    assert [s for _, s in order] == list("ABBAABBA")
    # Under a linear drift, each side's mean slot position is the same.
    a_slots = [i for i, (_, s) in enumerate(order) if s == "A"]
    b_slots = [i for i, (_, s) in enumerate(order) if s == "B"]
    assert sum(a_slots) == sum(b_slots)


def test_enough_runs():
    steady = [100.0, 100.1, 99.9]
    assert not stats.enough_runs(steady[:2], min_runs=3, max_runs=10, target_rel_sem=0.01)
    assert stats.enough_runs(steady, min_runs=3, max_runs=10, target_rel_sem=0.01)
    wild = [50.0, 150.0, 100.0]
    assert not stats.enough_runs(wild, min_runs=3, max_runs=10, target_rel_sem=0.01)
    assert stats.enough_runs(wild * 4, min_runs=3, max_runs=10, target_rel_sem=0.01)
