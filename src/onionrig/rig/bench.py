"""Measured benchmarking: thermally gated runs, adaptive repetition, and paired duels.

Every run gets a fresh start point and fresh targets, starts from a cooled GPU,
is sampled for thermal state throughout, is scored by verified proof-of-work,
and is appended to disk before the next one begins.

An entry is judged on every run it makes: one invalid run (a forged or wrong
hit, a crash, a protocol violation) fails the whole benchmark or duel. Averaging
over the runs that happened to go well would let a sometimes-wrong entry rank,
and a wrong key handed to a user is the one outcome this rig exists to prevent.
"""

import platform
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from typing import Any

from .. import __version__
from . import runner, stats
from . import telemetry as tm
from .plan import CALIBRATION_PLAN, TargetPlan, fresh_inputs, plan_for, planning_rate
from .results import JsonlWriter, expected_seconds
from .score import RunScore, score


@dataclass(frozen=True)
class BenchConfig:
    seconds: float = 60.0
    calibrate_seconds: float = 10.0
    hits_per_second: float = 2_000.0
    min_runs: int = 3
    max_runs: int = 10
    target_rel_sem: float = 0.005
    cool_to_c: float | None = None      # None: idle temperature + cool_margin_c
    cool_margin_c: float = 8.0
    cool_timeout_s: float = 180.0
    sample_interval_s: float = 1.0
    strict_thermal: bool = False
    verify_workers: int = 8


@dataclass(frozen=True)
class Measured:
    score: RunScore
    flags: tuple[str, ...]
    counted: bool  # valid, and not excluded by --strict-thermal


class EntryFailed(RuntimeError):
    """An entry produced an invalid run; it is out of this benchmark or duel."""


class Bench:
    """Shared machinery: verification pool, telemetry, and the durable run log."""

    def __init__(self, config: BenchConfig, writer: JsonlWriter,
                 telemetry: tm.Telemetry | None, limits: runner.Limits = runner.DEFAULT_LIMITS):
        self.config, self.writer, self.telemetry, self.limits = config, writer, telemetry, limits
        self.pool = ProcessPoolExecutor(config.verify_workers) if config.verify_workers > 1 \
            else None
        self.idle_c = telemetry.sample().temp_c if telemetry else None

    def close(self) -> None:
        if self.pool:
            self.pool.shutdown()

    def __enter__(self) -> "Bench":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _cool_down(self) -> float | None:
        if self.telemetry is None:
            return None
        threshold = self.config.cool_to_c or (self.idle_c or 50.0) + self.config.cool_margin_c
        return tm.wait_until_cool(self.telemetry, threshold, self.config.cool_timeout_s)

    def _attempt(self, spec: runner.EntrySpec, plan: TargetPlan,
                 seconds: float) -> tuple[RunScore, tm.ThermalReport, float | None, Any]:
        inputs = fresh_inputs(plan)
        start_temp = self._cool_down()
        samples: list[tm.Sample] = []
        if self.telemetry is None:
            record = runner.run_grind(spec, inputs.a0, inputs.target_texts, seconds, self.limits)
        else:
            with tm.Sampler(self.telemetry, self.config.sample_interval_s) as sampler:
                record = runner.run_grind(spec, inputs.a0, inputs.target_texts, seconds,
                                          self.limits)
            samples = sampler.samples
        return score(record, inputs.a0, inputs.targets, self.pool), tm.assess(samples), \
            start_temp, record

    def _write_run(self, labels: dict[str, Any], plan: TargetPlan, result: RunScore,
                   thermal: tm.ThermalReport, start_temp: float | None, record: Any,
                   attempt: int, counted: bool, flags: tuple[str, ...]) -> None:
        self.writer.write({
            "kind": "run", "time": time.time(), **labels, **result.as_record(),
            "attempt": attempt, "counted": counted, "flags": list(flags),
            "setup_seconds": record.setup_seconds, "plan": [plan.chars, plan.count],
            "start_temp_c": start_temp, "thermal": asdict(thermal),
            "log": str(record.log) if record.log else None})

    def measured_run(self, spec: runner.EntrySpec, plan: TargetPlan, seconds: float,
                     labels: dict[str, Any]) -> Measured:
        """One scored run, retried once if the GPU genuinely thermally throttled.

        Each attempt is written with its final `counted` decision, so the log
        never needs re-interpreting: a replaced attempt says so on disk."""
        result, thermal, start_temp, record = self._attempt(spec, plan, seconds)
        attempt = 1
        if result.valid and "thermal-throttle" in thermal.flags:
            self._write_run(labels, plan, result, thermal, start_temp, record, attempt, False,
                            result.flags + thermal.flags + ("replaced-by-retry",))
            result, thermal, start_temp, record = self._attempt(spec, plan, seconds)
            attempt = 2
        flags = result.flags + thermal.flags
        excluded = self.config.strict_thermal and bool(thermal.flags)
        if excluded:
            flags += ("excluded",)
        counted = result.valid and not excluded
        self._write_run(labels, plan, result, thermal, start_temp, record, attempt, counted, flags)
        return Measured(result, flags, counted)

    def calibrate(self, spec: runner.EntrySpec, labels: dict[str, Any]) -> float:
        """Rough candidates/s, used only to size targets (never as a score)."""
        m = self.measured_run(spec, CALIBRATION_PLAN, self.config.calibrate_seconds,
                              {**labels, "role": "calibration"})
        try:
            return planning_rate(m.score)
        except RuntimeError as exc:
            raise EntryFailed(str(exc)) from exc


def _labels(info: Any, gpu: str | None) -> dict[str, Any]:
    return {"entry": info.name, "version": info.version, "model": info.model,
            "gpu": gpu or info.device or "unknown", "rig": __version__,
            "host": platform.node()}


@dataclass(frozen=True)
class BenchSummary:
    entry: str
    gpu: str
    runs: int
    summary: stats.Summary | None
    plan: TargetPlan | None
    failure: str | None = None


def _summary_fields(summary: stats.Summary | None) -> dict[str, Any]:
    if summary is None:
        return {"rate": None}
    return {"rate": summary.mean, "rate_ci": [summary.ci_low, summary.ci_high],
            "rel_sem": summary.rel_sem,
            "eta10_median_s": expected_seconds(summary.mean, 10),
            "eta13_median_s": expected_seconds(summary.mean, 13)}


def _bench_runs(spec: runner.EntrySpec, bench_: Bench, plan: TargetPlan,
                labels: dict[str, Any]) -> list[float]:
    """Rates of the counted runs; raises EntryFailed on the first invalid run."""
    config, rates, attempts = bench_.config, [], 0
    while not stats.enough_runs(rates, min_runs=config.min_runs, max_runs=config.max_runs,
                                target_rel_sem=config.target_rel_sem):
        attempts += 1
        m = bench_.measured_run(spec, plan, config.seconds, {**labels, "role": "bench"})
        if not m.score.valid:
            raise EntryFailed(m.score.reason)
        if m.counted:
            rates.append(m.score.rate)
        if attempts >= 2 * config.max_runs:
            break  # thermal exclusions alone must not loop forever
    return rates


def bench(spec: runner.EntrySpec, bench_: Bench, gpu: str | None = None) -> BenchSummary:
    info = runner.run_info(spec, bench_.limits)
    labels = _labels(info, gpu)
    bench_.writer.write({"kind": "session", "mode": "bench", "time": time.time(), **labels,
                         "config": asdict(bench_.config), "isolated": spec.isolate,
                         "telemetry": bench_.telemetry.name if bench_.telemetry else None})
    plan: TargetPlan | None = None
    try:
        plan = plan_for(bench_.calibrate(spec, labels), bench_.config.hits_per_second)
        rates = _bench_runs(spec, bench_, plan, labels)
        failure = None
    except EntryFailed as exc:
        rates, failure = [], str(exc)
    summary = stats.summarize(rates) if rates else None
    bench_.writer.write({"kind": "summary", "mode": "bench", "time": time.time(), **labels,
                         "runs": len(rates), "plan": [plan.chars, plan.count] if plan else None,
                         "failed": failure, **_summary_fields(None if failure else summary)})
    return BenchSummary(info.name, labels["gpu"], len(rates), None if failure else summary,
                        plan, failure)


@dataclass(frozen=True)
class DuelSummary:
    a: str
    b: str
    pairs: int
    ratio: stats.Ratio | None
    failure: str | None = None


def _counted_blocks(results: dict[tuple[int, str], Measured], pairs: int) -> list[int]:
    """Pairs whose whole ABBA block (pairs 2i, 2i+1) counted, so drift still cancels."""
    keep: list[int] = []
    for block in range(0, pairs, 2):
        members = [p for p in (block, block + 1) if p < pairs]
        if all(results[(p, s)].counted for p in members for s in "AB"):
            keep += members
    return keep


def duel(spec_a: runner.EntrySpec, spec_b: runner.EntrySpec, bench_: Bench, pairs: int,
         gpu: str | None = None) -> DuelSummary:
    """B against A in ABBA-ordered pairs on the same card; the ratio is B's speedup."""
    specs = {"A": spec_a, "B": spec_b}
    infos = {side: runner.run_info(spec, bench_.limits) for side, spec in specs.items()}
    labels = {side: _labels(info, gpu) for side, info in infos.items()}
    bench_.writer.write({"kind": "session", "mode": "duel", "time": time.time(),
                         "a": labels["A"], "b": labels["B"], "pairs": pairs})
    side = "A"
    try:
        rates = {}
        for side in ("A", "B"):
            rates[side] = bench_.calibrate(specs[side], labels[side])
        plan = plan_for(max(rates.values()), bench_.config.hits_per_second)
        results: dict[tuple[int, str], Measured] = {}
        for pair, side in stats.abba_order(pairs):
            m = bench_.measured_run(specs[side], plan, bench_.config.seconds,
                                    {**labels[side], "role": f"duel-{side}", "pair": pair})
            if not m.score.valid:
                raise EntryFailed(m.score.reason)
            results[(pair, side)] = m
        good = _counted_blocks(results, pairs)
        ratio = stats.paired_ratio([results[(p, "A")].score.rate for p in good],
                                   [results[(p, "B")].score.rate for p in good]) if good else None
        failure = None
    except EntryFailed as exc:
        good, ratio, failure = [], None, f"{side} ({infos[side].name}): {exc}"
    bench_.writer.write({
        "kind": "summary", "mode": "duel", "time": time.time(), "a": infos["A"].name,
        "b": infos["B"].name, "pairs": len(good), "failed": failure,
        "ratio": ratio.ratio if ratio else None,
        "ratio_ci": [ratio.ci_low, ratio.ci_high] if ratio else None,
        "verdict": ratio.verdict if ratio else "no verdict"})
    return DuelSummary(infos["A"].name, infos["B"].name, len(good), ratio, failure)
