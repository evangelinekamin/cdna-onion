"""Conformance suite: does an entry honor the contract, before it is ever benchmarked?

Grinding checks are sized from a short calibration run, so a fast GPU entry gets
targets that yield a few thousand hits per second instead of millions. Required
checks gate benchmarking; advisory ones are reported but do not fail.
"""

import secrets
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .. import onion
from . import runner
from .plan import (
    CALIBRATION_PLAN,
    TargetPlan,
    fresh_inputs,
    invalid_start_point,
    plan_for,
    planning_rate,
)
from .protocol import sanitize
from .score import score

CHECK_HITS_PER_SECOND = 2_000.0


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    required: bool
    detail: str


@dataclass
class Context:
    spec: runner.EntrySpec
    limits: runner.Limits
    vectors: Path
    plan: TargetPlan | None = None


def _check_info(ctx: Context) -> tuple[bool, str]:
    try:
        info = runner.run_info(ctx.spec, ctx.limits)
    except Exception as exc:  # any failure here is the entry's, reported verbatim
        return False, f"{type(exc).__name__}: {sanitize(exc)}"
    return True, f"{info.name} {info.version} on {info.device or 'unreported device'}"


def _check_verify(ctx: Context) -> tuple[bool, str]:
    report = runner.run_verify(ctx.spec, ctx.vectors, ctx.limits)
    return report.ok, report.message or f"{report.summary.get('passed', 0)} self-checks passed"


def _check_calibrate(ctx: Context) -> tuple[bool, str]:
    inputs = fresh_inputs(CALIBRATION_PLAN)
    record = runner.run_grind(ctx.spec, inputs.a0, inputs.target_texts, 3.0, ctx.limits)
    try:
        rate = planning_rate(score(record, inputs.a0, inputs.targets))
    except RuntimeError as exc:
        return False, str(exc)
    ctx.plan = plan_for(rate, CHECK_HITS_PER_SECOND)
    return True, f"~{rate:,.0f} candidates/s; checks use {ctx.plan.count} target(s) of " \
                 f"{ctx.plan.chars} char(s)"


def _scored_grind(ctx: Context, seconds: float,
                  long_targets: tuple[int, ...] = ()) -> tuple[bool, str]:
    if ctx.plan is None:
        return False, "skipped: calibration failed"
    inputs = fresh_inputs(ctx.plan)
    extra = tuple(_long_target(n, inputs.target_texts) for n in long_targets)
    targets = onion.normalize_targets(inputs.target_texts + extra)
    record = runner.run_grind(ctx.spec, inputs.a0, tuple(t.text for t in targets), seconds,
                              ctx.limits)
    result = score(record, inputs.a0, targets)
    detail = (f"{result.hits} verified hits, {result.rate:,.0f} candidates/s"
              if result.valid else sanitize(result.reason))
    if result.flags:
        detail += f" [{', '.join(result.flags)}]"
    return result.valid and "self-report-inconsistent" not in result.flags, detail


def _check_grind(ctx: Context) -> tuple[bool, str]:
    return _scored_grind(ctx, 5.0)


def _long_target(length: int, avoid: tuple[str, ...]) -> str:
    """A random prefix that no short target is a prefix of (so none gets dropped)."""
    while True:
        text = "".join(secrets.choice(onion.ALPHABET) for _ in range(length))
        if not any(text.startswith(a) for a in avoid):
            return text


def _check_long_targets(ctx: Context) -> tuple[bool, str]:
    # Prefixes past 12 characters no longer fit a single 64-bit compare. These
    # never hit; the check is that the entry accepts them and reports no false
    # hits. Finding long targets is covered by the kat_long known-answer grind.
    return _scored_grind(ctx, 4.0, long_targets=(13, 20))


def _check_stop(ctx: Context) -> tuple[bool, str]:
    inputs = fresh_inputs(ctx.plan or CALIBRATION_PLAN)
    stop = threading.Event()
    timer = threading.Timer(3.0, stop.set)
    timer.start()
    try:
        record = runner.run_grind(ctx.spec, inputs.a0, inputs.target_texts, None, ctx.limits,
                                  stop=stop)
    finally:
        timer.cancel()
    detail = sanitize(record.message) or f"stopped after {record.wall_seconds:.1f} s"
    return record.outcome == runner.DONE, detail


def _check_bad_start(ctx: Context) -> tuple[bool, str]:
    record = runner.run_grind(ctx.spec, invalid_start_point(), ("ab",), 2.0, ctx.limits)
    if record.outcome == runner.ERROR:
        return True, "rejected an invalid start point"
    return False, f"expected an error event for an invalid a0, got {record.outcome}"


CHECKS: list[tuple[str, bool, Callable[[Context], tuple[bool, str]]]] = [
    ("info", True, _check_info),
    ("verify", True, _check_verify),
    ("calibrate", True, _check_calibrate),
    ("grind", True, _check_grind),
    ("long-targets", True, _check_long_targets),
    ("stop", True, _check_stop),
    ("bad-start-point", False, _check_bad_start),
]
_GATES = ("info", "verify")  # nothing else is meaningful if these fail


def run_checks(spec: runner.EntrySpec, vectors: Path,
               limits: runner.Limits = runner.DEFAULT_LIMITS,
               report: Callable[[CheckResult], None] | None = None) -> list[CheckResult]:
    ctx = Context(spec, limits, vectors)
    results = []
    for name, required, check in CHECKS:
        ok, detail = check(ctx)
        result = CheckResult(name, ok, required, detail)
        results.append(result)
        if report:
            report(result)
        if not ok and name in _GATES:
            break
    return results


def passed(results: list[CheckResult]) -> bool:
    return all(r.ok for r in results if r.required) and len(results) == len(CHECKS)
