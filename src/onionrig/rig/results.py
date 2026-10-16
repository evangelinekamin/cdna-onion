"""Streaming results to disk and turning them into a leaderboard.

Every record is appended as one JSON line and flushed + fsynced before the rig
moves on: a session that dies at 90% keeps 90% of its data, and progress can be
watched with `tail -f` without attaching to anything.
"""

import json
import math
import os
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .protocol import sanitize

ADDRESS_ALPHABET = 32


def _finite(value: Any) -> Any:
    """JSON-safe copy: inf and NaN become None, so standard tools (jq) can read the log."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_finite(v) for v in value]
    return value


class JsonlWriter:
    """Append-only JSON Lines file; each write is durable before it returns."""

    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: dict[str, Any]) -> None:
        self.write_many([record])

    def write_many(self, records: list[dict[str, Any]]) -> None:
        """Append several records durably with a single fsync."""
        if not records:
            return
        text = "".join(json.dumps(_finite(r), sort_keys=True, default=str, allow_nan=False)
                       + "\n" for r in records)
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())


def read_jsonl(paths: Iterable[Path]) -> list[dict[str, Any]]:
    """All records from the given files; a torn final line (crash mid-write) is skipped."""
    records = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
            except (ValueError, RecursionError):
                continue
            if isinstance(record, dict):
                records.append(record)
    return records


def expected_seconds(rate: float, chars: int, probability: float = 0.5) -> float:
    """Time until a `chars`-character prefix is found with the given probability.

    The search is memoryless, so P(found by t) = 1 - exp(-t * rate / 32^chars).
    probability=0.5 gives the median (0.693x the mean).
    """
    if rate <= 0:
        return math.inf
    return ADDRESS_ALPHABET**chars * -math.log1p(-probability) / rate


def human_duration(seconds: float) -> str:
    if math.isinf(seconds):
        return "never"
    for unit, size in (("years", 365.25 * 86400), ("days", 86400), ("h", 3600), ("min", 60)):
        if seconds >= size:
            return f"{seconds / size:,.1f} {unit}"
    return f"{seconds:.1f} s"


def human_rate(rate: float) -> str:
    for unit, size in (("G/s", 1e9), ("M/s", 1e6), ("k/s", 1e3)):
        if rate >= size:
            return f"{rate / size:,.2f} {unit}"
    return f"{rate:,.0f} /s"


def _rate(value: Any) -> float | None:
    return float(value) if isinstance(value, int | float) and math.isfinite(value) \
        and value > 0 else None


def _weight(summary: dict[str, Any]) -> int:
    runs = summary.get("runs")
    return runs if type(runs) is int and runs > 0 else 1


def leaderboard(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per (entry, version, gpu), from bench summaries, fastest first.

    Summaries carry the bench's own decisions (counted runs, failures), so the
    board never re-derives anything from raw runs. Any failed benchmark of an
    (entry, version) on a GPU disqualifies it there.
    """
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        if r.get("kind") == "summary" and r.get("mode") == "bench":
            key = (str(r.get("entry", "")), str(r.get("version", "")), str(r.get("gpu", "")))
            groups[key].append(r)
    rows = []
    for (entry, version, gpu), summaries in groups.items():
        if any(s.get("failed") for s in summaries):
            continue
        good = [(rate, s) for s in summaries if (rate := _rate(s.get("rate"))) is not None]
        if not good:
            continue
        runs = sum(_weight(s) for _, s in good)
        mean = sum(rate * _weight(s) for rate, s in good) / runs
        rows.append({"entry": entry, "version": version, "gpu": gpu,
                     "model": str(good[-1][1].get("model", "")), "sessions": len(good),
                     "runs": runs, "rate": mean,
                     "eta13_median_s": expected_seconds(mean, 13),
                     "eta10_median_s": expected_seconds(mean, 10)})
    return sorted(rows, key=lambda row: -row["rate"])


def format_leaderboard(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "no valid runs yet"
    header = f"{'#':>2}  {'entry':<24} {'gpu':<22} {'runs':>4}  {'rate':>12}  " \
             f"{'10-char median':>15}  {'13-char median':>15}"
    lines = [header, "-" * len(header)]
    for i, row in enumerate(rows, 1):
        lines.append(
            f"{i:>2}  {sanitize(row['entry'], 24):<24} {sanitize(row['gpu'], 22):<22} "
            f"{row['runs']:>4}  "
            f"{human_rate(row['rate']):>12}  {human_duration(row['eta10_median_s']):>15}  "
            f"{human_duration(row['eta13_median_s']):>15}")
    return "\n".join(lines)
