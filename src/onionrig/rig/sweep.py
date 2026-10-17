"""Tuning sweeps: interleaved, self-timed runs of an entry's builds on one GPU.

For choosing a build, not for scoring one. Each cell runs one build's own
development benchmark (`--bench SECONDS TARGETS CHARS`: grind a random start
point, then print one JSON line with the self-timed rate and the hit counts)
straight after the previous cell, so every build runs in the thermal steady
state a long search runs in. Round r rotates the order by r, so over as many
rounds as builds each build runs once in every slot. The first build is the
base, and every other build is compared with the base's cell of the same round,
so drift between rounds cancels. Every cell is on disk before the next starts.

A build's speed means nothing unless it does the work it claims: each cell's
verified hits are checked against the hits its candidate count predicts.
"""

import json
import math
import platform
import re
import shlex
import statistics
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, replace
from typing import Any

from . import runner, stats
from .protocol import sanitize
from .results import JsonlWriter

Z_LIMIT = 5.0             # verified hits further than this many sigma from the claim: wrong
SETUP_ALLOWANCE_S = 180.0  # per cell, on top of its seconds: start-up, tables, memory
_NAME = re.compile(r"[A-Za-z0-9_.+-]+")


@dataclass(frozen=True)
class Variant:
    name: str
    argv: tuple[str, ...]


def parse_variant(text: str) -> Variant:
    """'NAME=COMMAND' -> Variant; raises ValueError with the reason."""
    name, sep, command = text.partition("=")
    if not sep or not _NAME.fullmatch(name):
        raise ValueError(f"expected NAME=COMMAND with a name of letters, digits and ._+-: "
                         f"{text!r}")
    argv = tuple(shlex.split(command))
    if not argv:
        raise ValueError(f"{name}: empty command")
    return Variant(name, argv)


@dataclass(frozen=True)
class SweepConfig:
    rounds: int = 4
    seconds: float = 30.0
    targets: int = 1
    chars: int = 5
    warmup: float = 0.0  # seconds of the base before round 0, not counted


def cell_order(names: Sequence[str], rounds: int) -> list[tuple[int, str]]:
    """(round, name) in run order; round r starts at names[r mod len(names)]."""
    n = len(names)
    return [(r, names[(r + i) % n]) for r in range(rounds) for i in range(n)]


def run_cell(spec: runner.EntrySpec, config: SweepConfig,
             seconds: float | None = None) -> dict[str, Any]:
    """One self-timed benchmark of the build in `spec` (config.seconds long unless
    given): its JSON fields plus the proof-of-work z score, or an error."""
    seconds = config.seconds if seconds is None else seconds
    flags = ["--bench", f"{seconds:g}", str(config.targets), str(config.chars)]
    started = time.monotonic()
    ran = runner.run_bounded(spec, flags, seconds + SETUP_ALLOWANCE_S,
                             runner.DEFAULT_LIMITS.max_output_bytes)
    wall = time.monotonic() - started
    if ran.problem or ran.code != 0:
        why = ran.problem or f"exit {ran.code}"
        return {"wall_seconds": wall, "error": f"{why}: {sanitize(ran.stderr[-300:])}"}
    fields: dict[str, Any] = {}
    for line in ran.stdout.splitlines():
        try:
            parsed = json.loads(line)
        except ValueError:
            continue
        if isinstance(parsed, dict):
            fields.update(parsed)  # a profiling build's {"prof": [...]} line, then the rate
    numbers = [fields.get(k) for k in ("rate", "candidates", "verified")]
    if not all(isinstance(v, int | float) and math.isfinite(v) for v in numbers):
        return {"wall_seconds": wall,
                "error": f"no benchmark JSON in its output: {sanitize(ran.stdout[-200:])}"}
    p_match = config.targets / 32**config.chars
    z = stats.self_report_z(int(fields["verified"]), int(fields["candidates"]), p_match)
    return {**fields, "wall_seconds": wall, "z": z}


@dataclass(frozen=True)
class Row:
    """One build's summary over a sweep."""

    name: str
    command: str
    base: bool
    cells: int                 # cells that produced a rate
    errors: int                # cells that did not
    rate: float                # mean self-timed rate over its cells (nan if none)
    ratio: stats.Ratio | None  # against the base, paired by round; None for the base
    worst_z: float             # largest |z| over its cells


def _abs_z(record: dict[str, Any]) -> float:
    z = record.get("z")
    return abs(z) if isinstance(z, int | float) else math.inf  # inf was stored as null


def summarize(records: Iterable[dict[str, Any]]) -> list[Row]:
    """Rows in the order the builds were listed. The base is the session's, or the
    first cell's build when there is no session record."""
    records = list(records)
    cells = [r for r in records if r.get("kind") == "cell"]
    if not cells:
        return []
    session = next((r for r in records if r.get("kind") == "session"), None)
    base = session["base"] if session else cells[0]["variant"]
    rates = {(c["round"], c["variant"]): c["rate"] for c in cells if "error" not in c}
    rows = []
    for name in dict.fromkeys(c["variant"] for c in cells):
        mine = [c for c in cells if c["variant"] == name]
        good = [c for c in mine if "error" not in c]
        paired = sorted({c["round"] for c in good if (c["round"], base) in rates})
        ratio = None
        if name != base and paired:
            ratio = stats.paired_ratio([rates[(r, base)] for r in paired],
                                       [rates[(r, name)] for r in paired])
        rows.append(Row(
            name=name, command=mine[0]["command"], base=name == base, cells=len(good),
            errors=len(mine) - len(good),
            rate=statistics.fmean(c["rate"] for c in good) if good else math.nan,
            ratio=ratio, worst_z=max((_abs_z(c) for c in good), default=0.0)))
    return rows


def pick(rows: Iterable[Row]) -> str:
    """The pre-registered rule: of the builds whose 95% interval against the base
    lies above 1, that never failed a cell or the proof of work, and that are not
    the base's own command under another name, the one with the highest ratio;
    otherwise the base."""
    rows = list(rows)
    base = next(r for r in rows if r.base)
    eligible = [r for r in rows if not r.base and r.command != base.command and not r.errors
                and r.worst_z <= Z_LIMIT and r.ratio is not None and r.ratio.ci_low > 1]
    best = max(eligible, key=lambda r: r.ratio.ratio if r.ratio else 0.0, default=base)
    return best.name


def format_cell(record: dict[str, Any]) -> str:
    head = f"round {record['round']} {record['variant']}:"
    if "error" in record:
        return f"{head} ERROR {record['error']}"
    notes = f"proof of work z {record['z']:+.1f}"
    prof = record.get("prof")
    if isinstance(prof, list) and prof and isinstance(prof[0], int | float):
        notes += f"; effective clock {prof[0]:.0f} MHz"  # a profiling build's first figure
    return f"{head} {record['rate'] / 1e9:.3f} G/s ({notes})"


def format_rows(rows: Sequence[Row]) -> str:
    lines = [f"{'build':<16}{'cells':>6}{'G/s':>10}  {'vs base (95% CI)':<28}{'|z| max':>8}"
             f"{'errors':>7}"]
    for r in rows:
        if r.ratio is None:
            versus = "base" if r.base else "-"
        else:
            versus = f"{r.ratio.ratio:.4f} ({r.ratio.ci_low:.4f}-{r.ratio.ci_high:.4f})"
        lines.append(f"{r.name:<16}{r.cells:>6}{r.rate / 1e9:>10.3f}  {versus:<28}"
                     f"{r.worst_z:>8.1f}{r.errors:>7}")
    lines.append(f"pick: {pick(rows)}")
    return "\n".join(lines)


def run_sweep(variants: Sequence[Variant], spec: runner.EntrySpec, config: SweepConfig,
              writer: JsonlWriter, progress: Callable[[str], None]) -> list[Row]:
    """Run every cell, writing each as it lands; returns the summary rows."""
    writer.write({"kind": "session", "mode": "sweep", "time": time.time(),
                  "base": variants[0].name,
                  "variants": {v.name: shlex.join(v.argv) for v in variants},
                  **asdict(config), "device": spec.device, "isolated": spec.isolate,
                  "host": platform.node()})
    if config.warmup > 0:  # bring the GPU to the temperature the cells will run at
        writer.write({"kind": "warmup", "time": time.time(), "variant": variants[0].name,
                      **run_cell(replace(spec, argv=variants[0].argv), config, config.warmup)})
    by_name = {v.name: v for v in variants}
    records = []
    for round_, name in cell_order(list(by_name), config.rounds):
        variant = by_name[name]
        record = {"kind": "cell", "time": time.time(), "round": round_, "variant": name,
                  "command": shlex.join(variant.argv),
                  **run_cell(replace(spec, argv=variant.argv), config)}
        writer.write(record)
        records.append(record)
        progress(format_cell(record))
    rows = summarize(records)
    writer.write({"kind": "summary", "mode": "sweep", "time": time.time(),
                  "pick": pick(rows), "rows": [asdict(r) for r in rows]})
    return rows
