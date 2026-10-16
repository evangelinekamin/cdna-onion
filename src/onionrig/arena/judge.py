"""The judge's last step, on the cluster: turn the build outcome and what the rig
wrote into one result record.

Usage: python -m onionrig.arena.judge KIND OUT_DIR BUILD_OK BUILD_SECONDS BUILD_LOG RESULT
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from . import model

BENCH_FIELDS = ("rate", "rate_ci", "runs", "failed", "gpu", "eta10_median_s", "eta13_median_s")


def _tail(text: str, lines: int = model.MAX_LOG_LINES) -> str:
    return "\n".join(text.splitlines()[-lines:])


def _load_json(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _bench_summary(out_dir: Path) -> dict | None:
    for path in sorted(out_dir.glob("bench-*.jsonl"), reverse=True):
        for line in reversed(path.read_text().splitlines()):
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if isinstance(record, dict) and record.get("kind") == "summary":
                return record
    return None


def _latest_entry_log(out_dir: Path) -> str | None:
    logs = sorted((out_dir / "logs").glob("*.log"), key=lambda p: p.stat().st_mtime)
    return _tail(logs[-1].read_text(errors="replace")) if logs else None


def summarize(kind: str, out_dir: Path, build_ok: bool, build_seconds: float,
              build_log: str) -> dict:
    build = {"ok": build_ok, "seconds": round(build_seconds, 1), "log": _tail(build_log)}
    result = {"kind": kind, "status": "error", "reason": None, "check": None, "bench": None,
              "build": build, "entry_log": None}
    if not build_ok:
        return {**result, "status": "failed", "reason": "build failed"}
    if out_dir.is_dir():
        result["entry_log"] = _latest_entry_log(out_dir)
    if kind == "check":
        report = _load_json(out_dir / "check.json")
        if report is None:
            return {**result, "reason": "the rig wrote no check report"}
        conforms = report.get("conforms") is True
        return {**result, "check": report, "status": "passed" if conforms else "failed",
                "reason": None if conforms else "does not conform to onionrig-grind/1"}
    summary = _bench_summary(out_dir) if out_dir.is_dir() else None
    if summary is None:
        return {**result, "reason": "the rig wrote no benchmark summary"}
    result["bench"] = {k: summary.get(k) for k in BENCH_FIELDS}
    if summary.get("failed"):
        return {**result, "status": "failed", "reason": f"benchmark failed: {summary['failed']}"}
    if summary.get("rate") is None:
        return {**result, "status": "failed", "reason": "no countable benchmark runs"}
    return {**result, "status": "passed"}


def main(argv: list[str]) -> int:
    kind, out_dir, build_ok, build_seconds, build_log, result = argv
    log = Path(build_log).read_text(errors="replace") if Path(build_log).exists() else ""
    record = summarize(kind, Path(out_dir), build_ok == "1", float(build_seconds), log)
    Path(result).write_text(json.dumps(record, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
