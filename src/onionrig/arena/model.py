"""What crosses between the boxes and the runner, validated and bounded.

Everything a competitor's box sends is untrusted: names, records, and sizes are
checked here before the organizer's machine acts on them, and only ids matching
ID_RE ever reach a shell command.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ..rig.protocol import sanitize

KINDS = ("check", "bench")
STATUSES = ("passed", "failed", "error", "rejected")
ID_RE = re.compile(r"\d{8}T\d{6}-(check|bench)-[0-9a-f]{6}")
MAX_META_BYTES = 4096
MAX_TARBALL_BYTES = 20 * 1024 * 1024
MAX_LINE = 240
MAX_LOG_LINES = 60


@dataclass(frozen=True)
class Submission:
    box: str
    id: str
    kind: str
    created: float
    size: int


def ids_in_listing(listing: str) -> list[str]:
    """Submission ids from an outbox listing: `<id>.json` names with well-formed ids."""
    names = (line.strip() for line in listing.splitlines())
    return [name[:-5] for name in names if name.endswith(".json") and ID_RE.fullmatch(name[:-5])]


def parse_submission(box: str, meta_text: str, tarball_size: int) -> Submission:
    """A box's submission record, validated; ValueError says what is wrong."""
    if len(meta_text) > MAX_META_BYTES:
        raise ValueError("submission record too large")
    try:
        meta = json.loads(meta_text)
    except (ValueError, RecursionError):
        meta = None
    if meta is None:
        raise ValueError("submission record is not JSON")
    if not isinstance(meta, dict):
        raise ValueError("submission record is not an object")
    sid, kind, created = meta.get("id"), meta.get("kind"), meta.get("created")
    if not isinstance(sid, str) or not ID_RE.fullmatch(sid):
        raise ValueError("bad submission id")
    if kind not in KINDS or sid.split("-")[1] != kind:
        raise ValueError("kind must be check or bench, matching the id")
    if _number(created) is None:
        raise ValueError("created must be a time in seconds")
    if not 0 < tarball_size <= MAX_TARBALL_BYTES:
        raise ValueError(f"tarball must be 1 byte to {MAX_TARBALL_BYTES >> 20} MB")
    return Submission(box, sid, kind, float(created), tarball_size)


@dataclass(frozen=True)
class Quota:
    bench_per_window: int = 6
    window_s: float = 86_400.0


DEFAULT_QUOTA = Quota()


def bench_slots(accepted: Sequence[float], now: float, quota: Quota) -> tuple[int, float | None]:
    """Benchmarks left in the rolling window, and when the next one frees (None if free now)."""
    recent = sorted(t for t in accepted if now - t < quota.window_s)
    left = quota.bench_per_window - len(recent)
    if left > 0:
        return left, None
    return 0, recent[-quota.bench_per_window] + quota.window_s


def next_box(pending: Mapping[str, int], order: Sequence[str], last: str | None) -> str | None:
    """Round-robin over boxes with queued work, starting after the box served last."""
    start = order.index(last) + 1 if last in order else 0
    for i in range(len(order)):
        box = order[(start + i) % len(order)]
        if pending.get(box, 0) > 0:
            return box
    return None


def _text(value: object) -> str | None:
    return None if value is None else sanitize(value, MAX_LINE)


def _log(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    return "\n".join(sanitize(line, MAX_LINE) for line in value.splitlines()[-MAX_LOG_LINES:])


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        return None
    return float(value)


def _build(raw: object) -> dict | None:
    if not isinstance(raw, dict):
        return None
    return {"ok": raw.get("ok") is True, "seconds": _number(raw.get("seconds")),
            "log": _log(raw.get("log"))}


def _check(raw: object) -> dict | None:
    if not isinstance(raw, dict):
        return None
    checks = raw.get("checks") if isinstance(raw.get("checks"), list) else []
    return {"conforms": raw.get("conforms") is True,
            "checks": [{"name": _text(c.get("name")), "ok": c.get("ok") is True,
                        "required": c.get("required") is True, "detail": _text(c.get("detail"))}
                       for c in checks if isinstance(c, dict)]}


def _bench(raw: object) -> dict | None:
    if not isinstance(raw, dict):
        return None
    ci = raw.get("rate_ci")
    ci = [_number(x) for x in ci] if isinstance(ci, list) and len(ci) == 2 else None
    runs = raw.get("runs")
    return {"rate": _number(raw.get("rate")),
            "rate_ci": ci if ci and None not in ci else None,
            "runs": runs if isinstance(runs, int) and not isinstance(runs, bool) else None,
            "failed": _text(raw.get("failed")), "gpu": _text(raw.get("gpu")),
            "eta10_median_s": _number(raw.get("eta10_median_s")),
            "eta13_median_s": _number(raw.get("eta13_median_s"))}


def shape_for_box(raw: object) -> dict:
    """The judge's result reduced to known fields, with bounded, printable text."""
    raw = raw if isinstance(raw, dict) else {}
    return {"status": raw.get("status") if raw.get("status") in STATUSES else "error",
            "kind": raw.get("kind") if raw.get("kind") in KINDS else None,
            "reason": _text(raw.get("reason")),
            "build": _build(raw.get("build")), "check": _check(raw.get("check")),
            "bench": _bench(raw.get("bench")), "entry_log": _log(raw.get("entry_log"))}
