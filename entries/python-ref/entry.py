#!/usr/bin/env python3
"""Reference entry for the onionrig-grind/1 contract (docs/design/CONTRACT.md).

Correct and deliberately simple: pure Python, single core, about 1e5 candidates
per second. It exists to pin down the protocol in executable form and to let
the rig be tested end to end without a GPU. It is not a competitor.

Algorithm: walk P(k) = A0 + k*(8B) for consecutive k, batch-invert the Z
coordinates (Montgomery's trick), and compare the low stream bits of each
affine y against the targets.
"""

import argparse
import json
import queue
import sys
import threading
import time
from pathlib import Path

from onionrig import onion, splitkey
from onionrig.ed25519 import field as F
from onionrig.ed25519 import point as E

CONTRACT = "onionrig-grind/1"
BATCH = 256


def send(message: dict) -> None:
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


def batch_y(points: list[E.Point]) -> list[int]:
    """Affine y for every point with one field inversion (Montgomery's trick)."""
    prefix, acc = [], 1
    for p in points:
        prefix.append(acc)
        acc = acc * p.Z % F.P
    inv = F.inv(acc)
    ys = [0] * len(points)
    for i in range(len(points) - 1, -1, -1):
        ys[i] = points[i].Y * inv % F.P * prefix[i] % F.P
        inv = inv * points[i].Z % F.P
    return ys


class Matcher:
    def __init__(self, targets: tuple[onion.Target, ...]):
        by_bits: dict[int, set[int]] = {}
        for t in targets:
            by_bits.setdefault(t.bits, set()).add(t.value)
        self.by_bits = sorted(by_bits.items())

    def __call__(self, y: int) -> bool:
        # Only y matters: prefixes cover stream bits 0..254, and the x sign is bit 255.
        head = int.from_bytes(y.to_bytes(32, "little"), "big")
        return any((head >> (256 - bits)) in values for bits, values in self.by_bits)


def grind_range(a0: E.Point, matcher: Matcher, lo: int, hi: int):
    """Yield (k, matched) for every k in [lo, hi), batch by batch."""
    point = splitkey.point_at(a0, lo)
    k = lo
    while k < hi:
        n = min(BATCH, hi - k)
        points = []
        for _ in range(n):
            points.append(point)
            point = E.add(point, splitkey.STEP)
        for i, y in enumerate(batch_y(points)):
            yield k + i, matcher(y)
        k += n


def cmd_info() -> int:
    send({"contract": CONTRACT, "name": "python-ref", "version": "1.0",
          "model": "", "device": "cpu (pure Python, 1 core)"})
    return 0


def cmd_verify(vectors_dir: Path) -> int:
    document = json.loads((vectors_dir / "grind.json").read_text())
    checks = []
    for case in (c for c in document["cases"] if c["op"] == "kat"):
        a0 = E.decode(bytes.fromhex(case["a0"]))
        matcher = Matcher(onion.normalize_targets(case["targets"]))
        found = [k for k, hit in grind_range(a0, matcher, int(case["lo"]), int(case["hi"])) if hit]
        checks.append({"name": case["name"], "ok": found == [int(k) for k in case["hits"]]})
    failed = sum(not c["ok"] for c in checks)
    send({"passed": len(checks) - failed, "failed": failed, "checks": checks})
    return 0 if failed == 0 and checks else 1


def stdin_commands(out: "queue.Queue[dict | None]") -> None:
    for line in sys.stdin:
        if line.strip():
            out.put(json.loads(line))
    out.put(None)  # EOF: the rig is gone


def cmd_grind() -> int:
    commands: queue.Queue[dict | None] = queue.Queue()
    threading.Thread(target=stdin_commands, args=(commands,), daemon=True).start()
    send({"event": "ready"})
    start = commands.get()
    if not start or start.get("cmd") != "start":
        send({"event": "error", "message": "expected a start command"})
        return 1
    try:
        a0 = E.decode(bytes.fromhex(start["a0"]))
        matcher = Matcher(onion.normalize_targets(start["targets"]))
    except (KeyError, ValueError) as exc:
        send({"event": "error", "message": f"bad start command: {exc}"})
        return 1
    seconds = start.get("seconds")
    deadline = None if seconds is None else time.monotonic() + seconds

    tested, k, pending, last_flush = 0, 0, [], time.monotonic()
    last_progress = last_flush
    while deadline is None or time.monotonic() < deadline:
        try:
            if commands.get_nowait() in (None, {"cmd": "stop"}):
                break
        except queue.Empty:
            pass
        for offset, hit in grind_range(a0, matcher, k, k + BATCH):
            if hit:
                pending.append(str(offset))
        k += BATCH
        tested += BATCH
        if pending and time.monotonic() - last_flush > 0.5:
            send({"event": "hits", "k": pending})
            pending, last_flush = [], time.monotonic()
        if time.monotonic() - last_progress > 5:
            send({"event": "progress", "candidates": str(tested)})
            last_progress = time.monotonic()
    if pending:
        send({"event": "hits", "k": pending})
    send({"event": "done", "candidates": str(tested)})
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--info", action="store_true")
    group.add_argument("--verify", type=Path, metavar="VECTORS_DIR")
    group.add_argument("--grind", action="store_true")
    args = parser.parse_args()
    if args.info:
        return cmd_info()
    if args.verify:
        return cmd_verify(args.verify)
    return cmd_grind()


if __name__ == "__main__":
    sys.exit(main())
