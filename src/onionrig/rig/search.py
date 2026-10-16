"""Real vanity searches: open-ended, one session per GPU, verified hits streamed to disk.

The GPU host only ever sees public start points and emits public offsets. Start
points are handed out at most once across restarts (keyed by the point itself),
and only consumed once a session actually started; a device whose entry keeps
failing is retired instead of burning through the pool.
"""

import json
import math
import os
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

from .. import onion
from . import runner
from .protocol import sanitize
from .results import JsonlWriter
from .verify import HitVerifier

MAX_CONSECUTIVE_FAILURES = 3


def _atomic_write(path: Path, text: str) -> None:
    """Write via a temp file and rename, so a crash never leaves a torn file."""
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    except BaseException:
        Path(temp).unlink(missing_ok=True)
        raise
    dir_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


class PointPool:
    """Hands out each start point at most once, across restarts of the whole search."""

    def __init__(self, points: list[tuple[int, bytes]], used_file: Path):
        self.used_file = used_file
        used = set(json.loads(used_file.read_text())) if used_file.exists() else set()
        self._free = [(i, a0) for i, a0 in points if a0.hex() not in used]
        self._used: set[str] = used
        self._lock = threading.Lock()

    def take(self) -> tuple[int, bytes] | None:
        with self._lock:
            return self._free.pop(0) if self._free else None

    def consume(self, a0: bytes) -> None:
        """Record a point as walked; persisted before the caller continues."""
        with self._lock:
            self._used.add(a0.hex())
            self.used_file.parent.mkdir(parents=True, exist_ok=True)
            _atomic_write(self.used_file, json.dumps(sorted(self._used)))

    def give_back(self, point: tuple[int, bytes]) -> None:
        with self._lock:
            self._free.insert(0, point)

    @property
    def remaining(self) -> int:
        return len(self._free)


@dataclass(frozen=True)
class SearchConfig:
    targets: tuple[onion.Target, ...]
    devices: tuple[int, ...]
    max_hits: int | None = None
    restart: bool = True


class Search:
    def __init__(self, spec: runner.EntrySpec, pool: PointPool, config: SearchConfig,
                 writer: JsonlWriter, limits: runner.Limits = runner.DEFAULT_LIMITS,
                 progress: Callable[[str], None] | None = None):
        self.spec, self.pool, self.config, self.writer = spec, pool, config, writer
        self.limits = limits
        self.progress = progress or (lambda line: print(line, file=sys.stderr, flush=True))
        self.stop = threading.Event()
        self.hits = 0
        self._lock = threading.Lock()
        self.p_match = float(onion.match_probability(config.targets))

    def _record(self, *records: dict) -> None:
        """Append to the log; a failing disk is reported but never kills the search."""
        try:
            self.writer.write_many(list(records))
        except OSError as exc:
            self.progress(f"WARNING: could not write {self.writer.path}: {exc}")

    def _on_hits(self, device: int, point: tuple[int, bytes], verifier: HitVerifier,
                 seen: set[int], offsets: tuple[int, ...]) -> None:
        """Verify one batch from the entry; its hits become durable with one fsync."""
        point_id, a0 = point
        found: list[dict] = []
        try:
            for k in offsets:
                if k in seen:
                    continue  # duplicates count once
                seen.add(k)
                pubkey = verifier.candidate(k)
                if not verifier.matches_pubkey(pubkey):
                    self._record({"kind": "bad-hit", "time": time.time(), "device": device,
                                  "point": point_id, "a0_pub": a0.hex(), "k": str(k)})
                    raise RuntimeError(f"device {device}: entry reported a non-matching "
                                       f"offset {k}")
                address = onion.address_from_pubkey(pubkey)
                found.append({"kind": "hit", "time": time.time(), "device": device,
                              "point": point_id, "a0_pub": a0.hex(), "k": str(k),
                              "address": address})
        finally:
            self._report(found)

    def _report(self, found: list[dict]) -> None:
        # Shown before the disk write, so a full disk cannot lose a verified hit.
        for hit in found[:20]:
            self.progress(f"HIT  {hit['address']}.onion  (point {hit['point']}, k={hit['k']})")
        if len(found) > 20:
            self.progress(f"... and {len(found) - 20} more hits in this batch (see the log)")
        self._record(*found)
        with self._lock:
            self.hits += len(found)
            if self.config.max_hits and self.hits >= self.config.max_hits:
                self.stop.set()

    def _on_progress(self, device: int, candidates: int, seconds: float) -> None:
        if seconds > 0 and candidates > 0:
            rate = candidates / seconds
            median = math.log(2) / (self.p_match * rate)  # memoryless search
            self.progress(f"gpu {device}: {rate / 1e6:,.1f} M/s (self-reported); "
                          f"median wait for a hit ~{median / 3600:,.1f} h on this GPU")

    def _session(self, spec: runner.EntrySpec, device: int,
                 point: tuple[int, bytes]) -> runner.GrindRecord:
        verifier = HitVerifier(point[1], self.config.targets)
        seen: set[int] = set()
        texts = tuple(t.text for t in self.config.targets)
        self._record({"kind": "session-start", "time": time.time(), "device": device,
                      "point": point[0], "a0_pub": point[1].hex()})
        return runner.run_grind(
            spec, point[1], texts, None, self.limits, stop=self.stop,
            on_hits=lambda ks: self._on_hits(device, point, verifier, seen, ks),
            on_progress=lambda c, s: self._on_progress(device, c, s))

    def _device_loop(self, device: int) -> None:
        spec = replace(self.spec, device=device)
        failures = 0
        while not self.stop.is_set():
            point = self.pool.take()
            if point is None:
                self._record({"kind": "exhausted", "time": time.time(), "device": device})
                return
            try:
                record = self._session(spec, device, point)
            except RuntimeError as exc:  # a bad hit: this entry cannot be trusted
                self.pool.consume(point[1])
                self._record({"kind": "device-stopped", "device": device,
                              "reason": sanitize(exc)})
                return
            if record.started:
                self.pool.consume(point[1])
            else:
                self.pool.give_back(point)  # never walked; keep it for later
            self._record({"kind": "session-end", "time": time.time(), "device": device,
                          "point": point[0], "outcome": record.outcome,
                          "message": sanitize(record.message), "wall_seconds": record.wall_seconds,
                          "candidates": None if record.candidates is None
                          else str(record.candidates)})
            failures = 0 if record.outcome == runner.DONE else failures + 1
            if failures and (not self.config.restart or failures >= MAX_CONSECUTIVE_FAILURES):
                self._record({"kind": "device-stopped", "device": device,
                              "reason": f"{failures} consecutive failed session(s): "
                                        f"{sanitize(record.message)}"})
                return

    def run(self) -> int:
        threads = [threading.Thread(target=self._device_loop, args=(d,), daemon=True)
                   for d in self.config.devices]
        for t in threads:
            t.start()
        for t in threads:
            while t.is_alive():
                t.join(timeout=0.5)
        return self.hits
