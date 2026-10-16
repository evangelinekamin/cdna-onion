"""The arena's runner, on the organizer's machine: collect submissions from the
competitors' boxes, judge them one at a time on the cluster, deliver results.

Every event is appended to a ledger before the runner acts on it, so a runner
that restarts replays the ledger and carries on: nothing is collected twice,
and a job already running on the cluster is picked up where it was.
"""

from __future__ import annotations

import contextlib
import json
import shlex
import subprocess
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ..rig.results import JsonlWriter, read_jsonl
from . import model

OUTBOX, INBOX, SENT = "arena/outbox", "arena/inbox", "arena/sent"


class LinkError(RuntimeError):
    """A box or the cluster could not be reached; the runner retries next step."""


class BoxLink(Protocol):
    """One competitor's box, reached as its unprivileged user."""

    def listing(self) -> str: ...
    def size(self, name: str) -> int: ...
    def read(self, name: str) -> bytes: ...
    def retire(self, sid: str) -> None: ...
    def put(self, name: str, text: str) -> None: ...


class SolLink(Protocol):
    """The cluster: one directory per submission, one batch job per judgment."""

    def upload(self, sub_dir: str, data: bytes) -> None: ...
    def launch(self, sub_dir: str, kind: str) -> str: ...
    def running(self, job: str) -> bool: ...
    def result(self, sub_dir: str) -> str | None: ...


@dataclass
class Active:
    box: str
    id: str
    kind: str
    job: str


class Runner:
    def __init__(self, boxes: Mapping[str, BoxLink], sol: SolLink, data_dir: Path,
                 quota: model.Quota = model.DEFAULT_QUOTA,
                 clock: Callable[[], float] = time.time,
                 notify: Callable[[dict], None] | None = None):
        self.boxes, self.sol, self.data_dir = dict(boxes), sol, data_dir
        self.quota, self.clock, self.notify = quota, clock, notify
        self.ledger = JsonlWriter(data_dir / "ledger.jsonl")
        self.seen: set[str] = set()
        self.queues: dict[str, deque[tuple[str, str]]] = {box: deque() for box in self.boxes}
        self.bench_times: dict[str, list[float]] = {box: [] for box in self.boxes}
        self.undelivered: dict[str, tuple[str, str]] = {}  # id -> (box, result text)
        self.posted_quota: dict[str, str] = {}
        self.active: Active | None = None
        self.last_box: str | None = None
        self._replay()

    # --- state ------------------------------------------------------------------

    def _replay(self) -> None:
        if not self.ledger.path.exists():
            return
        for e in read_jsonl([self.ledger.path]):
            box, sid, event = e.get("box"), e.get("id"), e.get("event")
            if box not in self.boxes:
                continue
            if event in ("received", "rejected"):
                self.seen.add(sid)
            elif event == "queued":
                self.queues[box].append((sid, e["kind"]))
                if e["kind"] == "bench":
                    self.bench_times[box].append(e["time"])
            elif event == "launched":
                self.queues[box] = deque(item for item in self.queues[box] if item[0] != sid)
                self.active, self.last_box = Active(box, sid, e["kind"], e["job"]), box
            elif event == "finished":
                if self.active is not None and self.active.id == sid:
                    self.active = None
                saved = self.data_dir / box / f"{sid}.result.json"
                self.undelivered[sid] = (box, saved.read_text())
            elif event == "delivered":
                self.undelivered.pop(sid, None)

    def _log(self, event: str, box: str, sid: str, **fields: object) -> None:
        record = {"event": event, "box": box, "id": sid, "time": self.clock(), **fields}
        self.ledger.write(record)
        if self.notify is not None:
            self.notify(record)

    # --- the loop -----------------------------------------------------------------

    def step(self) -> None:
        """One pass: collect, finish the running judgment if done, start the next."""
        self.collect()
        if self.active is not None:
            try:
                done = not self.sol.running(self.active.job)
            except LinkError:
                done = False
            if done:
                self._finish(self.active)
                self.active = None
        if self.active is None:
            self._launch_next()
        self._deliver_pending()

    def serve(self, interval: float = 15.0) -> None:
        while True:
            self.step()
            time.sleep(interval)

    # --- collecting -----------------------------------------------------------------

    def collect(self) -> None:
        for box, link in self.boxes.items():
            try:
                listing = link.listing()
            except LinkError:
                continue
            for sid in model.ids_in_listing(listing):
                if sid not in self.seen:
                    self._receive(box, link, sid)
            self._post_quota(box, link)

    def _receive(self, box: str, link: BoxLink, sid: str) -> None:
        try:
            meta_size, size = link.size(f"{sid}.json"), link.size(f"{sid}.tar.gz")
            if meta_size > model.MAX_META_BYTES:
                raise ValueError("submission record too large")
            meta = link.read(f"{sid}.json").decode("utf-8", "replace")
            sub = model.parse_submission(box, meta, size)
            data = link.read(f"{sid}.tar.gz")
        except ValueError as exc:
            self._reject(box, link, sid, str(exc))
            return
        except LinkError:
            return  # try again next step
        archive = self.data_dir / box
        archive.mkdir(parents=True, exist_ok=True)
        (archive / f"{sid}.tar.gz").write_bytes(data)
        self.seen.add(sid)
        self._log("received", box, sid, kind=sub.kind, size=len(data), created=sub.created)
        self._retire(link, sid)
        if sub.kind == "bench":
            left, next_at = model.bench_slots(self.bench_times[box], self.clock(), self.quota)
            if left == 0:
                when = time.strftime("%H:%M UTC", time.gmtime(next_at))
                self._reject(box, link, sid, f"bench quota reached ({self.quota.bench_per_window} "
                             f"per {self.quota.window_s / 3600:g} h); next slot at {when}")
                return
            self.bench_times[box].append(self.clock())
        self.queues[box].append((sid, sub.kind))
        self._log("queued", box, sid, kind=sub.kind)
        ahead = sum(len(q) for q in self.queues.values()) - 1 + (self.active is not None)
        self._put(link, f"{sid}.status", f"queued: {ahead} ahead of it")

    def _reject(self, box: str, link: BoxLink, sid: str, reason: str) -> None:
        self.seen.add(sid)
        self._log("rejected", box, sid, reason=reason)
        self._retire(link, sid)
        shaped = model.shape_for_box({"status": "rejected", "reason": reason})
        self._put(link, f"{sid}.json", json.dumps({"id": sid, "box": box, **shaped}, indent=1))

    # --- judging -----------------------------------------------------------------

    def _launch_next(self) -> None:
        box = model.next_box({b: len(q) for b, q in self.queues.items()}, list(self.boxes),
                             self.last_box)
        if box is None:
            return
        sid, kind = self.queues[box][0]
        sub_dir = f"{box}/{sid}"
        try:
            self.sol.upload(sub_dir, (self.data_dir / box / f"{sid}.tar.gz").read_bytes())
            job = self.sol.launch(sub_dir, kind)
        except LinkError:
            return  # the cluster is unreachable; the submission stays first in line
        self.queues[box].popleft()
        self.active, self.last_box = Active(box, sid, kind, job), box
        self._log("launched", box, sid, kind=kind, job=job)
        since = time.strftime("%H:%M UTC", time.gmtime(self.clock()))
        self._put(self.boxes[box], f"{sid}.status", f"running on the MI210 since {since}")

    def _finish(self, a: Active) -> None:
        try:
            text = self.sol.result(f"{a.box}/{a.id}")
            raw = json.loads(text) if text else None
        except (LinkError, ValueError):
            raw = None
        shaped = model.shape_for_box(raw) if raw is not None else model.shape_for_box(
            {"status": "error", "reason": "the judge produced no result"})
        result = {"id": a.id, "box": a.box, **shaped, "kind": a.kind, "finished": self.clock()}
        text = json.dumps(result, indent=1)
        (self.data_dir / a.box / f"{a.id}.result.json").write_text(text)
        rate = (result["bench"] or {}).get("rate")
        self._log("finished", a.box, a.id, status=result["status"], rate=rate)
        self.undelivered[a.id] = (a.box, text)

    def _deliver_pending(self) -> None:
        for sid, (box, text) in list(self.undelivered.items()):
            if self._put(self.boxes[box], f"{sid}.json", text):
                self.undelivered.pop(sid)
                self._log("delivered", box, sid)

    # --- small helpers -----------------------------------------------------------

    def _post_quota(self, box: str, link: BoxLink) -> None:
        left, next_at = model.bench_slots(self.bench_times[box], self.clock(), self.quota)
        text = json.dumps({"bench_left": left, "bench_per_window": self.quota.bench_per_window,
                           "window_hours": self.quota.window_s / 3600, "next_bench_at": next_at})
        if self.posted_quota.get(box) != text and self._put(link, "quota.json", text):
            self.posted_quota[box] = text

    @staticmethod
    def _put(link: BoxLink, name: str, text: str) -> bool:
        try:
            link.put(name, text)
        except LinkError:
            return False
        return True

    @staticmethod
    def _retire(link: BoxLink, sid: str) -> None:
        with contextlib.suppress(LinkError):  # the ledger marks it seen; leftovers are harmless
            link.retire(sid)


def summarize_ledger(events: list[dict], now: float, quota: model.Quota) -> dict[str, dict]:
    """Per box: submissions by outcome, benchmarks left, best rate, and what is running."""
    boxes: dict[str, dict] = {}
    ids: dict[str, set] = {}
    bench_times: dict[str, list[float]] = {}
    for e in events:
        box, sid, event = e.get("box"), e.get("id"), e.get("event")
        s = boxes.setdefault(box, {"submitted": 0, "passed": 0, "failed": 0, "rejected": 0,
                                   "pending": 0, "bench_left": 0, "best_rate": None,
                                   "running": None})
        seen = ids.setdefault(box, set())
        if event in ("received", "rejected") and sid not in seen:
            seen.add(sid)
            s["submitted"] += 1
        if event == "rejected":
            s["rejected"] += 1
        elif event == "queued" and e.get("kind") == "bench":
            bench_times.setdefault(box, []).append(e["time"])
        elif event == "launched":
            s["running"] = sid
        elif event == "finished":
            s["running"] = None if s["running"] == sid else s["running"]
            s["passed" if e.get("status") == "passed" else "failed"] += 1
            rate = e.get("rate")
            if e.get("status") == "passed" and isinstance(rate, int | float):
                s["best_rate"] = max(rate, s["best_rate"] or 0.0)
    for box, s in boxes.items():
        s["pending"] = s["submitted"] - s["passed"] - s["failed"] - s["rejected"]
        s["bench_left"] = model.bench_slots(bench_times.get(box, []), now, quota)[0]
    return boxes


# --- the real links, over SSH --------------------------------------------------------

def _ssh(host: str, command: str, stdin: bytes | None = None, timeout: float = 120.0) -> bytes:
    try:
        argv = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", host, command]
        done = subprocess.run(argv, input=stdin, capture_output=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise LinkError(f"{host}: {exc}") from exc
    if done.returncode != 0:
        detail = done.stderr.decode(errors="replace")[:200]
        raise LinkError(f"{host}: exit {done.returncode}: {detail}")
    return done.stdout


class SshBox:
    """A box as its `agent` user. Ids are validated by the caller; names are quoted anyway."""

    def __init__(self, host: str):
        self.host = host

    def listing(self) -> str:
        # Regular files only: no symlinks into the rest of the box, no devices.
        command = f"mkdir -p {OUTBOX} && find {OUTBOX} -maxdepth 1 -type f -printf '%f\\n'"
        return _ssh(self.host, command).decode("utf-8", "replace")

    def size(self, name: str) -> int:
        path = shlex.quote(f"{OUTBOX}/{name}")
        out = _ssh(self.host, f"test -f {path} && test ! -L {path} && stat -c %s {path}")
        return int(out.strip())

    def read(self, name: str) -> bytes:
        limit = model.MAX_TARBALL_BYTES + 1
        return _ssh(self.host, f"head -c {limit} {shlex.quote(f'{OUTBOX}/{name}')}", timeout=600)

    def retire(self, sid: str) -> None:
        files = " ".join(shlex.quote(f"{OUTBOX}/{sid}{ext}") for ext in (".json", ".tar.gz"))
        _ssh(self.host, f"mkdir -p {SENT} && mv -f {files} {SENT}/")

    def put(self, name: str, text: str) -> None:
        final, tmp = shlex.quote(f"{INBOX}/{name}"), shlex.quote(f"{INBOX}/.{name}.tmp")
        _ssh(self.host, f"mkdir -p {INBOX} && cat > {tmp} && mv -f {tmp} {final}", text.encode())


class SshSol:
    """The cluster: submissions under scratch, judged by tools/arena/judge.sbatch."""

    def __init__(self, host: str = "sol"):
        self.host = host
        self.root = _ssh(host, 'echo "/scratch/$USER/onionrig-arena/subs"').decode().strip()

    def _dir(self, sub_dir: str) -> str:
        return shlex.quote(f"{self.root}/{sub_dir}")

    def upload(self, sub_dir: str, data: bytes) -> None:
        d = self._dir(sub_dir)
        _ssh(self.host, f"mkdir -p {d} && cat > {d}/sub.tar.gz", data, timeout=600)

    def launch(self, sub_dir: str, kind: str) -> str:
        d = self._dir(sub_dir)
        out = _ssh(self.host, f"sbatch --parsable --output={d}/judge-%j.log "
                              f"~/onionrig/tools/arena/judge.sbatch {d} {shlex.quote(kind)}")
        return out.decode().strip().split(";")[0]

    def running(self, job: str) -> bool:
        return bool(_ssh(self.host, f"squeue -h -j {int(job)} -o %T").strip())

    def result(self, sub_dir: str) -> str | None:
        try:
            text = _ssh(self.host, f"cat {self._dir(sub_dir)}/result.json")
        except LinkError:
            return None
        return text.decode("utf-8", "replace")
