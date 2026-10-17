"""Run entries under the contract: process control, confinement, and rig-owned timing.

The rig owns time and the process tree:
- Every entry runs in its own session (and, by default, its own namespaces; see
  isolation.py). Its whole process group is killed when the conversation ends,
  and live entries are registered so an aborted search can reap them.
- Every output stream is bounded: --info/--verify capture, stderr logs, the
  event queue (backpressure), and the number of offsets per session.
- Events are timestamped on arrival by a reader thread that cannot die silently,
  so wall time and deadlines never depend on how fast the rig drains its queue.
"""

import contextlib
import json
import os
import queue
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from array import array
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Literal

from . import isolation
from . import protocol as proto

MAX_LINE = 16 * 1024 * 1024  # longest accepted protocol line, in bytes

DONE, ERROR, PROTOCOL, TIMEOUT, CRASH = "done", "error", "protocol", "timeout", "crash"


@dataclass(frozen=True)
class EntrySpec:
    """How to launch and confine an entry; the rig appends contract flags to argv."""

    argv: tuple[str, ...]
    cwd: Path | None = None
    env: tuple[tuple[str, str], ...] = ()
    device: int = 0                     # index into the GPUs visible to the rig
    cpus: tuple[int, ...] | Literal["auto"] | None = None  # AUTO_CPUS: per device
    isolate: bool = True
    log_dir: Path | None = None         # stderr logs; None discards them

    @property
    def label(self) -> str:
        return Path(self.argv[-1]).stem or "entry"


@dataclass(frozen=True)
class Limits:
    info_seconds: float = 30.0
    verify_seconds: float = 900.0
    setup_seconds: float = 300.0
    grace_seconds: float = 10.0
    max_output_bytes: int = 1 << 20     # per stream, --info and --verify
    max_log_bytes: int = 64 << 20       # grind stderr kept on disk
    max_offsets: int = 1_000_000        # per session: a matcher reporting everything
    queue_events: int = 10_000          # reader backpressure


DEFAULT_LIMITS = Limits()


@dataclass(frozen=True)
class GrindRecord:
    """What happened in one grind session. `offsets` holds every reported k in order."""

    outcome: str
    message: str
    offsets: array
    candidates: int | None
    setup_seconds: float
    wall_seconds: float
    log: Path | None
    started: bool = False               # the start command reached the entry


# --------------------------------------------------------------------------- processes

_LIVE: set[subprocess.Popen] = set()
_LIVE_LOCK = threading.Lock()


def _kill_group(proc: subprocess.Popen) -> None:
    """Kill the entry's whole process group, whatever state its leader is in."""
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(proc.pid, signal.SIGKILL)
    proc.wait()


def kill_all() -> None:
    """Reap every live entry (for aborts and SIGTERM)."""
    with _LIVE_LOCK:
        live = list(_LIVE)
    for proc in live:
        _kill_group(proc)


class _Launch:
    """A confined entry process; the context exit always reaps it and its home."""

    def __init__(self, spec: EntrySpec, flags: list[str], stdin: bool = False):
        self.home = Path(tempfile.mkdtemp(prefix="onionrig-entry-"))
        env = isolation.scrubbed_environment(os.environ, self.home)
        env.update(spec.env)
        env, _ = isolation.select_gpu(env, spec.device, isolation.gpu_vendor())
        argv = isolation.wrap_command([*spec.argv, *flags],
                                      isolation.entry_cpus(spec.cpus, spec.device),
                                      spec.isolate)
        try:
            self.proc = subprocess.Popen(
                argv, cwd=spec.cwd, env=env, start_new_session=True,
                stdin=subprocess.PIPE if stdin else subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except BaseException:
            shutil.rmtree(self.home, ignore_errors=True)
            raise
        with _LIVE_LOCK:
            _LIVE.add(self.proc)

    def __enter__(self) -> "_Launch":
        return self

    def __exit__(self, *exc: object) -> None:
        _kill_group(self.proc)
        with _LIVE_LOCK:
            _LIVE.discard(self.proc)
        shutil.rmtree(self.home, ignore_errors=True)


class _Capture(threading.Thread):
    """Read a stream to EOF, keeping at most `cap` bytes; flags anything beyond."""

    def __init__(self, stream: IO[bytes], cap: int):
        super().__init__(daemon=True)
        self.stream, self.cap = stream, cap
        self.data = bytearray()
        self.overflow = False
        self.start()

    def run(self) -> None:
        with contextlib.suppress(OSError, ValueError):
            while chunk := self.stream.read1(65536):
                room = self.cap - len(self.data)
                self.data += chunk[:max(room, 0)]
                self.overflow = self.overflow or len(chunk) > room

    @property
    def text(self) -> str:
        return self.data.decode("utf-8", errors="replace")


@dataclass(frozen=True)
class _Ran:
    code: int | None   # None: killed for exceeding a limit
    stdout: str
    stderr: str
    problem: str


def _run_bounded(spec: EntrySpec, flags: list[str], seconds: float, cap: int) -> _Ran:
    with _Launch(spec, flags) as launch:
        assert launch.proc.stdout is not None and launch.proc.stderr is not None
        out, err = _Capture(launch.proc.stdout, cap), _Capture(launch.proc.stderr, cap)
        deadline = time.monotonic() + seconds
        while launch.proc.poll() is None and time.monotonic() < deadline \
                and not (out.overflow or err.overflow):
            time.sleep(0.05)
        code = launch.proc.poll()
    out.join(timeout=2)
    err.join(timeout=2)
    if out.overflow or err.overflow:
        return _Ran(None, out.text, err.text, f"printed more than {cap} bytes")
    if code is None:
        return _Ran(None, out.text, err.text, f"exceeded {seconds:.0f} s")
    return _Ran(code, out.text, err.text, "")


def run_info(spec: EntrySpec, limits: Limits = DEFAULT_LIMITS) -> proto.Info:
    ran = _run_bounded(spec, ["--info"], limits.info_seconds, limits.max_output_bytes)
    if ran.problem:
        raise proto.ProtocolError(f"--info {ran.problem}")
    if ran.code != 0:
        raise proto.ProtocolError(f"--info exited {ran.code}: {proto.sanitize(ran.stderr[-300:])}")
    return proto.parse_info(ran.stdout)


@dataclass(frozen=True)
class VerifyReport:
    ok: bool
    summary: dict
    message: str


def _summary(stdout: str) -> dict | None:
    lines = [ln for ln in stdout.splitlines() if ln.strip()]
    try:
        summary = json.loads(lines[-1]) if lines else None
    except (ValueError, RecursionError):
        return None
    counts_ok = isinstance(summary, dict) and all(
        type(summary.get(k)) is int and summary[k] >= 0 for k in ("passed", "failed"))
    return summary if counts_ok else None


def run_verify(spec: EntrySpec, vectors_dir: Path, limits: Limits = DEFAULT_LIMITS) -> VerifyReport:
    ran = _run_bounded(spec, ["--verify", str(vectors_dir.resolve())], limits.verify_seconds,
                       limits.max_output_bytes)
    if ran.problem:
        return VerifyReport(False, {}, f"--verify {ran.problem}")
    summary = _summary(ran.stdout)
    if summary is None:
        return VerifyReport(False, {}, "--verify printed no valid JSON summary "
                                       f"(integer 'passed' and 'failed'): "
                                       f"{proto.sanitize(ran.stdout[-200:])}")
    ok = ran.code == 0 and summary["failed"] == 0 and summary["passed"] > 0
    return VerifyReport(ok, summary, "" if ok else f"exit {ran.code}, failed {summary['failed']}")


# --------------------------------------------------------------------------- grinding

_EOF = object()


def _offer(events: queue.Queue, item: object, closed: threading.Event) -> bool:
    """Put with backpressure; gives up once the session has ended."""
    while not closed.is_set():
        try:
            events.put(item, timeout=0.25)
            return True
        except queue.Full:
            continue
    return False


def _pump(stream: IO[bytes], events: queue.Queue, closed: threading.Event) -> None:
    """Reader thread: parse each stdout line and timestamp it on arrival.

    Whatever happens (bad bytes, hostile JSON, a closed pipe), the session hears
    about it: a parse failure becomes a ProtocolError event and EOF always follows.
    """
    try:
        while True:
            raw = stream.readline(MAX_LINE + 1)
            stamp = time.monotonic()
            if not raw:
                return
            if len(raw) > MAX_LINE:
                _offer(events, (stamp, proto.ProtocolError("protocol line too long")), closed)
                return
            line = raw.decode("utf-8", errors="replace")
            if not line.strip():
                continue
            try:
                event: object = proto.parse_event(line)
            except proto.ProtocolError as exc:
                event = exc
            if not _offer(events, (stamp, event), closed) or isinstance(event, proto.ProtocolError):
                return
    except Exception as exc:  # the reader must never die silently
        _offer(events, (time.monotonic(), proto.ProtocolError(f"stdout unreadable: {exc}")), closed)
    finally:
        _offer(events, (time.monotonic(), _EOF), closed)


def _drain_log(stream: IO[bytes], path: Path | None, cap: int) -> None:
    """Copy stderr to the log, keeping at most `cap` bytes."""
    kept = 0
    with contextlib.ExitStack() as stack:
        log = stack.enter_context(open(path, "wb")) if path else None
        with contextlib.suppress(OSError, ValueError):
            while chunk := stream.read1(65536):
                if log is not None and kept < cap:
                    log.write(chunk[:cap - kept])
                    kept += len(chunk[:cap - kept])
                    if kept >= cap:
                        log.write(b"\n[onionrig: log truncated]\n")
                        kept = cap + 1


def _log_path(spec: EntrySpec) -> Path | None:
    if spec.log_dir is None:
        return None
    spec.log_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    return spec.log_dir / f"{spec.label}-grind-{stamp}-{os.getpid()}-{time.monotonic_ns()}.log"


class _Session:
    """One grind conversation; run() returns the record and always reaps the process."""

    def __init__(self, spec: EntrySpec, limits: Limits,
                 on_progress: Callable[[int, float], None] | None):
        self.spec, self.limits, self.on_progress = spec, limits, on_progress
        self.log = _log_path(spec)
        self.offsets: array = array("q")
        self.setup = self.wall = 0.0
        self.started = False

    def record(self, outcome: str, message: str = "", candidates: int | None = None) -> GrindRecord:
        return GrindRecord(outcome, message, self.offsets, candidates, self.setup, self.wall,
                           self.log, self.started)

    def write(self, text: str) -> bool:
        try:
            assert self.proc.stdin is not None
            self.proc.stdin.write(text.encode())
            self.proc.stdin.flush()
            return True
        except (BrokenPipeError, OSError, ValueError):
            return False

    def failure(self, item: object) -> GrindRecord:
        if item is _EOF:
            try:
                code = self.proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                code = None
            return self.record(CRASH, f"closed stdout mid-session (exit code {code})")
        if isinstance(item, proto.EntryError):
            return self.record(ERROR, item.message)
        if isinstance(item, proto.ProtocolError):
            return self.record(PROTOCOL, str(item))
        return self.record(PROTOCOL, f"unexpected {type(item).__name__} event")

    def run(self, start_line: str, seconds: float | None, on_hits: Callable | None,
            stop: threading.Event | None) -> GrindRecord:
        events: queue.Queue = queue.Queue(maxsize=self.limits.queue_events)
        closed = threading.Event()
        with _Launch(self.spec, ["--grind"], stdin=True) as launch:
            self.proc = launch.proc
            assert self.proc.stdout is not None and self.proc.stderr is not None
            threading.Thread(target=_drain_log, daemon=True,
                             args=(self.proc.stderr, self.log, self.limits.max_log_bytes)).start()
            launched = time.monotonic()
            threading.Thread(target=_pump, args=(self.proc.stdout, events, closed),
                             daemon=True).start()
            try:
                return self._converse(events, launched, start_line, seconds, on_hits, stop)
            finally:
                closed.set()

    def _await_ready(self, events: queue.Queue, launched: float) -> GrindRecord | None:
        try:
            stamp, item = events.get(timeout=self.limits.setup_seconds)
        except queue.Empty:
            return self.record(TIMEOUT, f"no ready within {self.limits.setup_seconds:.0f} s")
        if not isinstance(item, proto.Ready):
            return self.failure(item)
        self.setup = stamp - launched
        return None

    def _converse(self, events, launched, start_line, seconds, on_hits, stop) -> GrindRecord:
        if (early := self._await_ready(events, launched)) is not None:
            return early
        if not self.write(start_line):
            return self.record(CRASH, "entry closed stdin before start")
        self.started = True
        started = time.monotonic()
        soft = started + seconds + self.limits.grace_seconds if seconds else None
        hard: float | None = None
        exited_at: float | None = None
        while True:
            now = time.monotonic()
            if hard is None and ((soft and now >= soft) or (stop and stop.is_set())):
                self.write(proto.STOP_COMMAND)
                hard = now + self.limits.grace_seconds
            try:
                stamp, item = events.get(timeout=0.2) if hard is None or now < hard \
                    else events.get_nowait()
            except queue.Empty:
                if hard is not None and now >= hard:  # queue drained past the deadline
                    self.wall = now - started
                    return self.record(TIMEOUT, "did not finish within the grace period after stop")
                if self.proc.poll() is not None:  # exited; a grandchild may hold stdout open
                    exited_at = exited_at or now
                    if now - exited_at > self.limits.grace_seconds:
                        return self.record(CRASH, f"exited (code {self.proc.returncode}) "
                                                  "without sending done")
                continue
            if hard is not None and stamp > hard and not isinstance(item, proto.Hits):
                self.wall = stamp - started
                return self.record(TIMEOUT, "did not finish within the grace period after stop")
            result = self._handle(item, stamp, started, on_hits)
            if result is not None:
                return result

    def _handle(self, item: object, stamp: float, started: float,
                on_hits: Callable | None) -> GrindRecord | None:
        if isinstance(item, proto.Hits):
            self.offsets.extend(item.offsets)
            if len(self.offsets) > self.limits.max_offsets:
                self.wall = stamp - started
                return self.record(PROTOCOL, f"hit flood: more than {self.limits.max_offsets:,} "
                                             "offsets in one session (a matcher reporting "
                                             "everything?)")
            if on_hits is not None:
                on_hits(item.offsets)
            return None
        if isinstance(item, proto.Progress):
            if self.on_progress is not None:
                self.on_progress(item.candidates, stamp - started)
            return None
        self.wall = stamp - started
        if not isinstance(item, proto.Done):
            return self.failure(item)
        return self._finish(item.candidates)

    def _finish(self, candidates: int) -> GrindRecord:
        assert self.proc.stdin is not None
        with contextlib.suppress(OSError):
            self.proc.stdin.close()
        try:
            code = self.proc.wait(timeout=self.limits.grace_seconds)
        except subprocess.TimeoutExpired:
            return self.record(PROTOCOL, "did not exit after done", candidates)
        if code != 0:
            return self.record(PROTOCOL, f"exited with code {code} after done", candidates)
        return self.record(DONE, "", candidates)


def run_grind(spec: EntrySpec, a0: bytes, targets: tuple[str, ...], seconds: float | None,
              limits: Limits = DEFAULT_LIMITS, *, on_hits: Callable | None = None,
              stop: threading.Event | None = None,
              on_progress: Callable[[int, float], None] | None = None) -> GrindRecord:
    """One full grind session. `on_hits` streams offsets as they arrive, `on_progress`
    receives optional (candidates, seconds) reports, and `stop` ends an open-ended run."""
    start_line = proto.start_command(a0, targets, seconds)
    return _Session(spec, limits, on_progress).run(start_line, seconds, on_hits, stop)
