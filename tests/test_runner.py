"""Process control and the grind session, against the reference and hostile entries."""

import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from onionrig.ed25519 import point as E
from onionrig.rig import isolation, runner
from onionrig.rig.protocol import ProtocolError

ROOT = Path(__file__).resolve().parents[1]
ISOLATED = isolation.isolation_available()
REFERENCE = runner.EntrySpec((sys.executable, str(ROOT / "entries/python-ref/entry.py")),
                             isolate=ISOLATED)
FAST = runner.Limits(setup_seconds=10, grace_seconds=1.5)
A0 = E.encode(E.scalar_mult(424242, E.BASE))
ORPHAN = "sleep 3017"


def mock(behavior: str, **kwargs) -> runner.EntrySpec:
    argv = (sys.executable, str(ROOT / "tests/fixtures/mock_entry.py"), "--behavior", behavior)
    return runner.EntrySpec(argv, **{"isolate": ISOLATED, **kwargs})


def leftovers() -> int:
    found = subprocess.run(["pgrep", "-f", ORPHAN], capture_output=True, text=True)
    return len(found.stdout.split())


@pytest.fixture(autouse=True)
def no_leaked_processes_between_tests():
    yield
    subprocess.run(["pkill", "-f", ORPHAN], capture_output=True)


def test_run_info():
    info = runner.run_info(REFERENCE)
    assert info.name == "python-ref" and info.version


def test_run_verify(tmp_path):
    assert runner.run_verify(REFERENCE, ROOT / "vectors").ok
    empty = tmp_path / "empty"
    empty.mkdir()
    report = runner.run_verify(REFERENCE, empty)
    assert not report.ok and report.message


def test_honest_grind(tmp_path):
    spec = runner.EntrySpec(REFERENCE.argv, isolate=ISOLATED, log_dir=tmp_path)
    streamed: list[int] = []
    record = runner.run_grind(spec, A0, ("ab", "q"), 1.5, FAST, on_hits=streamed.extend)
    assert record.outcome == runner.DONE, record.message
    assert record.started and len(record.offsets) > 100 and list(record.offsets) == streamed
    assert record.candidates and record.candidates > 1000
    assert 1.4 < record.wall_seconds < 4.0
    assert record.log is not None and record.log.exists()


def test_pinned_grind():
    spec = runner.EntrySpec(REFERENCE.argv, cpus=(0,), isolate=ISOLATED)
    assert runner.run_grind(spec, A0, ("ab",), 0.5, FAST).outcome == runner.DONE


def test_open_ended_grind_stops_on_request():
    stop = threading.Event()
    threading.Timer(1.0, stop.set).start()
    record = runner.run_grind(REFERENCE, A0, ("ab",), None, FAST, stop=stop)
    assert record.outcome == runner.DONE
    assert 0.9 < record.wall_seconds < 4.0


@pytest.mark.parametrize("behavior,outcome,fragment", [
    ("silent", runner.TIMEOUT, "no ready"),
    ("garbage", runner.PROTOCOL, "not JSON"),
    ("crash", runner.CRASH, "exit code 3"),
    ("error", runner.ERROR, "deliberate"),
    ("overrun", runner.TIMEOUT, "grace"),
    ("linger", runner.PROTOCOL, "did not exit"),
    ("bad_exit", runner.PROTOCOL, "code 5"),
])
def test_misbehaving_entries_are_caught(behavior, outcome, fragment):
    limits = runner.Limits(setup_seconds=2, grace_seconds=1)
    record = runner.run_grind(mock(behavior), A0, ("ab",), 0.5, limits)
    assert record.outcome == outcome, record.message
    assert fragment in record.message


@pytest.mark.parametrize("behavior", ["nonutf8", "deepjson", "hugeint"])
def test_hostile_output_ends_even_an_open_ended_session(behavior):
    # Before the fix these killed the reader thread and an open-ended search hung.
    safety = threading.Event()
    threading.Timer(20.0, safety.set).start()
    started = time.monotonic()
    record = runner.run_grind(mock(behavior), A0, ("ab",), None, FAST, stop=safety)
    assert record.outcome == runner.PROTOCOL, record.message
    assert time.monotonic() - started < 10 and not safety.is_set()


def test_hit_flood_is_stopped():
    limits = runner.Limits(setup_seconds=10, grace_seconds=1.5, max_offsets=50_000)
    record = runner.run_grind(mock("flood"), A0, ("ab",), 5.0, limits)
    assert record.outcome == runner.PROTOCOL and "hit flood" in record.message
    assert len(record.offsets) <= 55_000


def test_stdout_hostage_is_a_crash_and_is_reaped():
    # The entry exits without `done` while a grandchild holds its stdout open.
    spec = mock("stdout_hostage", isolate=False)
    record = runner.run_grind(spec, A0, ("ab",), None, FAST)
    assert record.outcome == runner.CRASH and "without sending done" in record.message
    time.sleep(0.3)
    assert leftovers() == 0  # same process group: killed with the entry


@pytest.mark.skipif(not ISOLATED, reason="needs unprivileged namespaces")
def test_isolated_entries_leave_no_processes_behind():
    record = runner.run_grind(mock("orphan"), A0, ("ab",), 0.5, FAST)
    assert record.outcome == runner.DONE, record.message
    runner.run_info(mock("info_orphan"))
    time.sleep(0.3)
    assert leftovers() == 0  # setsid escapees die with the PID namespace


def test_runner_reports_but_does_not_judge_hits():
    # Verifying hits is the scorer's job; the runner just records them faithfully.
    record = runner.run_grind(mock("liar"), A0, ("ab",), 0.2, FAST)
    assert record.outcome == runner.DONE and len(record.offsets) == 41  # 40 genuine + 1 forged


@pytest.mark.parametrize("behavior,fragment", [
    ("escape_info", "printable ASCII"),
    ("loud_info", "printed more than"),
])
def test_hostile_info_is_rejected(behavior, fragment):
    with pytest.raises(ProtocolError, match=fragment):
        runner.run_info(mock(behavior))


def test_run_info_rejects_bad_output(tmp_path):
    script = tmp_path / "bad.py"
    script.write_text("print('{\"contract\": \"nope\"}')\n")
    with pytest.raises(ProtocolError):
        runner.run_info(runner.EntrySpec((sys.executable, str(script)), isolate=ISOLATED))


def test_verify_summary_must_have_integer_counts(tmp_path):
    script = tmp_path / "v.py"
    script.write_text("print('{\"passed\": \"5\", \"failed\": 0}')\n")
    report = runner.run_verify(runner.EntrySpec((sys.executable, str(script)), isolate=ISOLATED),
                               tmp_path)
    assert not report.ok and "integer" in report.message
