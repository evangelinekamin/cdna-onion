"""The arena: validating what boxes submit, quotas, fair scheduling, the judge's
summary, what boxes get back, and the runner's restart-safe event loop."""

import json
from pathlib import Path

import pytest

from onionrig.arena import judge, model
from onionrig.arena.runner import Runner

GOOD_ID = "20261016T120000-check-0a1b2c"
BENCH_ID = "20261016T120500-bench-ffee00"


def meta(sid=GOOD_ID, kind="check", created=1.0e9, **extra):
    return json.dumps({"id": sid, "kind": kind, "created": created, **extra})


# --- submissions -------------------------------------------------------------------

def test_parse_submission_accepts_a_well_formed_record():
    sub = model.parse_submission("claude", meta(), 1234)
    assert (sub.box, sub.id, sub.kind, sub.size) == ("claude", GOOD_ID, "check", 1234)


@pytest.mark.parametrize("text,size,reason", [
    ("{not json", 10, "not JSON"),
    ("[1, 2]", 10, "not an object"),
    (meta(sid="x; rm -rf ~"), 10, "id"),
    (meta(sid=BENCH_ID, kind="check"), 10, "kind"),
    (meta(kind="sprint"), 10, "kind"),
    (meta(created=True), 10, "created"),
    (meta(), 0, "tarball"),
    (meta(), model.MAX_TARBALL_BYTES + 1, "tarball"),
    (meta(pad="x" * model.MAX_META_BYTES), 10, "too large"),
])
def test_parse_submission_rejects_bad_records(text, size, reason):
    with pytest.raises(ValueError, match=reason):
        model.parse_submission("claude", text, size)


def test_only_well_formed_ids_leave_a_directory_listing():
    listing = "\n".join([f"{GOOD_ID}.json", f"{GOOD_ID}.tar.gz", "x; rm -rf ~.json",
                         f"{BENCH_ID}.json", "../../etc.json", "notes.txt", ""])
    assert model.ids_in_listing(listing) == [GOOD_ID, BENCH_ID]


# --- quota and scheduling ----------------------------------------------------------

def test_bench_slots_count_a_rolling_window():
    quota = model.Quota(bench_per_window=2, window_s=100.0)
    assert model.bench_slots([], now=1000.0, quota=quota) == (2, None)
    assert model.bench_slots([950.0], now=1000.0, quota=quota) == (1, None)
    # Full: the next slot opens when the oldest counted run leaves the window.
    assert model.bench_slots([950.0, 980.0], now=1000.0, quota=quota) == (0, 1050.0)
    assert model.bench_slots([800.0, 950.0], now=1000.0, quota=quota) == (1, None)


def test_round_robin_serves_the_next_box_with_work():
    order = ["a", "b", "c"]
    assert model.next_box({"a": 1, "c": 2}, order, last="a") == "c"
    assert model.next_box({"a": 1, "c": 2}, order, last="c") == "a"
    assert model.next_box({"b": 3}, order, last=None) == "b"
    assert model.next_box({}, order, last="b") is None


# --- what a box gets back ----------------------------------------------------------

def test_shape_for_box_keeps_known_fields_and_neutralizes_text():
    raw = {"status": "failed", "kind": "check", "reason": "x\x1b[31m" + "y" * 1000,
           "build": {"ok": True, "seconds": 3.5, "log": "a\nb\x07\n" + "z" * 999},
           "check": {"conforms": False, "checks": [
               {"name": "info", "ok": True, "required": True, "detail": "fine"},
               {"name": "grind", "ok": False, "required": True, "detail": "\x1b]0;evil\x07"},
               "junk"]},
           "secret_field": "dropped", "entry_log": "\n".join(f"line {i}" for i in range(500))}
    shaped = model.shape_for_box(raw)
    assert "secret_field" not in shaped
    assert "\x1b" not in json.dumps(shaped) and len(shaped["reason"]) <= model.MAX_LINE
    assert [c["name"] for c in shaped["check"]["checks"]] == ["info", "grind"]
    assert len(shaped["entry_log"].splitlines()) == model.MAX_LOG_LINES
    assert shaped["entry_log"].splitlines()[-1] == "line 499"


def test_shape_for_box_survives_garbage():
    shaped = model.shape_for_box({"status": 7, "check": "nope", "bench": [1], "build": None})
    assert shaped["status"] == "error" and shaped["check"] is None and shaped["bench"] is None


# --- the judge's summary (runs on Sol) ---------------------------------------------

def write_check(out: Path, conforms: bool):
    out.mkdir(parents=True, exist_ok=True)
    (out / "check.json").write_text(json.dumps({"entry": "/sub/src/entry", "conforms": conforms,
        "checks": [{"name": "info", "ok": True, "required": True, "detail": "ok"}]}))


def test_judge_check_passes_on_conformance(tmp_path):
    write_check(tmp_path / "out", conforms=True)
    r = judge.summarize("check", tmp_path / "out", build_ok=True, build_seconds=4.0, build_log="ok")
    assert r["status"] == "passed" and r["check"]["conforms"] and r["build"]["ok"]


def test_judge_reports_a_failed_build(tmp_path):
    r = judge.summarize("check", tmp_path / "out", build_ok=False, build_seconds=1.0,
                        build_log="src/k.cu:3: error: expected ';'")
    assert r["status"] == "failed" and r["reason"] == "build failed"
    assert "expected ';'" in r["build"]["log"] and r["check"] is None


def test_judge_bench_reads_the_summary(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    records = [{"kind": "session"}, {"kind": "run", "rate": 1.0},
               {"kind": "summary", "rate": 1.44e9, "rate_ci": [1.41e9, 1.47e9], "runs": 3,
                "failed": None, "gpu": "AMD Instinct MI210", "eta10_median_s": 5.4e5}]
    (out / "bench-1.jsonl").write_text("".join(json.dumps(x) + "\n" for x in records))
    r = judge.summarize("bench", out, build_ok=True, build_seconds=4.0, build_log="")
    assert r["status"] == "passed" and r["bench"]["rate"] == 1.44e9 and r["bench"]["runs"] == 3
    records[-1].update(rate=None, failed="run 2: wrong hit at offset 17")
    (out / "bench-1.jsonl").write_text("".join(json.dumps(x) + "\n" for x in records))
    r = judge.summarize("bench", out, build_ok=True, build_seconds=4.0, build_log="")
    assert r["status"] == "failed" and "wrong hit" in r["reason"]


def test_judge_without_output_is_an_error(tmp_path):
    r = judge.summarize("bench", tmp_path / "missing", build_ok=True, build_seconds=1.0,
                        build_log="")
    assert r["status"] == "error"


# --- the runner, against in-memory boxes and Sol ------------------------------------

class FakeBox:
    def __init__(self):
        self.outbox: dict[str, bytes] = {}  # file name -> content
        self.inbox: dict[str, str] = {}
        self.sent: list[str] = []

    def submit(self, sid, kind="check", created=1.0e9):
        self.outbox[f"{sid}.tar.gz"] = b"tarball-" + sid.encode()
        self.outbox[f"{sid}.json"] = meta(sid, kind, created).encode()

    def listing(self):
        return "\n".join(sorted(self.outbox))

    def size(self, name):
        return len(self.outbox[name])

    def read(self, name):
        return self.outbox[name]

    def retire(self, sid):
        for name in (f"{sid}.json", f"{sid}.tar.gz"):
            self.outbox.pop(name, None)
        self.sent.append(sid)

    def put(self, name, text):
        self.inbox[name] = text


class FakeSol:
    def __init__(self):
        self.uploads: dict[str, bytes] = {}
        self.jobs: dict[str, tuple[str, str]] = {}  # job id -> (sub dir, kind)
        self.finished: set[str] = set()

    def upload(self, sub_dir, data):
        self.uploads[sub_dir] = data

    def launch(self, sub_dir, kind):
        job = str(100 + len(self.jobs))
        self.jobs[job] = (sub_dir, kind)
        return job

    def running(self, job):
        return job not in self.finished

    def result(self, sub_dir):
        kind = next(k for d, k in self.jobs.values() if d == sub_dir)
        return json.dumps({"status": "passed", "kind": kind, "build": {"ok": True},
                           "bench": {"rate": 1.0e9, "runs": 3} if kind == "bench" else None})


def make_runner(tmp_path, boxes, sol, now=lambda: 2.0e9, quota=model.DEFAULT_QUOTA):
    return Runner(boxes, sol, tmp_path / "arena", quota=quota, clock=now)


def test_runner_collects_judges_and_delivers(tmp_path):
    box, sol = FakeBox(), FakeSol()
    box.submit(GOOD_ID)
    box.outbox["x; rm -rf ~.json"] = b"{}"
    runner = make_runner(tmp_path, {"claude": box}, sol)
    runner.step()  # collect, then launch
    assert box.sent == [GOOD_ID] and "x; rm -rf ~.json" in box.outbox
    archived = tmp_path / "arena" / "claude" / f"{GOOD_ID}.tar.gz"
    assert archived.read_bytes() == b"tarball-" + GOOD_ID.encode()
    assert "running" in box.inbox[f"{GOOD_ID}.status"]
    runner.step()
    assert f"{GOOD_ID}.json" not in box.inbox  # still running
    sol.finished.update(sol.jobs)
    runner.step()
    result = json.loads(box.inbox[f"{GOOD_ID}.json"])
    assert result["status"] == "passed" and result["id"] == GOOD_ID and result["box"] == "claude"


def test_runner_rejects_bench_over_quota(tmp_path):
    box, sol = FakeBox(), FakeSol()
    quota = model.Quota(bench_per_window=1, window_s=86_400.0)
    runner = make_runner(tmp_path, {"gpt": box}, sol, quota=quota)
    box.submit(BENCH_ID, kind="bench")
    runner.step()
    second = "20261016T121000-bench-abcdef"
    box.submit(second, kind="bench")
    runner.step()
    rejected = json.loads(box.inbox[f"{second}.json"])
    assert rejected["status"] == "rejected" and "quota" in rejected["reason"]
    assert json.loads(box.inbox["quota.json"])["bench_left"] == 0


def test_runner_takes_turns_between_boxes(tmp_path):
    a, b, sol = FakeBox(), FakeBox(), FakeSol()
    for i in range(3):
        a.submit(f"20261016T12000{i}-check-00000{i}")
    b.submit("20261016T120009-check-0000ff")
    runner = make_runner(tmp_path, {"a": a, "b": b}, sol)
    served = []
    for _ in range(4):
        runner.step()
        served.append(sol.jobs[max(sol.jobs, key=int)][0].split("/")[-2])
        sol.finished.update(sol.jobs)
        runner.step()
    assert served[:2] == ["a", "b"]  # b is not stuck behind a's whole queue


def test_runner_resumes_from_its_ledger(tmp_path):
    box, sol = FakeBox(), FakeSol()
    box.submit(GOOD_ID)
    box.submit("20261016T120100-check-0a1b2d")
    first = make_runner(tmp_path, {"claude": box}, sol)
    first.step()  # both collected, the first launched
    # A restart mid-job: the new runner must neither re-collect nor lose anything.
    second = make_runner(tmp_path, {"claude": box}, sol)
    sol.finished.update(sol.jobs)
    second.step()
    second.step()
    sol.finished.update(sol.jobs)
    second.step()
    assert len(sol.jobs) == 2
    assert {f"{GOOD_ID}.json", "20261016T120100-check-0a1b2d.json"} <= set(box.inbox)


def test_ledger_summary_per_box():
    from onionrig.arena.runner import summarize_ledger
    t = 2.0e9
    events = [
        {"event": "received", "box": "a", "id": "1", "kind": "bench", "time": t - 10},
        {"event": "queued", "box": "a", "id": "1", "kind": "bench", "time": t - 10},
        {"event": "launched", "box": "a", "id": "1", "kind": "bench", "job": "7", "time": t - 9},
        {"event": "finished", "box": "a", "id": "1", "status": "passed", "rate": 2.5e9, "time": t},
        {"event": "received", "box": "a", "id": "2", "kind": "check", "time": t - 5},
        {"event": "rejected", "box": "b", "id": "3", "reason": "bad", "time": t - 4},
        {"event": "received", "box": "b", "id": "4", "kind": "check", "time": t - 3},
        {"event": "queued", "box": "b", "id": "4", "kind": "check", "time": t - 3},
        {"event": "launched", "box": "b", "id": "4", "kind": "check", "job": "8", "time": t - 2},
    ]
    s = summarize_ledger(events, now=t, quota=model.Quota(bench_per_window=6))
    assert s["a"] == {"submitted": 2, "passed": 1, "failed": 0, "rejected": 0, "pending": 1,
                      "bench_left": 5, "best_rate": 2.5e9, "running": None}
    assert s["b"]["rejected"] == 1 and s["b"]["pending"] == 1 and s["b"]["running"] == "4"


def test_judge_command_line_writes_the_result(tmp_path):
    write_check(tmp_path / "out", conforms=True)
    (tmp_path / "build.log").write_text("hipcc ... ok\n")
    argv = ["check", str(tmp_path / "out"), "1", "7", str(tmp_path / "build.log"),
            str(tmp_path / "result.json")]
    assert judge.main(argv) == 0
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["status"] == "passed" and result["build"] == {"ok": True, "seconds": 7.0,
                                                                "log": "hipcc ... ok"}
