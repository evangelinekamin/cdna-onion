"""Conformance, benchmarking, and duels end to end, through the CLI."""

import json
import sys
from pathlib import Path

import pytest

from onionrig import cli
from onionrig.rig import bench as bench_mod
from onionrig.rig import check as check_mod
from onionrig.rig import isolation, plan, results, runner
from onionrig.rig import telemetry as tm

ISOLATED = isolation.isolation_available()

ROOT = Path(__file__).resolve().parents[1]
REF = f"{sys.executable} {ROOT / 'entries/python-ref/entry.py'}"
QUICK = ["--seconds", "1.5", "--calibrate-seconds", "1.5", "--min-runs", "2", "--max-runs", "3",
         "--rel-sem", "0.5", "--no-telemetry", "--verify-workers", "1"]
QUICK += [] if ISOLATED else ["--no-isolation"]


def mock(behavior: str) -> str:
    return f"{sys.executable} {ROOT / 'tests/fixtures/mock_entry.py'} --behavior {behavior}"


def records(out: Path) -> list[dict]:
    return results.read_jsonl(sorted(out.glob("*.jsonl")))


def test_plan_for_keeps_benchmarks_like_real_searches():
    for rate in (1e5, 3e6, 9.3e8, 3e9, 1.2e10):
        p = plan.plan_for(rate, 2_000)
        assert 1 <= p.count <= 4                       # few targets, as in a real search
        assert 1 / 3 < rate * p.p_match / 2_000 < 3    # hit rate within ~3x of the wish
    assert plan.plan_for(100, 1000) == plan.TargetPlan(1, 4)  # cannot get likelier than this
    assert plan.plan_for(1e9, 2_000, max_targets=1).count == 1
    with pytest.raises(ValueError):
        plan.plan_for(0, 1)


def test_fresh_inputs_are_fresh():
    a, b = plan.fresh_inputs(plan.TargetPlan(3, 4)), plan.fresh_inputs(plan.TargetPlan(3, 4))
    assert a.a0 != b.a0 and len(a.targets) == 4 and all(t.bits == 15 for t in a.targets)


def test_reference_entry_conforms(capsys, tmp_path):
    report = tmp_path / "check.json"
    flags = ["--out", str(tmp_path), "--json", str(report),
             *([] if ISOLATED else ["--no-isolation"])]
    assert cli.main(["check", REF, *flags]) == 0
    out = capsys.readouterr().out
    assert "conforms" in out and "FAIL" not in out and out.count("PASS") == 7
    # The same verdict, machine-readable, for the arena's judge.
    data = json.loads(report.read_text())
    assert data["conforms"] is True and len(data["checks"]) == 7
    assert all(c["ok"] for c in data["checks"]) and data["checks"][0]["name"] == "info"


def test_cheating_entry_fails_conformance():
    spec = runner.EntrySpec(tuple(mock("liar").split()), isolate=ISOLATED)
    results_ = check_mod.run_checks(spec, ROOT / "vectors")
    by_name = {r.name: r for r in results_}
    assert not check_mod.passed(results_)
    # Caught at the first grinding step: the calibration run's hits are verified too.
    assert not by_name["calibrate"].ok and "do not match" in by_name["calibrate"].detail
    assert not by_name["grind"].ok


def test_bench_writes_a_durable_log_and_a_summary(tmp_path, capsys):
    out = tmp_path / "bench"
    assert cli.main(["bench", REF, "--out", str(out), *QUICK]) == 0
    recs = records(out)
    kinds = [r["kind"] for r in recs]
    assert kinds[0] == "session" and kinds[-1] == "summary"
    runs = [r for r in recs if r["kind"] == "run" and r["role"] == "bench"]
    assert len(runs) >= 2 and all(r["valid"] for r in runs)
    assert all(abs(r["self_report_z"]) < 6 for r in runs)  # honest entry: counts agree
    summary = recs[-1]
    assert 20_000 < summary["rate"] < 2_000_000 and summary["eta13_median_s"] > 1e9
    assert "13-char" in capsys.readouterr().out


def test_bench_of_a_cheat_fails(tmp_path, capsys):
    out = tmp_path / "bench"
    assert cli.main(["bench", mock("liar"), "--out", str(out), *QUICK]) == 1
    assert "FAILED" in capsys.readouterr().out
    assert records(out)[-1]["failed"] and records(out)[-1]["rate"] is None


def sometimes_liar(tmp_path) -> runner.EntrySpec:
    argv = (sys.executable, str(ROOT / "tests/fixtures/mock_entry.py"),
            "--behavior", "sometimes_liar")
    return runner.EntrySpec(argv, env=(("MOCK_STATE", str(tmp_path / "state")),),
                            isolate=ISOLATED)


def quick_bench(writer, **overrides):
    config = bench_mod.BenchConfig(**{"seconds": 1.0, "calibrate_seconds": 1.0, "min_runs": 4,
                                      "max_runs": 4, "verify_workers": 1, **overrides})
    return bench_mod.Bench(config, writer, None)


def test_one_bad_run_fails_the_whole_bench(tmp_path):
    # The critical review finding: honest for two runs, then one forged key. Averaging
    # the good runs would rank an entry that hands users a wrong key.
    writer = results.JsonlWriter(tmp_path / "b.jsonl")
    with quick_bench(writer) as b:
        summary = bench_mod.bench(sometimes_liar(tmp_path), b)
    assert summary.failure and "do not match" in summary.failure and summary.summary is None
    recs = results.read_jsonl([writer.path])
    assert recs[-1]["kind"] == "summary" and recs[-1]["failed"]
    assert results.leaderboard(recs) == []


def test_a_lying_side_fails_the_duel(tmp_path):
    writer = results.JsonlWriter(tmp_path / "d.jsonl")
    with quick_bench(writer) as b:
        d = bench_mod.duel(runner.EntrySpec(tuple(REF.split()), isolate=ISOLATED),
                           sometimes_liar(tmp_path), b, pairs=2)
    # Side B (the liar) is blamed; both report the reference entry's name via --info.
    assert d.ratio is None and d.failure and d.failure.startswith("B (")
    assert "do not match" in d.failure


def test_duel_pairs_must_be_whole_abba_blocks():
    with pytest.raises(SystemExit):
        cli.main(["duel", REF, REF, "--pairs", "3"])


def test_duel_of_identical_entries_is_indistinguishable(tmp_path, capsys):
    out = tmp_path / "duel"
    assert cli.main(["duel", REF, REF, "--pairs", "2", "--out", str(out), *QUICK]) == 0
    summary = records(out)[-1]
    assert summary["kind"] == "summary" and summary["pairs"] == 2
    assert 0.7 < summary["ratio"] < 1.4
    assert "vs" in capsys.readouterr().out


def test_board_ranks_bench_results(tmp_path, capsys):
    out = tmp_path / "bench"
    cli.main(["bench", REF, "--out", str(out), *QUICK])
    assert cli.main(["board", "--out", str(out)]) == 0
    assert "python-ref" in capsys.readouterr().out


class SteadyTelemetry:
    name = "steady"

    def __init__(self, throttle: bool = False):
        self.throttle = throttle

    def sample(self):
        return tm.Sample(0.0, 50.0, 100.0, 1500.0, self.throttle, False)


def test_thermal_throttle_triggers_one_retry(tmp_path):
    writer = results.JsonlWriter(tmp_path / "t.jsonl")
    config = bench_mod.BenchConfig(seconds=1.0, verify_workers=1, sample_interval_s=0.2,
                                   cool_timeout_s=1)
    with bench_mod.Bench(config, writer, SteadyTelemetry(throttle=True)) as b:
        spec = runner.EntrySpec(tuple(REF.split()), isolate=ISOLATED)
        m = b.measured_run(spec, plan.TargetPlan(1, 2), 1.0, {"role": "test"})
    runs = [r for r in results.read_jsonl([writer.path]) if r["kind"] == "run"]
    assert [r["attempt"] for r in runs] == [1, 2] and "thermal-throttle" in m.flags
    assert runs[0]["counted"] is False and "replaced-by-retry" in runs[0]["flags"]


def test_vectors_command_round_trip(tmp_path, capsys):
    assert cli.main(["vectors", "--out", str(tmp_path / "v")]) == 0
    assert cli.main(["vectors", "--check", "--out", str(tmp_path / "v")]) == 0
    manifest = json.loads((tmp_path / "v" / "MANIFEST.json").read_text())
    assert len(manifest["sha256"]) == 14


def test_committed_vectors_are_current():
    assert cli.main(["vectors", "--check", "--out", str(ROOT / "vectors")]) == 0


def test_parse_cpus():
    assert cli.parse_cpus("0-3,8") == (0, 1, 2, 3, 8)
    assert cli.parse_cpus("5") == (5,)
