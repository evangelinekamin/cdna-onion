"""Scoring a run (proof-of-work, anti-cheat) and the results/leaderboard layer."""

import json
import math
from array import array

import pytest

from onionrig import onion, vectors
from onionrig.rig import results, runner
from onionrig.rig.score import score


@pytest.fixture(scope="module")
def kat():
    case = next(c for c in vectors.build()["grind"] if c["name"] == "kat_multi")
    lo, hi = int(case["lo"]), int(case["hi"])
    hits = [int(k) for k in case["hits"]]
    miss = next(k for k in range(lo, hi) if k not in set(hits))
    return {"a0": bytes.fromhex(case["a0"]), "targets": onion.normalize_targets(case["targets"]),
            "hits": hits, "window": hi - lo, "miss": miss}


def record(offsets, candidates, outcome=runner.DONE, wall=1.0, message=""):
    return runner.GrindRecord(outcome, message, array("q", offsets), candidates, 0.1, wall, None)


def test_honest_run_is_valid_and_rate_matches_the_work(kat):
    s = score(record(kat["hits"], kat["window"]), kat["a0"], kat["targets"])
    assert s.valid and s.flags == ()
    # 32768 candidates in 1 s, estimated from ~96 hits: within a few counting errors.
    assert s.rate == pytest.approx(kat["window"], rel=4 * s.rate_rel_err)
    assert abs(s.self_report_z) < 4


def test_one_fake_offset_voids_the_run(kat):
    s = score(record(kat["hits"] + [kat["miss"]], kat["window"]), kat["a0"], kat["targets"])
    assert not s.valid and s.rate == 0
    assert s.bad_offsets == (kat["miss"],) and "do not match" in s.reason


def test_duplicates_count_once_and_are_flagged(kat):
    honest = score(record(kat["hits"], kat["window"]), kat["a0"], kat["targets"])
    padded = score(record(kat["hits"] + kat["hits"][:10], kat["window"]), kat["a0"],
                   kat["targets"])
    assert padded.valid and padded.duplicates == 10
    assert padded.rate == honest.rate and "duplicate-offsets" in padded.flags


def test_inflated_candidate_count_is_flagged_but_not_scored(kat):
    honest = score(record(kat["hits"], kat["window"]), kat["a0"], kat["targets"])
    inflated = score(record(kat["hits"], kat["window"] * 100), kat["a0"], kat["targets"])
    assert inflated.valid and inflated.rate == honest.rate
    assert "self-report-inconsistent" in inflated.flags and inflated.self_report_z < -6


def test_failed_sessions_and_empty_runs_are_invalid(kat):
    crashed = score(record(kat["hits"], None, runner.CRASH, message="boom"), kat["a0"],
                    kat["targets"])
    assert not crashed.valid and crashed.reason.startswith("crash")
    empty = score(record([], 0), kat["a0"], kat["targets"])
    assert not empty.valid and "no verified hits" in empty.reason


def test_as_record_is_json_safe(kat):
    s = score(record(kat["hits"] + [kat["miss"]], kat["window"]), kat["a0"], kat["targets"])
    json.dumps(s.as_record())


def test_jsonl_writer_appends_durably_and_reader_skips_torn_lines(tmp_path):
    path = tmp_path / "sub" / "bench.jsonl"
    writer = results.JsonlWriter(path)
    writer.write({"a": 1})
    writer.write({"b": 2})
    with open(path, "a") as handle:
        handle.write('{"torn": ')  # crash mid-write
    assert results.read_jsonl([path]) == [{"a": 1}, {"b": 2}]


def test_expected_seconds_matches_the_feasibility_tables():
    # The prior feasibility report: at 10 G/s, a 10-char prefix takes ~22 h at
    # even odds and a 13-char prefix ~81 years.
    assert results.expected_seconds(10e9, 10) / 3600 == pytest.approx(21.7, rel=0.01)
    assert results.expected_seconds(10e9, 13) / (365.25 * 86400) == pytest.approx(81, rel=0.01)
    assert results.expected_seconds(0, 10) == math.inf


def test_human_formatting():
    assert results.human_rate(3.2e9) == "3.20 G/s"
    assert results.human_rate(512) == "512 /s"
    assert results.human_duration(7200) == "2.0 h"
    assert results.human_duration(math.inf) == "never"
    assert results.human_duration(2 * 365.25 * 86400) == "2.0 years"


def summary(entry, rate, runs=3, failed=None, gpu="MI210"):
    return {"kind": "summary", "mode": "bench", "entry": entry, "version": "1", "gpu": gpu,
            "rate": rate, "runs": runs, "failed": failed}


def test_leaderboard_uses_bench_summaries_only():
    records = [
        summary("slow", 1e9),
        summary("fast", 3.0e9, runs=2),
        summary("fast", 3.3e9, runs=4),
        # Raw runs (calibrations, replaced attempts, duel runs) never reach the board.
        {"kind": "run", "valid": True, "entry": "fast", "gpu": "MI210", "rate": 9e9,
         "role": "calibration"},
        {"kind": "summary", "mode": "duel", "a": "fast", "b": "slow", "ratio": 3.0},
    ]
    rows = results.leaderboard(records)
    assert [r["entry"] for r in rows] == ["fast", "slow"]
    assert rows[0]["runs"] == 6 and rows[0]["rate"] == pytest.approx(3.2e9)  # run-weighted
    table = results.format_leaderboard(rows)
    assert "fast" in table and "years" in table
    assert results.format_leaderboard([]) == "no valid runs yet"


def test_one_failed_bench_disqualifies_the_entry_on_that_gpu():
    records = [summary("sly", 5e9), summary("sly", None, runs=0, failed="1 bad offset"),
               summary("sly", 5e9, gpu="MI355X"), summary("honest", 1e9)]
    rows = {(r["entry"], r["gpu"]) for r in results.leaderboard(records)}
    assert rows == {("sly", "MI355X"), ("honest", "MI210")}


def test_board_survives_malformed_records():
    records = [summary("ok", 2e9), summary("weird", "fast"), summary("neg", -5.0),
               {"kind": "summary", "mode": "bench", "rate": float("nan")}]
    assert [r["entry"] for r in results.leaderboard(records)] == ["ok"]


def test_jsonl_never_contains_nan_or_infinity(tmp_path):
    writer = results.JsonlWriter(tmp_path / "x.jsonl")
    writer.write({"a": math.inf, "b": [math.nan, 1.0], "c": {"d": -math.inf}})
    text = (tmp_path / "x.jsonl").read_text()
    assert "Infinity" not in text and "NaN" not in text
    assert json.loads(text) == {"a": None, "b": [None, 1.0], "c": {"d": None}}


def test_entry_strings_are_sanitized_for_the_terminal():
    rows = results.leaderboard([summary("evil\x1b]52;c;x\x07\x1b[8m", 1e9)])
    table = results.format_leaderboard(rows)
    assert "\x1b" not in table and "\x07" not in table


def test_scoring_a_flood_of_bad_offsets_stops_early(kat):
    import time
    started = time.monotonic()
    s = score(record(list(range(0, 200_000)), 200_000), kat["a0"], kat["targets"])
    assert not s.valid and len(s.bad_offsets) <= 20
    assert time.monotonic() - started < 5  # sampled first, not 200k verifications
