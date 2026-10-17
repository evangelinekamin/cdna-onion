"""Tuning sweeps: interleaved, self-timed runs of an entry's builds on one GPU."""

import sys
from pathlib import Path

import pytest

from onionrig import cli
from onionrig.rig import isolation, results, sweep

ROOT = Path(__file__).resolve().parents[1]
MOCK = f"{sys.executable} {ROOT / 'tests/fixtures/mock_bench.py'}"
ISO = [] if isolation.isolation_available() else ["--no-isolation"]


def test_rounds_rotate_the_order():
    order = sweep.cell_order(["a", "b", "c"], 3)
    assert order[:3] == [(0, "a"), (0, "b"), (0, "c")]
    assert [name for r, name in order if r == 1] == ["b", "c", "a"]
    for name in "abc":  # over len(names) rounds, every build runs once in every slot
        assert sorted(i % 3 for i, (_, n) in enumerate(order) if n == name) == [0, 1, 2]


def cell(round_: int, name: str, rate: float | None, z: float = 0.0,
         command: str | None = None, error: str | None = None) -> dict:
    record = {"kind": "cell", "round": round_, "variant": name, "command": command or name}
    return {**record, "error": error} if error else {**record, "rate": rate, "z": z}


def test_builds_are_paired_with_the_base_of_their_round():
    # The base drifts down 6% over the sweep; 'fast' stays 3% ahead in every round.
    records = []
    for r, base in enumerate((100.0, 98.0, 96.0, 94.0)):
        wobble = 1 + 0.001 * (-1) ** r
        records += [cell(r, "base", base), cell(r, "fast", base * 1.03 * wobble),
                    cell(r, "same", base * wobble), cell(r, "slow", base * 0.95 * wobble)]
    rows = {row.name: row for row in sweep.summarize(records)}
    assert rows["base"].base and rows["base"].ratio is None and rows["base"].cells == 4
    assert rows["fast"].ratio.ratio == pytest.approx(1.03, abs=0.002)
    assert rows["fast"].ratio.ci_low > 1 and rows["slow"].ratio.ci_high < 1
    assert rows["same"].ratio.ci_low < 1 < rows["same"].ratio.ci_high
    assert sweep.pick(rows.values()) == "fast"


def test_pick_skips_wrong_failing_and_duplicate_builds():
    records = []
    for r, base in enumerate((100.0, 99.0, 98.0)):
        records += [cell(r, "base", base), cell(r, "liar", base * 1.5, z=-30.0),
                    cell(r, "flaky", base * 1.2), cell(r, "copy", base * 1.1, command="base")]
    records.append(cell(3, "flaky", None, error="exit 3"))
    rows = sweep.summarize(records)
    assert {row.name: row.errors for row in rows}["flaky"] == 1
    assert sweep.pick(rows) == "base"


def test_sweep_streams_every_cell_and_picks_the_fastest_honest_build(tmp_path, capsys):
    out = tmp_path / "sweep"
    code = cli.main(["sweep", f"base={MOCK} --rate 1e9", f"fast={MOCK} --rate 1.2e9",
                     f"liar={MOCK} --rate 2e9 --liar", f"prof={MOCK} --rate 1e9 --prof",
                     "--rounds", "2", "--seconds", "0.1", "--chars", "3", "--warmup", "0.1",
                     "--out", str(out), *ISO])
    assert code == 0
    [log] = out.glob("sweep-*.jsonl")
    records = results.read_jsonl([log])
    cells = [r for r in records if r["kind"] == "cell"]
    assert len(cells) == 8 and not any("error" in c for c in cells)
    assert next(c for c in cells if c["variant"] == "prof")["prof"][0] == 1500
    assert min(c["z"] for c in cells if c["variant"] == "liar") < -sweep.Z_LIMIT
    assert records[0]["kind"] == "session" and records[0]["base"] == "base"
    assert records[1]["kind"] == "warmup" and records[1]["variant"] == "base"
    assert records[-1]["kind"] == "summary" and records[-1]["pick"] == "fast"
    printed = capsys.readouterr().out
    assert "pick: fast" in printed
    assert "prof: 1.000 G/s (proof of work z +0.0; effective clock 1500 MHz)" in printed
    assert cli.main(["sweep", "--summarize", str(log)]) == 0
    assert "pick: fast" in capsys.readouterr().out


def test_a_failing_cell_is_recorded_and_the_sweep_goes_on(tmp_path):
    code = cli.main(["sweep", f"base={MOCK}", f"bad={MOCK} --crash", "--rounds", "2",
                     "--seconds", "0.1", "--chars", "3", "--out", str(tmp_path), *ISO])
    assert code == 0
    cells = [r for r in results.read_jsonl(tmp_path.glob("sweep-*.jsonl")) if r["kind"] == "cell"]
    bad = [c for c in cells if c["variant"] == "bad"]
    assert len(cells) == 4 and len(bad) == 2
    assert all("exit 3" in c["error"] and "GPU fault" in c["error"] for c in bad)


@pytest.mark.parametrize("variants", [["noequals"], [f"a={MOCK}", f"a={MOCK}"], ["=x"], ["a="]])
def test_variants_need_unique_names_and_commands(tmp_path, variants):
    with pytest.raises(SystemExit):
        cli.main(["sweep", *variants, "--out", str(tmp_path)])
