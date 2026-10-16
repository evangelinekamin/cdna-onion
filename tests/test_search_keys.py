"""The full split-key workflow: keygen locally, search remotely, combine locally, Tor loads it."""

import json
import os
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

from onionrig import cli, keys, onion, torkeys
from onionrig.rig import isolation, results, runner
from onionrig.rig import search as search_mod

ISO = [] if isolation.isolation_available() else ["--no-isolation"]

ROOT = Path(__file__).resolve().parents[1]
REF = f"{sys.executable} {ROOT / 'entries/python-ref/entry.py'}"


def test_keygen_keeps_scalars_private_and_points_public(tmp_path):
    public = keys.keygen(tmp_path / "keys", 5)
    secret = tmp_path / "keys" / keys.SECRET_NAME
    assert stat.S_IMODE(secret.stat().st_mode) == 0o600
    points = keys.load_points(public)
    assert [i for i, _ in points] == list(range(5))
    assert all(json.loads(line)["a0"] not in public.read_text()
               for line in secret.read_text().splitlines())
    with pytest.raises(FileExistsError):
        keys.keygen(tmp_path / "keys", 5)


def test_point_pool_hands_out_each_walked_point_once(tmp_path):
    points = [(i, bytes([i]) * 32) for i in range(3)]
    used = tmp_path / "used.json"
    pool = search_mod.PointPool(points, used)
    first = pool.take()
    pool.give_back(first)                      # never started: still available
    assert pool.take() == first
    pool.consume(first[1])
    second = pool.take()
    pool.consume(second[1])
    restarted = search_mod.PointPool(points, used)
    assert restarted.take()[0] == 2 and restarted.take() is None
    # Keyed by the point itself, so a fresh keygen reusing ids 0..n is unaffected.
    fresh = search_mod.PointPool([(0, bytes([9]) * 32)], used)
    assert fresh.take() == (0, bytes([9]) * 32)


def test_search_then_combine_yields_the_found_address(tmp_path, capsys):
    keys.keygen(tmp_path / "keys", 4)
    out = tmp_path / "search"
    code = cli.main(["search", REF, "--points", str(tmp_path / "keys" / keys.PUBLIC_NAME),
                     "--prefix", "ab", "--prefix", "q", "--max-hits", "2", "--out", str(out),
                     *ISO])
    assert code == 0
    hits = [r for r in results.read_jsonl([out / "search.jsonl"]) if r["kind"] == "hit"]
    assert len(hits) >= 2
    hit = hits[0]
    assert hit["address"].startswith(("ab", "q"))

    service = tmp_path / "hs"
    assert cli.main(["combine", "--secrets", str(tmp_path / "keys" / keys.SECRET_NAME),
                     "--point", str(hit["point"]), "--k", hit["k"], "--address", hit["address"],
                     "--out", str(service)]) == 0
    key = torkeys.OnionKey.from_secret_key_file((service / "hs_ed25519_secret_key").read_bytes())
    assert key.address == hit["address"]


def test_combine_refuses_a_mismatched_hit(tmp_path):
    keys.keygen(tmp_path / "keys", 2)
    with pytest.raises(ValueError, match="wrong point id or offset"):
        keys.combine(tmp_path / "keys" / keys.SECRET_NAME, 0, 5, tmp_path / "hs",
                     expected_address="a" * 56)
    assert not (tmp_path / "hs").exists()


def test_search_stops_a_device_whose_entry_lies(tmp_path):
    keys.keygen(tmp_path / "keys", 2)
    liar = mock_spec("liar")
    writer = results.JsonlWriter(tmp_path / "search.jsonl")
    config = search_mod.SearchConfig(onion.normalize_targets(["ab"]), (0,))
    pool = search_mod.PointPool(keys.load_points(tmp_path / "keys" / keys.PUBLIC_NAME),
                                tmp_path / "used.json")
    search_mod.Search(liar, pool, config, writer).run()
    kinds = [r["kind"] for r in results.read_jsonl([writer.path])]
    assert "bad-hit" in kinds and "device-stopped" in kinds


@pytest.mark.integration
@pytest.mark.skipif(shutil.which("tor") is None, reason="tor not installed")
def test_found_key_is_accepted_by_real_tor(tmp_path):
    keys.keygen(tmp_path / "keys", 1)
    out = tmp_path / "search"
    cli.main(["search", REF, "--points", str(tmp_path / "keys" / keys.PUBLIC_NAME),
              "--prefix", "a", "--max-hits", "1", "--out", str(out), *ISO])
    hit = next(r for r in results.read_jsonl([out / "search.jsonl"]) if r["kind"] == "hit")
    service = tmp_path / "hs"
    key = keys.combine(tmp_path / "keys" / keys.SECRET_NAME, hit["point"], int(hit["k"]),
                       service)
    (service / "hostname").unlink()  # make Tor derive it from our secret key
    data = tmp_path / "data"
    data.mkdir(mode=0o700)
    torrc = tmp_path / "torrc"
    torrc.write_text(f"DataDirectory {data}\nDisableNetwork 1\nSocksPort 0\n"
                     f"HiddenServiceDir {service}\nHiddenServicePort 80 127.0.0.1:9\n")
    proc = subprocess.Popen(["tor", "-f", str(torrc)], stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
    try:
        deadline = time.monotonic() + 30
        while not (service / "hostname").exists() and time.monotonic() < deadline:
            time.sleep(0.1)
    finally:
        proc.terminate()
        proc.wait(timeout=10)
    assert (service / "hostname").read_text().strip() == f"{key.address}.onion" \
        == f"{hit['address']}.onion"


def mock_spec(behavior: str) -> runner.EntrySpec:
    return runner.EntrySpec((sys.executable, str(ROOT / "tests/fixtures/mock_entry.py"),
                             "--behavior", behavior),
                            isolate=isolation.isolation_available())


def run_search(tmp_path, behavior, points=5, max_hits=None, writer=None):
    keys.keygen(tmp_path / "keys", points)
    pool = search_mod.PointPool(keys.load_points(tmp_path / "keys" / keys.PUBLIC_NAME),
                                tmp_path / "used.json")
    writer = writer or results.JsonlWriter(tmp_path / "search.jsonl")
    config = search_mod.SearchConfig(onion.normalize_targets(["ab"]), (0,), max_hits)
    limits = runner.Limits(setup_seconds=3, grace_seconds=1)
    search = search_mod.Search(mock_spec(behavior), pool, config, writer, limits,
                               progress=lambda line: None)
    return search.run(), pool, results.read_jsonl([writer.path]) if writer.path.exists() else []


def test_points_are_not_burned_by_an_entry_that_never_starts(tmp_path):
    _, pool, recs = run_search(tmp_path, "silent")        # never sends ready: never started
    assert pool.remaining == 5
    assert any(r["kind"] == "device-stopped" for r in recs)


def test_a_crashing_entry_is_retired_after_three_sessions(tmp_path):
    _, pool, recs = run_search(tmp_path, "crash")         # dies right after start
    assert pool.remaining == 5 - search_mod.MAX_CONSECUTIVE_FAILURES
    assert [r["kind"] for r in recs].count("session-end") == search_mod.MAX_CONSECUTIVE_FAILURES


def test_search_counts_duplicate_hits_once(tmp_path):
    found, _, recs = run_search(tmp_path, "duplicator", max_hits=40)
    hits = [r["k"] for r in recs if r["kind"] == "hit"]
    assert found == len(hits) == len(set(hits)) == 40


class BrokenDisk(results.JsonlWriter):
    def write(self, record):
        raise OSError(28, "No space left on device")


def test_search_survives_an_unwritable_log(tmp_path):
    found, _, _ = run_search(tmp_path, "honest", max_hits=3,
                             writer=BrokenDisk(tmp_path / "s.jsonl"))
    # Hits are still found and shown even though nothing could be logged. The final
    # batch is recorded whole (every hit in it is a genuine find), hence ">=".
    assert found >= 3


def test_keygen_refuses_the_project_tree():
    with pytest.raises(ValueError, match="copied to the GPU host"):
        keys.keygen(ROOT / "keys-should-not-go-here", 1)
    assert not (ROOT / "keys-should-not-go-here").exists()


def test_malformed_secret_record_never_quotes_the_scalar(tmp_path):
    secret = tmp_path / keys.SECRET_NAME
    scalar_hex = "7" * 63 + "z"  # one corrupted digit
    secret.write_text(json.dumps({"id": 0, "a0": scalar_hex}) + "\n")
    with pytest.raises(ValueError) as caught:
        keys.combine(secret, 0, 1, tmp_path / "hs")
    text = str(caught.value) + repr(caught.value.__cause__) + repr(caught.value.__context__)
    assert "7777777" not in text


def test_failed_secret_write_leaves_nothing_behind(tmp_path, monkeypatch):
    real_write = os.write
    calls = []

    def fail_second(fd, data):
        calls.append(1)
        if len(calls) > 1:
            raise OSError(28, "No space left on device")
        return real_write(fd, bytes(data)[:40])

    monkeypatch.setattr(torkeys.os, "write", fail_second)
    with pytest.raises(OSError):
        torkeys.write_service_dir(tmp_path / "hs", torkeys.OnionKey.from_scalar(2**254 + 8))
    assert list((tmp_path / "hs").iterdir()) == []


def test_combine_requires_the_expected_address(tmp_path):
    with pytest.raises(SystemExit):
        cli.main(["combine", "--secrets", "x", "--point", "0", "--k", "1", "--out", "y"])
