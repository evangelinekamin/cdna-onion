"""The hit verifier agrees with the reference math and with the known-answer grinds."""

import random
from concurrent.futures import ProcessPoolExecutor

import pytest

from onionrig import onion, splitkey, vectors
from onionrig.ed25519 import point as E
from onionrig.rig.verify import HitVerifier, parallel_failures


@pytest.fixture(scope="module")
def kats():
    return [c for c in vectors.build()["grind"] if c["op"] == "kat"]


def make_verifier(case) -> HitVerifier:
    targets = onion.normalize_targets(case["targets"])
    return HitVerifier(bytes.fromhex(case["a0"]), targets)


def test_candidates_match_reference_math():
    rng = random.Random(7)
    a0_point = E.scalar_mult(rng.randrange(1, E.L), E.BASE)
    verifier = HitVerifier(E.encode(a0_point), onion.normalize_targets(["a"]))
    for k in [0, 1, -1, 2**62, -(2**62)] + [rng.randrange(-(2**63) + 1, 2**63) for _ in range(30)]:
        assert verifier.candidate(k) == E.encode(splitkey.point_at(a0_point, k))


def test_kat_hits_verify_and_neighbours_do_not(kats):
    for case in kats:
        verifier = make_verifier(case)
        hits = [int(k) for k in case["hits"]]
        assert verifier.failures(hits) == []
        lo, hi = int(case["lo"]), int(case["hi"])
        misses = sorted(set(range(lo, hi)) - set(hits))
        assert verifier.failures(misses[:500]) == misses[:500]


def test_matches_agrees_with_address_strings():
    rng = random.Random(9)
    targets = onion.normalize_targets(["b", "z7", "mo"])
    verifier = HitVerifier(E.encode(E.scalar_mult(rng.randrange(1, E.L), E.BASE)), targets)
    for _ in range(300):
        k = rng.randrange(-(2**40), 2**40)
        address = onion.address_from_pubkey(verifier.candidate(k))
        assert verifier.matches(k) == any(address.startswith(t.text) for t in targets)


def test_rejects_invalid_start_points():
    with pytest.raises(ValueError, match="a0"):
        HitVerifier((2**255 - 19).to_bytes(32, "little"), onion.normalize_targets(["a"]))
    with pytest.raises(ValueError, match="target"):
        HitVerifier(E.encode(E.BASE), ())


def test_parallel_matches_serial(kats):
    verifier = make_verifier(kats[1])
    lo = int(kats[1]["lo"])
    offsets = list(range(lo, lo + 6000))
    with ProcessPoolExecutor(max_workers=4) as pool:
        assert parallel_failures(verifier, offsets, pool, chunk=500) \
            == verifier.failures(offsets)
    assert parallel_failures(verifier, offsets[:10], None) == verifier.failures(offsets[:10])
