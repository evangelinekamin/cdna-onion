"""The shared vectors: deterministic, tamper-evident, and independently cross-validated."""

import base64
import hashlib
import random

import nacl.bindings as sodium
import pytest

from onionrig import vectors
from onionrig.ed25519 import field as F
from onionrig.ed25519 import point as E

P = 2**255 - 19
L = 2**252 + 27742317777372353535851937790883648493


@pytest.fixture(scope="module")
def built():
    return vectors.build()


def le(h: str) -> int:
    return int.from_bytes(bytes.fromhex(h), "little")


def sodium_base(k: int) -> bytes:
    return sodium.crypto_scalarmult_ed25519_base_noclamp((k % L).to_bytes(32, "little"))


def sodium_offset(a0_hex: str, k: int) -> bytes:
    a0 = bytes.fromhex(a0_hex)
    if k % L == 0:
        return a0
    return sodium.crypto_core_ed25519_add(a0, sodium_base(8 * k))


def test_deterministic(built):
    again = vectors.build()
    for name in built:
        assert vectors.render(name, built[name]) == vectors.render(name, again[name])


def test_write_then_check_detects_tampering(tmp_path):
    vectors.write_all(tmp_path)
    assert vectors.check(tmp_path) == []
    target = tmp_path / "field.txt"
    target.write_bytes(target.read_bytes().replace(b"mul", b"mux", 1))
    assert any("field.txt" in p for p in vectors.check(tmp_path))


def test_field_vectors_match_plain_integers(built):
    plain = {
        "add": lambda a, b: (a + b) % P, "sub": lambda a, b: (a - b) % P,
        "mul": lambda a, b: a * b % P, "sqr": lambda a, _: a * a % P,
        "neg": lambda a, _: -a % P, "inv": lambda a, _: pow(a, P - 2, P),
    }
    for case in built["field"]:
        if case["op"] == "frombytes":
            assert le(case["out"]) == (le(case["a"]) & (2**255 - 1)) % P
            continue
        b = le(case["b"]) if case["b"] != "-" else None
        assert le(case["out"]) == plain[case["op"]](le(case["a"]), b), case


def test_field_vectors_cover_edge_values(built):
    inputs = {le(c["a"]) for c in built["field"] if c["op"] != "frombytes"}
    assert {0, 1, P - 1, 2**51 - 1, 2**64 - 1} <= inputs


def test_point_vectors_match_libsodium(built):
    for case in built["point"]:
        out = bytes.fromhex(case["out"])
        if case["op"] == "scalarmult_base":
            assert out == sodium_base(int(case["k"]))
        elif case["op"] == "add":
            p, q = bytes.fromhex(case["p"]), bytes.fromhex(case["q"])
            assert out == sodium.crypto_core_ed25519_add(p, q)
        elif case["op"] == "sub":
            p, q = bytes.fromhex(case["p"]), bytes.fromhex(case["q"])
            assert out == sodium.crypto_core_ed25519_sub(p, q)
        else:
            p = bytes.fromhex(case["p"])
            assert out == sodium.crypto_core_ed25519_add(p, p)


def test_decode_vectors(built):
    outcomes = {c["out"] == "invalid" for c in built["decode"]}
    assert outcomes == {True, False}
    for case in built["decode"]:
        if case["out"] != "invalid":
            assert case["out"] == case["in"]  # valid inputs are already canonical


@pytest.mark.parametrize("family", ["chain", "offset"])
def test_offset_rule_matches_libsodium(built, family):
    for case in built[family]:
        k = int(case["i"] if family == "chain" else case["k"])
        assert bytes.fromhex(case["out"]) == sodium_offset(case["a0"], k), case


def test_address_vectors_match_independent_encoder(built):
    for case in (c for c in built["onion"] if c["op"] == "address"):
        pk = bytes.fromhex(case["pubkey"])
        check = hashlib.sha3_256(b".onion checksum" + pk + b"\x03").digest()[:2]
        assert case["out"] == base64.b32encode(pk + check + b"\x03").decode().lower()


def test_fast_path_masks_agree_with_string_prefix(built):
    rng = random.Random(1)
    for case in (c for c in built["onion"] if c["op"] == "target"):
        mask, value = int(case["mask64"], 16), int(case["value64"], 16)
        for _ in range(200):
            pk = rng.randbytes(32)
            if rng.random() < 0.5:  # force some positives
                head = int.from_bytes(pk[:8], "big") & ~mask | value
                pk = head.to_bytes(8, "big") + pk[8:]
            address = vectors.onion.address_from_pubkey(pk)
            fast = int.from_bytes(pk[:8], "big") & mask == value
            assert fast == address.startswith(case["text"])


def test_match_vectors(built):
    cases = [c for c in built["onion"] if c["op"] == "match"]
    assert {c["out"] for c in cases} == {"0", "1"}
    for case in cases:
        address = vectors.onion.address_from_pubkey(bytes.fromhex(case["pubkey"]))
        assert (case["out"] == "1") == address.startswith(case["text"])


def test_grind_kats_are_complete_and_sound(built):
    for case in built["grind"]:
        lo, hi = int(case["lo"]), int(case["hi"])
        hits = [int(k) for k in case["hits"]]
        assert hits == sorted(hits) and all(lo <= k < hi for k in hits)
        assert any(k < 0 for k in hits) and any(k > 0 for k in hits)
        for k in hits:  # soundness, via libsodium
            address = vectors.onion.address_from_pubkey(sodium_offset(case["a0"], k))
            assert any(address.startswith(t) for t in case["targets"])
        rng = random.Random(2)
        misses = set(range(lo, hi)) - set(hits)
        for k in rng.sample(sorted(misses), 300):  # completeness, sampled
            address = vectors.onion.address_from_pubkey(sodium_offset(case["a0"], k))
            assert not any(address.startswith(t) for t in case["targets"])


def test_txt_carries_the_same_data_as_json(built):
    for name, cases in built.items():
        _, txt = vectors.render(name, cases)
        lines = [ln for ln in txt.decode().splitlines() if not ln.startswith("#")]
        assert lines == [vectors._txt_line(c) for c in cases]
        assert all(ln.split()[0] == c["op"] for ln, c in zip(lines, cases, strict=True))


def test_field_hex_is_canonical_little_endian():
    assert vectors._hex(F.P + 3) == (3).to_bytes(32, "little").hex()
    assert E.encode(E.BASE).hex() == "58" + "66" * 31
