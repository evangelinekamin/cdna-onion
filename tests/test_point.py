"""ed25519 group reference, cross-checked against libsodium and `cryptography`."""

import hashlib
import secrets

import nacl.bindings as sodium
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
from hypothesis import given, settings
from hypothesis import strategies as st
from nacl.exceptions import CryptoError

from onionrig.ed25519 import field as F
from onionrig.ed25519 import point as E

scalars = st.integers(min_value=-(2**260), max_value=2**260)
small = st.integers(min_value=-(2**40), max_value=2**40)

# RFC 8032 section 7.1: (secret seed, public key).
RFC8032 = [
    ("9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60",
     "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a"),
    ("4ccd089b28ff96da9db6c346ec114e0f5b8a319f35aba624da8cf6ed4fb8a6fb",
     "3d4017c3e843895a92b70aa74d1b7ebc9c982ccf2ec4968cc0cd55f12af4660c"),
    ("c5aa8df43f9f837bedb7442f31dcb7b166d38535076f094b85ce3a2e0b4458f7",
     "fc51cd8e6218a1a38da47ed00230f0580816ed13ba3303ac5deb911548908025"),
]


def clamped_scalar_from_seed(seed: bytes) -> int:
    h = bytearray(hashlib.sha512(seed).digest()[:32])
    h[0] &= 248
    h[31] &= 127
    h[31] |= 64
    return int.from_bytes(h, "little")


def test_base_point_encoding():
    assert E.encode(E.BASE).hex() == "58" + "66" * 31


def test_base_point_has_order_l():
    assert E.equal(E.scalar_mult(E.L, E.BASE), E.IDENTITY)
    assert not E.equal(E.scalar_mult(E.L - 1, E.BASE), E.IDENTITY)


@pytest.mark.parametrize("seed_hex,pub_hex", RFC8032)
def test_rfc8032_public_keys(seed_hex, pub_hex):
    scalar = clamped_scalar_from_seed(bytes.fromhex(seed_hex))
    assert E.encode(E.scalar_mult(scalar, E.BASE)).hex() == pub_hex


def test_public_keys_match_cryptography_library():
    for _ in range(20):
        seed = secrets.token_bytes(32)
        theirs = Ed25519PrivateKey.from_private_bytes(seed).public_key().public_bytes(
            Encoding.Raw, PublicFormat.Raw)
        ours = E.encode(E.scalar_mult(clamped_scalar_from_seed(seed), E.BASE))
        assert ours == theirs


@settings(max_examples=40)
@given(st.integers(min_value=1, max_value=E.L - 1))
def test_scalar_mult_matches_libsodium(s):
    theirs = sodium.crypto_scalarmult_ed25519_base_noclamp(s.to_bytes(32, "little"))
    assert E.encode(E.scalar_mult(s, E.BASE)) == theirs


@settings(max_examples=40)
@given(st.integers(min_value=1, max_value=E.L - 1), st.integers(min_value=1, max_value=E.L - 1))
def test_add_matches_libsodium(a, b):
    pa, pb = E.scalar_mult(a, E.BASE), E.scalar_mult(b, E.BASE)
    theirs = sodium.crypto_core_ed25519_add(E.encode(pa), E.encode(pb))
    assert E.encode(E.add(pa, pb)) == theirs


@settings(max_examples=30)
@given(small, small, small)
def test_group_laws(a, b, c):
    pa, pb, pc = (E.scalar_mult(k, E.BASE) for k in (a, b, c))
    assert E.equal(E.add(pa, pb), E.add(pb, pa))
    assert E.equal(E.add(E.add(pa, pb), pc), E.add(pa, E.add(pb, pc)))
    assert E.equal(E.add(pa, E.IDENTITY), pa)
    assert E.equal(E.add(pa, E.neg(pa)), E.IDENTITY)
    assert E.equal(E.double(pa), E.add(pa, pa))


@settings(max_examples=30)
@given(scalars, scalars)
def test_scalar_mult_is_linear(a, b):
    lhs = E.scalar_mult(a + b, E.BASE)
    rhs = E.add(E.scalar_mult(a, E.BASE), E.scalar_mult(b, E.BASE))
    assert E.equal(lhs, rhs)


@given(scalars)
def test_negative_scalars(a):
    assert E.equal(E.scalar_mult(-a, E.BASE), E.neg(E.scalar_mult(a, E.BASE)))


@settings(max_examples=40)
@given(scalars)
def test_encode_decode_round_trip(a):
    p = E.scalar_mult(a, E.BASE)
    assert E.equal(E.decode(E.encode(p)), p)
    assert E.encode(E.decode(E.encode(p))) == E.encode(p)


def test_decode_rejects_noncanonical_y():
    with pytest.raises(ValueError):
        E.decode(F.P.to_bytes(32, "little"))
    with pytest.raises(ValueError):
        E.decode((F.P + 1).to_bytes(32, "little"))


def test_decode_rejects_negative_zero():
    # Identity is x = 0, y = 1; setting the sign bit asks for "-0".
    identity = E.encode(E.IDENTITY)
    assert identity == (1).to_bytes(32, "little")
    with pytest.raises(ValueError):
        E.decode(identity[:31] + bytes([identity[31] | 0x80]))


def test_decode_rejects_points_off_the_curve():
    rejected = 0
    for _ in range(200):
        raw = bytearray(secrets.token_bytes(32))
        raw[31] &= 0x7F                       # canonical range almost surely
        try:
            p = E.decode(bytes(raw))
        except ValueError:
            rejected += 1
            continue
        assert E.encode(p) == bytes(raw)      # accepted encodings are canonical
    assert 50 < rejected < 150                # about half of y values have no x


def test_decode_agrees_with_libsodium_on_random_encodings():
    # Adding the identity in libsodium is a pure decode-and-reencode, and it
    # rejects exactly the encodings with no curve point. Random inputs are
    # canonical with overwhelming probability, so the two must agree.
    identity = E.encode(E.IDENTITY)
    for _ in range(200):
        raw = secrets.token_bytes(32)
        try:
            ours = E.encode(E.decode(raw))
        except ValueError:
            ours = None
        try:
            theirs = sodium.crypto_core_ed25519_add(raw, identity)
        except CryptoError:
            theirs = None
        assert ours == theirs
