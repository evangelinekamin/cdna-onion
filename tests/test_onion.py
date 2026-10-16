"""Onion addresses, prefix targets, and the little-endian matching trap."""

import base64
import hashlib
import secrets
from fractions import Fraction

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from onionrig import onion
from onionrig.ed25519 import point as E

# All five validate (checksum + version); duckduckgo is a real 10-char vanity.
KNOWN_ADDRESSES = [
    "pg6mmjiyjmcrsslvykfwnntlaru7p5svn6y2ymmju6nubxndf4pscryd",  # Tor spec examples
    "sp3k262uwy4r2k3ycr5awluarykdpag6a7y33jxop4cs2lu5uz5sseqd",
    "xa4r2iadxm55fbnqgwwi5mymqdcofiu3w6rpbtqn7b2dyn7mgwj64jyd",
    "duckduckgogg42xjoc72x3sjasowoarfbgcmvfimaftt6twagswzczad",
    "2gzyxa5ihm7nsggfxnu52rck2vv4rvmdlkiu3zzui5du4xyclen53wid",  # torproject.org
]

pubkeys = st.binary(min_size=32, max_size=32)


@pytest.mark.parametrize("address", KNOWN_ADDRESSES)
def test_known_addresses_round_trip(address):
    pubkey = onion.pubkey_from_address(address)
    assert onion.address_from_pubkey(pubkey) == address
    assert onion.pubkey_from_address(address.upper() + ".onion") == pubkey


@given(pubkeys)
def test_address_matches_independent_encoder(pubkey):
    check = hashlib.sha3_256(b".onion checksum" + pubkey + b"\x03").digest()[:2]
    expected = base64.b32encode(pubkey + check + b"\x03").decode().lower()
    assert onion.address_from_pubkey(pubkey) == expected
    assert len(expected) == 56


def test_rejects_malformed_addresses():
    good = KNOWN_ADDRESSES[0]
    flipped = ("b" if good[10] != "b" else "c").join((good[:10], good[11:]))
    for bad in (good[:-1], good + "a", good[:5] + "0" + good[6:], flipped, ""):
        with pytest.raises(ValueError):
            onion.pubkey_from_address(bad)


def test_rejects_wrong_version_even_with_matching_checksum():
    pubkey = secrets.token_bytes(32)
    check = hashlib.sha3_256(b".onion checksum" + pubkey + b"\x02").digest()[:2]
    v2 = base64.b32encode(pubkey + check + b"\x02").decode().lower()
    with pytest.raises(ValueError, match="version"):
        onion.pubkey_from_address(v2)


def test_parse_target():
    t = onion.parse_target("Ba")
    assert (t.text, t.bits, t.value) == ("ba", 10, (1 << 5) | 0)
    assert onion.parse_target("7").value == 31
    assert onion.parse_target("a" * 49).bits == 245


@pytest.mark.parametrize("bad", ["", "a0", "1", "abc8", "ab-c", "a" * 50])
def test_parse_target_rejects(bad):
    with pytest.raises(ValueError):
        onion.parse_target(bad)


@given(pubkeys, st.integers(min_value=1, max_value=onion.MAX_PREFIX_CHARS))
def test_match_equals_string_prefix_of_own_address(pubkey, n):
    address = onion.address_from_pubkey(pubkey)
    assert onion.matches(pubkey, onion.parse_target(address[:n]))


@settings(max_examples=300)
@given(pubkeys, st.text(alphabet=onion.ALPHABET, min_size=1, max_size=3))
def test_match_is_exactly_string_prefix(pubkey, text):
    # The little-endian trap: the prefix lives in the LOW bytes of y. A matcher
    # that compares the wrong end disagrees with startswith() here.
    expected = onion.address_from_pubkey(pubkey).startswith(text)
    assert onion.matches(pubkey, onion.parse_target(text)) == expected


def test_prefix_constrains_low_bits_of_y():
    # y = 2^200 has its low 200 bits clear, so its address starts with 'a's even
    # though y is huge; y = 1 sets bit 0, which lands in the second character.
    high = (1 << 200).to_bytes(32, "little")
    assert onion.address_from_pubkey(high).startswith("a" * 39)
    assert onion.address_from_pubkey((1).to_bytes(32, "little")).startswith("ae")


def test_leading_bits():
    pubkey = bytes([0b10110011, 0b01000000]) + bytes(30)
    assert onion.leading_bits(pubkey, 5) == 0b10110
    assert onion.leading_bits(pubkey, 10) == 0b1011001101


def test_normalize_targets_dedupes_and_drops_redundant_extensions():
    targets = onion.normalize_targets(["abc", "ab", "xy", "XY", "abd"])
    assert [t.text for t in targets] == ["ab", "xy"]


def test_match_probability_is_exact():
    assert onion.match_probability(onion.normalize_targets(["abcd"])) == Fraction(1, 32**4)
    many = onion.normalize_targets(["aa", "ab", "ba", "aab"])
    assert onion.match_probability(many) == Fraction(3, 32**2)


def test_real_points_match_like_random_bytes():
    # Matching is a function of the encoding alone; spot-check on curve points.
    for k in range(1, 30):
        pubkey = E.encode(E.scalar_mult(k, E.BASE))
        address = onion.address_from_pubkey(pubkey)
        assert onion.matches(pubkey, onion.parse_target(address[:4]))
        other = "a" if address[0] != "a" else "b"
        assert not onion.matches(pubkey, onion.parse_target(other))


def test_x_sign_bit_first_affects_character_49():
    # P and -P share y and differ only in the x sign (stream bit 248, the top bit of
    # byte 31), so their addresses agree on characters 0..48 and differ at 49. This
    # is why targets stop at 49 characters: up to there, y alone decides a match.
    for k in range(1, 40):
        p = E.scalar_mult(k, E.BASE)
        a = onion.address_from_pubkey(E.encode(p))
        b = onion.address_from_pubkey(E.encode(E.neg(p)))
        assert a[:49] == b[:49] and a[49] != b[49]
