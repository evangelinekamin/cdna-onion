"""Tor v3 onion addresses and vanity prefix targets.

address = base32(PUBKEY || CHECKSUM || VERSION), 56 characters, where
CHECKSUM = SHA3-256(".onion checksum" || PUBKEY || VERSION)[:2] and VERSION = 3.

base32 reads the byte stream MSB-first, so character i covers stream bits
[5i, 5i + 5). PUBKEY comes first and is the little-endian y coordinate, so a
prefix constrains the LOW bits of y. The x sign bit is the top bit of byte 31,
which is stream bit 248, inside character 49. Prefixes of at most 49 characters
therefore depend on y alone, which is the limit set here.
"""

import base64
import hashlib
from collections.abc import Iterable
from fractions import Fraction
from typing import NamedTuple

ALPHABET = "abcdefghijklmnopqrstuvwxyz234567"
VERSION = 3
ADDRESS_CHARS = 56
MAX_PREFIX_CHARS = 49

_CHECKSUM_TAG = b".onion checksum"
_INDEX = {c: i for i, c in enumerate(ALPHABET)}


def checksum(pubkey: bytes, version: int = VERSION) -> bytes:
    return hashlib.sha3_256(_CHECKSUM_TAG + pubkey + bytes([version])).digest()[:2]


def address_from_pubkey(pubkey: bytes) -> str:
    """The 56-character address (without '.onion') for a 32-byte public key."""
    if len(pubkey) != 32:
        raise ValueError(f"public key must be 32 bytes, got {len(pubkey)}")
    raw = pubkey + checksum(pubkey) + bytes([VERSION])
    return base64.b32encode(raw).decode("ascii").lower()


def pubkey_from_address(address: str) -> bytes:
    """Decode and fully validate an address; accepts a '.onion' suffix and any case."""
    text = address.lower().removesuffix(".onion")
    if len(text) != ADDRESS_CHARS or any(c not in _INDEX for c in text):
        raise ValueError(f"not a 56-character base32 v3 address: {address!r}")
    raw = base64.b32decode(text.upper())
    pubkey, check, version = raw[:32], raw[32:34], raw[34]
    if version != VERSION:
        raise ValueError(f"unsupported onion address version {version}")
    if check != checksum(pubkey, version):
        raise ValueError("onion address checksum mismatch")
    return pubkey


class Target(NamedTuple):
    """A vanity prefix: its text, and the first `bits` stream bits it requires."""

    text: str
    bits: int
    value: int


def parse_target(text: str) -> Target:
    normalized = text.lower()
    if not 1 <= len(normalized) <= MAX_PREFIX_CHARS:
        raise ValueError(f"prefix must be 1..{MAX_PREFIX_CHARS} characters: {text!r}")
    value = 0
    for c in normalized:
        if c not in _INDEX:
            raise ValueError(f"{c!r} is not in the onion alphabet a-z, 2-7 (prefix {text!r})")
        value = value << 5 | _INDEX[c]
    return Target(normalized, 5 * len(normalized), value)


def leading_bits(pubkey: bytes, n: int) -> int:
    """The first n bits of the byte stream, MSB-first, as an integer."""
    return int.from_bytes(pubkey, "big") >> (8 * len(pubkey) - n)


def matches(pubkey: bytes, target: Target) -> bool:
    return leading_bits(pubkey, target.bits) == target.value


def normalize_targets(texts: Iterable[str]) -> tuple[Target, ...]:
    """Parse, dedupe, and drop any target that extends a shorter one.

    The result has no target that is a prefix of another, so the match events
    are disjoint and their probabilities simply add.
    """
    kept: list[Target] = []
    for target in sorted({parse_target(t) for t in texts}, key=lambda t: (t.bits, t.text)):
        if not any(target.text.startswith(k.text) for k in kept):
            kept.append(target)
    return tuple(sorted(kept, key=lambda t: t.text))


def match_probability(targets: tuple[Target, ...]) -> Fraction:
    """Exact chance that one uniformly random public key matches any target.

    Requires normalized (prefix-free) targets.
    """
    return sum((Fraction(1, 1 << t.bits) for t in targets), Fraction(0))
