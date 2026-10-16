"""GF(2^255 - 19) reference arithmetic."""

import pytest
from hypothesis import given
from hypothesis import strategies as st

from onionrig.ed25519 import field as F

elements = st.integers(min_value=0, max_value=F.P - 1)
nonzero = st.integers(min_value=1, max_value=F.P - 1)


def test_modulus():
    assert F.P == 2**255 - 19


def test_sqrt_minus_one():
    assert F.mul(F.SQRT_M1, F.SQRT_M1) == F.P - 1


@given(elements, elements)
def test_ring_ops_match_integer_arithmetic(a, b):
    assert F.add(a, b) == (a + b) % F.P
    assert F.sub(a, b) == (a - b) % F.P
    assert F.mul(a, b) == (a * b) % F.P
    assert F.neg(a) == (-a) % F.P


@given(nonzero)
def test_inverse(a):
    assert F.mul(a, F.inv(a)) == 1


def test_inverse_of_zero_raises():
    with pytest.raises(ZeroDivisionError):
        F.inv(0)


@given(elements)
def test_bytes_round_trip(a):
    encoded = F.to_bytes(a)
    assert len(encoded) == 32
    assert F.from_bytes(encoded) == a


def test_from_bytes_ignores_bit_255():
    # Bit 255 carries the x sign in point encodings; field decoding drops it.
    assert F.from_bytes((1).to_bytes(32, "little")[:31] + b"\x80") == 1


def test_from_bytes_reduces_noncanonical_by_default():
    assert F.from_bytes(F.P.to_bytes(32, "little")) == 0
    assert F.from_bytes((F.P + 5).to_bytes(32, "little")) == 5


def test_from_bytes_strict_rejects_noncanonical():
    with pytest.raises(ValueError):
        F.from_bytes(F.P.to_bytes(32, "little"), strict=True)
    with pytest.raises(ValueError):
        F.from_bytes((2**255 - 1).to_bytes(32, "little"), strict=True)
    assert F.from_bytes((F.P - 1).to_bytes(32, "little"), strict=True) == F.P - 1


def test_from_bytes_requires_32_bytes():
    with pytest.raises(ValueError):
        F.from_bytes(b"\x00" * 31)
