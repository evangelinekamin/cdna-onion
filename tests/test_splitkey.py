"""Split-key search: offsets on a public point recombine into a Tor-usable secret."""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from onionrig import splitkey
from onionrig.ed25519 import point as E

offsets = st.integers(min_value=-(2**62), max_value=2**62)


def test_step_is_eight_times_base():
    assert E.equal(splitkey.STEP, E.scalar_mult(8, E.BASE))


@pytest.mark.parametrize("s,ok", [
    (2**254, True),
    (2**255 - 8, True),
    (2**254 + 4, False),       # low bits not clear
    (2**254 - 8, False),       # bit 254 not set
    (2**255, False),           # bit 255 set
])
def test_is_clamped(s, ok):
    assert splitkey.is_clamped(s) is ok


def test_random_start_scalar_is_clamped_with_headroom():
    seen = set()
    for _ in range(50):
        a0 = splitkey.random_start_scalar()
        assert splitkey.is_clamped(a0)
        assert splitkey.is_clamped(a0 - splitkey.HEADROOM)
        assert splitkey.is_clamped(a0 + splitkey.HEADROOM - 8)
        seen.add(a0)
    assert len(seen) == 50


@settings(max_examples=25)
@given(offsets)
def test_point_at_equals_scalar_mult_of_combined_scalar(k):
    a0 = splitkey.random_start_scalar()
    a0_point = E.scalar_mult(a0, E.BASE)
    expected = E.scalar_mult(splitkey.combine(a0, k), E.BASE)
    assert E.equal(splitkey.point_at(a0_point, k), expected)


def test_offset_zero_is_the_start_point():
    a0 = splitkey.random_start_scalar()
    a0_point = E.scalar_mult(a0, E.BASE)
    assert E.equal(splitkey.point_at(a0_point, 0), a0_point)
    assert splitkey.combine(a0, 0) == a0


def test_combine_refuses_offsets_that_break_clamping():
    with pytest.raises(ValueError, match="clamp"):
        splitkey.combine(2**255 - 8, 1)
    with pytest.raises(ValueError, match="clamp"):
        splitkey.combine(2**254, -1)
    with pytest.raises(TypeError):
        splitkey.combine(2**254 + 8, 0.5)  # type: ignore[arg-type]


def test_start_point_encodes_a0_times_base():
    a0 = splitkey.random_start_scalar()
    assert splitkey.start_point(a0) == E.encode(E.scalar_mult(a0, E.BASE))
