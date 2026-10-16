"""Split-key vanity search: the secret never leaves the machine that made it.

1. Locally, draw a clamped start scalar a0 and publish only A0 = a0*B.
2. The search host walks P(k) = A0 + k*(8B) and reports the integer k of a
   match. It holds no secret: recovering a0 from A0 is the discrete-log problem.
3. Locally, the key for P(k) is the scalar a0 + 8k.

The step is 8B because ed25519 clamping clears the low 3 bits of every scalar.
Tor re-clamps the scalar when it loads hs_ed25519_secret_key, so a0 + 8k must
already be clamped (a multiple of 8 in [2^254, 2^255)) or Tor would silently
load a different key. a0 is drawn HEADROOM inside that range, so any
|8k| < HEADROOM is safe; combine() still checks instead of assuming.
"""

import secrets

import nacl.bindings as sodium

from .ed25519 import point as E

CLAMP_LOW = 1 << 254
CLAMP_HIGH = 1 << 255
HEADROOM = 1 << 250
STEP = E.scalar_mult(8, E.BASE)


def is_clamped(scalar: int) -> bool:
    return scalar % 8 == 0 and CLAMP_LOW <= scalar < CLAMP_HIGH


def random_start_scalar() -> int:
    """A uniformly random clamped scalar with HEADROOM to spare on both sides."""
    choices = (CLAMP_HIGH - CLAMP_LOW - 2 * HEADROOM) // 8
    return CLAMP_LOW + HEADROOM + 8 * secrets.randbelow(choices)


def public_point(scalar: int) -> bytes:
    """scalar*B for a SECRET scalar, via libsodium's constant-time base multiplication.

    The pure-Python reference is variable-time: fine for public math, not for secrets.
    """
    if not 0 < scalar < CLAMP_HIGH:
        raise ValueError("secret scalar out of range")
    return sodium.crypto_scalarmult_ed25519_base_noclamp(scalar.to_bytes(32, "little"))


def start_point(a0: int) -> bytes:
    """The public start point A0 = a0*B, the only thing a search host receives."""
    return public_point(a0)


def point_at(a0_point: E.Point, k: int) -> E.Point:
    """The candidate a search host tests at offset k: A0 + k*(8B)."""
    return E.add(a0_point, E.scalar_mult(k, STEP))


def combine(a0: int, k: int) -> int:
    """The secret scalar of point_at(A0, k), refusing any offset Tor would mangle."""
    if not isinstance(k, int):
        raise TypeError(f"offset must be an integer, got {type(k).__name__}")
    scalar = a0 + 8 * k
    if not is_clamped(scalar):
        raise ValueError("offset moves the scalar out of the clamped range; "
                         "Tor would load a different key")
    return scalar
