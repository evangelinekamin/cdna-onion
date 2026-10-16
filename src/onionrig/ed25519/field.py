"""Arithmetic in GF(p) for p = 2^255 - 19.

Elements are plain Python ints kept canonical in [0, p). This is the readable
reference that the vectors and the rig's verifier are checked against, so
clarity wins over speed.
"""

P = 2**255 - 19
SQRT_M1 = pow(2, (P - 1) // 4, P)  # 2 is a non-residue (p = 5 mod 8), so this squares to -1

_LOW_255_BITS = (1 << 255) - 1


def add(a: int, b: int) -> int:
    return (a + b) % P


def sub(a: int, b: int) -> int:
    return (a - b) % P


def mul(a: int, b: int) -> int:
    return (a * b) % P


def neg(a: int) -> int:
    return (-a) % P


def inv(a: int) -> int:
    """Multiplicative inverse by Fermat's little theorem."""
    if a % P == 0:
        raise ZeroDivisionError("0 has no inverse in GF(p)")
    return pow(a, P - 2, P)


def from_bytes(data: bytes, *, strict: bool = False) -> int:
    """Decode 32 little-endian bytes, ignoring bit 255 (the x sign in point encodings).

    strict=True rejects non-canonical values (>= p), as RFC 8032 point decoding
    requires; otherwise they are reduced mod p.
    """
    if len(data) != 32:
        raise ValueError(f"field element must be 32 bytes, got {len(data)}")
    value = int.from_bytes(data, "little") & _LOW_255_BITS
    if strict and value >= P:
        raise ValueError("non-canonical field element (>= p)")
    return value % P


def to_bytes(a: int) -> bytes:
    """Canonical 32-byte little-endian encoding."""
    return (a % P).to_bytes(32, "little")
