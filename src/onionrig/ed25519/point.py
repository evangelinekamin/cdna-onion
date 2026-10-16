"""The ed25519 group: -x^2 + y^2 = 1 + d*x^2*y^2 over GF(2^255 - 19).

Points are immutable extended coordinates (X:Y:Z:T) with x = X/Z, y = Y/Z and
x*y = T/Z (Hisil, Wong, Carter, Dawson 2008). The a = -1 formulas below are
complete, so add() needs no special cases for doubling or the identity.
"""

from typing import NamedTuple

from . import field as F
from .field import P

D = F.mul(F.neg(121665), F.inv(121666))
D2 = F.add(D, D)
L = 2**252 + 27742317777372353535851937790883648493  # prime order of the base point


class Point(NamedTuple):
    X: int
    Y: int
    Z: int
    T: int


IDENTITY = Point(0, 1, 1, 0)


def from_affine(x: int, y: int) -> Point:
    return Point(x % P, y % P, 1, x * y % P)


def add(p: Point, q: Point) -> Point:
    """add-2008-hwcd-3 (EFD, twisted Edwards with a = -1)."""
    a = (p.Y - p.X) * (q.Y - q.X) % P
    b = (p.Y + p.X) * (q.Y + q.X) % P
    c = p.T * D2 % P * q.T % P
    d = 2 * p.Z * q.Z % P
    e, f, g, h = b - a, d - c, d + c, b + a
    return Point(e * f % P, g * h % P, f * g % P, e * h % P)


def double(p: Point) -> Point:
    """dbl-2008-hwcd (EFD) with a = -1."""
    a = p.X * p.X % P
    b = p.Y * p.Y % P
    c = 2 * p.Z * p.Z % P
    e = ((p.X + p.Y) ** 2 - a - b) % P
    g = (b - a) % P
    f = (g - c) % P
    h = (-a - b) % P
    return Point(e * f % P, g * h % P, f * g % P, e * h % P)


def neg(p: Point) -> Point:
    return Point((-p.X) % P, p.Y, p.Z, (-p.T) % P)


def scalar_mult(k: int, p: Point) -> Point:
    """k*p for any integer k (negative k multiplies -p), MSB-first double-and-add."""
    if k < 0:
        return scalar_mult(-k, neg(p))
    result = IDENTITY
    for bit in bin(k)[2:]:
        result = double(result)
        if bit == "1":
            result = add(result, p)
    return result


def to_affine(p: Point) -> tuple[int, int]:
    zi = F.inv(p.Z)
    return p.X * zi % P, p.Y * zi % P


def equal(p: Point, q: Point) -> bool:
    """Projective equality: X1*Z2 = X2*Z1 and Y1*Z2 = Y2*Z1."""
    return (p.X * q.Z - q.X * p.Z) % P == 0 and (p.Y * q.Z - q.Y * p.Z) % P == 0


def encode(p: Point) -> bytes:
    """RFC 8032 encoding: little-endian y with the low bit of x in bit 255."""
    x, y = to_affine(p)
    return (y | (x & 1) << 255).to_bytes(32, "little")


def recover_x(y: int, sign: int) -> int:
    """Solve the curve equation for x given y and the wanted low bit of x."""
    yy = y * y % P
    u = (yy - 1) % P
    v = (D * yy + 1) % P
    x = u * pow(v, 3, P) * pow(u * pow(v, 7, P), (P - 5) // 8, P) % P
    vxx = v * x * x % P
    if vxx == u:
        pass
    elif vxx == (-u) % P:
        x = x * F.SQRT_M1 % P
    else:
        raise ValueError("no curve point has this y coordinate")
    if x == 0 and sign:
        raise ValueError("invalid encoding of x = 0 with the sign bit set")
    return P - x if x & 1 != sign else x


def decode(data: bytes) -> Point:
    """RFC 8032 section 5.1.3 decoding; raises ValueError on any invalid input."""
    y = F.from_bytes(data, strict=True)
    return from_affine(recover_x(y, data[31] >> 7), y)


BASE = decode(bytes.fromhex("58" + "66" * 31))
