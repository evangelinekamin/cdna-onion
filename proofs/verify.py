#!/usr/bin/env python3
"""Zero-dependency verifier for Tor v3 .onion Ed25519 proof-of-possession files.

Usage:
    python3 proofs/verify.py [proofs/evangeline-proofs.json]

Verifies every entry in pure Python 3 (stdlib only, no pip packages required):
  1. Decodes the 56-char base32 .onion address, checks version == 0x03 and the
     SHA3-256 Tor v3 checksum, and matches the embedded 32-byte Ed25519 pubkey.
  2. Verifies the RFC 8032 Ed25519 signature over `message` against that pubkey.
  3. If PyNaCl (libsodium) is installed, cross-checks the signature with it too.
"""

from __future__ import annotations

import base64
import hashlib
import json
import sys
from pathlib import Path

# Ed25519 parameters (RFC 8032 Section 5.1)
P = (1 << 255) - 19
L = (1 << 252) + 27742317777372353535851937790883648493
D = (-121665 * pow(121666, P - 2, P)) % P
SQRT_M1 = pow(2, (P - 1) // 4, P)

BY = (4 * pow(5, P - 2, P)) % P
BX = 15112221349535400772501151409588531511454012693041857206046113283949847762202
B = (BX, BY, 1, (BX * BY) % P)
IDENTITY = (0, 1, 1, 0)

TOR_CHECKSUM_TAG = b".onion checksum"
BASE32_ALPHABET = set("abcdefghijklmnopqrstuvwxyz234567")


def _point_add(
    p1: tuple[int, int, int, int], p2: tuple[int, int, int, int]
) -> tuple[int, int, int, int]:
    x1, y1, z1, t1 = p1
    x2, y2, z2, t2 = p2
    a = ((y1 - x1) * (y2 - x2)) % P
    b = ((y1 + x1) * (y2 + x2)) % P
    c = (2 * D * t1 * t2) % P
    d = (2 * z1 * z2) % P
    e = (b - a) % P
    f = (d - c) % P
    g = (d + c) % P
    h = (b + a) % P
    return ((e * f) % P, (g * h) % P, (f * g) % P, (e * h) % P)


def _scalar_mul(pt: tuple[int, int, int, int], k: int) -> tuple[int, int, int, int]:
    acc = IDENTITY
    addend = pt
    while k > 0:
        if k & 1:
            acc = _point_add(acc, addend)
        addend = _point_add(addend, addend)
        k >>= 1
    return acc


def _point_eq(p1: tuple[int, int, int, int], p2: tuple[int, int, int, int]) -> bool:
    x1, y1, z1, _ = p1
    x2, y2, z2, _ = p2
    return (x1 * z2 - x2 * z1) % P == 0 and (y1 * z2 - y2 * z1) % P == 0


def _decode_point(raw: bytes) -> tuple[int, int, int, int]:
    if len(raw) != 32:
        raise ValueError("point encoding must be 32 bytes")
    y_int = int.from_bytes(raw, "little")
    x_sign = y_int >> 255
    y = y_int & ((1 << 255) - 1)
    if y >= P:
        raise ValueError("non-canonical y coordinate")
    y2 = (y * y) % P
    u = (y2 - 1) % P
    v = (D * y2 + 1) % P
    # RFC 8032 Section 5.1.3 square root candidate for u/v
    v3 = (v * v * v) % P
    v7 = (v3 * v3 * v) % P
    x = (u * v3 * pow((u * v7) % P, (P - 5) // 8, P)) % P
    vx2 = (v * x * x) % P
    if vx2 == u:
        pass
    elif vx2 == (-u) % P:
        x = (x * SQRT_M1) % P
    else:
        raise ValueError("point is not on the Ed25519 curve")
    if x == 0 and x_sign == 1:
        raise ValueError("invalid sign bit for x = 0")
    if (x & 1) != x_sign:
        x = (-x) % P
    return (x, y, 1, (x * y) % P)


def verify_ed25519(pubkey: bytes, message: bytes, signature: bytes) -> None:
    if len(pubkey) != 32:
        raise ValueError("public key must be 32 bytes")
    if len(signature) != 64:
        raise ValueError("signature must be 64 bytes")
    r_bytes, s_bytes = signature[:32], signature[32:]
    s = int.from_bytes(s_bytes, "little")
    if s >= L:
        raise ValueError("signature scalar S >= L")
    a_pt = _decode_point(pubkey)
    r_pt = _decode_point(r_bytes)
    h = int.from_bytes(hashlib.sha512(r_bytes + pubkey + message).digest(), "little") % L
    lhs = _scalar_mul(B, s)
    rhs = _point_add(r_pt, _scalar_mul(a_pt, h))
    if not _point_eq(lhs, rhs):
        raise ValueError("Ed25519 signature verification failed")


def verify_onion_v3(onion_addr: str) -> bytes:
    host = onion_addr.strip().lower()
    if not host.endswith(".onion"):
        raise ValueError(f"expected .onion suffix: {onion_addr!r}")
    stem = host[:-6]
    if len(stem) != 56 or any(ch not in BASE32_ALPHABET for ch in stem):
        raise ValueError(f"invalid 56-char base32 Tor v3 address: {onion_addr!r}")
    raw = base64.b32decode(stem.upper())
    pubkey, check, version = raw[:32], raw[32:34], raw[34]
    if version != 3:
        raise ValueError(f"expected Tor v3 (version 3), got {version}")
    expected_check = hashlib.sha3_256(TOR_CHECKSUM_TAG + pubkey + b"\x03").digest()[:2]
    if check != expected_check:
        raise ValueError("Tor v3 SHA3-256 checksum mismatch")
    return pubkey


def verify_file(path: Path) -> int:
    data = json.loads(path.read_text(encoding="utf-8"))
    proofs = data.get("proofs", [])
    if not proofs:
        print(f"FAIL  {path}: no proofs found", file=sys.stderr)
        return 1

    nacl_verify = None
    try:
        from nacl.signing import VerifyKey  # type: ignore

        nacl_verify = VerifyKey
    except ImportError:
        pass

    failures = 0
    for item in proofs:
        addr = item["onion"]
        try:
            pubkey_from_addr = verify_onion_v3(addr)
            pubkey = bytes.fromhex(item["pubkey_hex"])
            if pubkey != pubkey_from_addr:
                raise ValueError("pubkey_hex does not match .onion address")
            sig = bytes.fromhex(item["signature_hex"])
            msg = item["message"].encode("utf-8")
            verify_ed25519(pubkey, msg, sig)
            if nacl_verify is not None:
                nacl_verify(pubkey).verify(msg, sig)
            engine = "pure-Python + libsodium" if nacl_verify else "pure-Python"
            print(f"PASS  {addr}  ({engine})")
        except Exception as exc:
            failures += 1
            print(f"FAIL  {addr}: {exc}", file=sys.stderr)

    if failures:
        print(f"\n{failures} of {len(proofs)} proof(s) failed.", file=sys.stderr)
        return 1
    print(f"\nAll {len(proofs)} proof(s) verified cleanly.")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    target = Path(args[0]) if args else Path(__file__).resolve().parent / "evangeline-proofs.json"
    if not target.is_file():
        print(f"Proof file not found: {target}", file=sys.stderr)
        return 2
    return verify_file(target)


if __name__ == "__main__":
    sys.exit(main())
