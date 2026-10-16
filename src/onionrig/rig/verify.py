"""Independent verification of reported hits: does A0 + k*(8B) match a target?

The hot path uses libsodium (via PyNaCl), a separate, widely deployed ed25519
implementation, so a bug shared between an entry and this repo's reference
math cannot slip through. Verification fans out over a process pool, so the rig
keeps up with tens of thousands of hits per second.
"""

from collections.abc import Iterable, Sequence
from concurrent.futures import Executor

import nacl.bindings as sodium
from nacl.exceptions import CryptoError

from .. import onion
from ..ed25519 import point as E

L = E.L


def _scalar_bytes(value: int) -> bytes:
    return (value % L).to_bytes(32, "little")


class HitVerifier:
    """Checks offsets against one start point and target set. Picklable."""

    def __init__(self, a0: bytes, targets: Sequence[onion.Target]):
        try:
            E.decode(a0)
        except ValueError as exc:
            raise ValueError(f"a0 is not a valid point encoding: {exc}") from exc
        if not targets:
            raise ValueError("need at least one target")
        self.a0 = bytes(a0)
        self.targets = tuple(targets)
        # Group target values by length, so each candidate needs one bit
        # extraction per distinct length and a set lookup.
        by_bits: dict[int, set[int]] = {}
        for t in self.targets:
            by_bits.setdefault(t.bits, set()).add(t.value)
        self._by_bits = tuple((bits, frozenset(values)) for bits, values in sorted(by_bits.items()))

    def candidate(self, k: int) -> bytes:
        """The RFC 8032 encoding of A0 + k*(8B), computed by libsodium."""
        if (8 * k) % L == 0:
            return self.a0
        step = sodium.crypto_scalarmult_ed25519_base_noclamp(_scalar_bytes(8 * k))
        return sodium.crypto_core_ed25519_add(self.a0, step)

    def matches_pubkey(self, pubkey: bytes) -> bool:
        head = int.from_bytes(pubkey, "big")
        return any((head >> (256 - bits)) in values for bits, values in self._by_bits)

    def matches(self, k: int) -> bool:
        try:
            return self.matches_pubkey(self.candidate(k))
        except CryptoError:
            return False

    def failures(self, offsets: Iterable[int], limit: int | None = None) -> list[int]:
        """The offsets that do NOT match (at most `limit`); empty means all are genuine."""
        bad: list[int] = []
        for k in offsets:
            if not self.matches(k):
                bad.append(k)
                if limit is not None and len(bad) >= limit:
                    break
        return bad


def _failures_task(verifier: HitVerifier, offsets: list[int], limit: int | None) -> list[int]:
    return verifier.failures(offsets, limit)


def parallel_failures(verifier: HitVerifier, offsets: Sequence[int], pool: Executor | None,
                      chunk: int = 2048, limit: int | None = None) -> list[int]:
    """failures() over a process pool in chunks (serial when pool is None), stopping
    early once `limit` failures are known."""
    if pool is None or len(offsets) <= chunk:
        return verifier.failures(offsets, limit)
    futures = [pool.submit(_failures_task, verifier, list(offsets[i:i + chunk]), limit)
               for i in range(0, len(offsets), chunk)]
    bad: list[int] = []
    for future in futures:
        bad += future.result()
        if limit is not None and len(bad) >= limit:
            for pending in futures:
                pending.cancel()
            return bad[:limit]
    return bad
