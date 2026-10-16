"""Deterministic correctness vectors: the shared ground truth every entry is built against.

Each file comes in two forms with identical content:
  <name>.json  canonical, self-describing
  <name>.txt   one case per line, space separated, for C/HIP entries (fscanf-able)

Encoding conventions (both forms):
  field elements and points: 32-byte little-endian hex (RFC 8032 encodings)
  offsets k: signed decimal
  64-bit fast-path masks: 16 hex digits, big-endian over pubkey bytes 0..7
"""

import hashlib
import json
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

from . import onion, splitkey
from .ed25519 import field as F
from .ed25519 import point as E

SCHEMA = "onionrig-vectors/1"
DEFAULT_SEED = 0x6F6E696F6E726967  # "onionrig"
FAST_PATH_CHARS = 12  # floor(64 / 5): prefixes this short fit one 64-bit compare

Case = dict[str, Any]

# Values that stress limb boundaries in every common radix (2^25.5, 2^32, 2^51, 2^64).
_EDGE = sorted({
    0, 1, 2, 19, 38, F.P - 1, F.P - 2, F.P - 19, (F.P - 1) // 2, 2**254, 2**255 - 20,
    *(2**k - 1 for k in (25, 26, 32, 51, 52, 64, 102, 128, 153, 192, 204, 255) if 2**k - 1 < F.P),
    *(2**k for k in (25, 26, 32, 51, 64, 128, 204)),
})


def _hex(value: int) -> str:
    return F.to_bytes(value).hex()


def _field_cases(rng: random.Random) -> list[Case]:
    values = _EDGE + [rng.randrange(F.P) for _ in range(48)]
    cases: list[Case] = []
    for i, a in enumerate(values):
        b = values[(i * 7 + 3) % len(values)]
        cases += [
            {"op": "add", "a": _hex(a), "b": _hex(b), "out": _hex(F.add(a, b))},
            {"op": "sub", "a": _hex(a), "b": _hex(b), "out": _hex(F.sub(a, b))},
            {"op": "mul", "a": _hex(a), "b": _hex(b), "out": _hex(F.mul(a, b))},
            {"op": "sqr", "a": _hex(a), "b": "-", "out": _hex(F.mul(a, a))},
            {"op": "neg", "a": _hex(a), "b": "-", "out": _hex(F.neg(a))},
        ]
        if a:
            cases.append({"op": "inv", "a": _hex(a), "b": "-", "out": _hex(F.inv(a))})
    # Non-canonical encodings decode to their reduction mod p (bit 255 ignored).
    for raw in (F.P, F.P + 1, F.P + 18, 2**255 - 1, 2**256 - 1):
        data = (raw % 2**256).to_bytes(32, "little")
        cases.append({"op": "frombytes", "a": data.hex(), "b": "-",
                      "out": _hex(F.from_bytes(data))})
    return cases


def _point_cases(rng: random.Random) -> list[Case]:
    scalars = [1, 2, 3, 8, 16, 255, 2**64 + 1, E.L - 1, splitkey.CLAMP_LOW,
               splitkey.CLAMP_HIGH - 8] + [rng.randrange(1, E.L) for _ in range(12)]
    cases: list[Case] = [
        {"op": "scalarmult_base", "k": str(k), "out": E.encode(E.scalar_mult(k, E.BASE)).hex()}
        for k in scalars
    ]
    for _ in range(12):
        p = E.scalar_mult(rng.randrange(1, E.L), E.BASE)
        q = E.scalar_mult(rng.randrange(1, E.L), E.BASE)
        pe, qe = E.encode(p).hex(), E.encode(q).hex()
        cases += [
            {"op": "add", "p": pe, "q": qe, "out": E.encode(E.add(p, q)).hex()},
            {"op": "sub", "p": pe, "q": qe, "out": E.encode(E.add(p, E.neg(q))).hex()},
            {"op": "double", "p": pe, "q": "-", "out": E.encode(E.double(p)).hex()},
        ]
    return cases


def _decode_cases(rng: random.Random) -> list[Case]:
    cases: list[Case] = []
    specials = [F.P.to_bytes(32, "little"), (1).to_bytes(31, "little") + b"\x80"]
    for raw in specials + [rng.randbytes(32) for _ in range(40)]:
        try:
            out = E.encode(E.decode(raw)).hex()
        except ValueError:
            out = "invalid"
        cases.append({"op": "decode", "in": raw.hex(), "out": out})
    return cases


def _chain_cases(rng: random.Random) -> list[Case]:
    """The incremental walk every entry runs: A0 + i*(8B) for consecutive i."""
    a0 = E.scalar_mult(rng.randrange(1, E.L), E.BASE)
    point, cases = a0, []
    for i in range(64):
        cases.append({"op": "chain", "a0": E.encode(a0).hex(), "i": str(i),
                      "out": E.encode(point).hex()})
        point = E.add(point, splitkey.STEP)
    return cases


def _offset_cases(rng: random.Random) -> list[Case]:
    """The rule the rig verifies every reported hit with: P(k) = A0 + k*(8B)."""
    a0 = E.scalar_mult(rng.randrange(1, E.L), E.BASE)
    ks = [0, 1, -1, 2, -2, 1000, -1000, 2**32, -(2**32), 2**50 + 7, -(2**50) - 7]
    ks += [rng.randrange(-(2**48), 2**48) for _ in range(9)]
    return [{"op": "offset", "a0": E.encode(a0).hex(), "k": str(k),
             "out": E.encode(splitkey.point_at(a0, k)).hex()} for k in ks]


def _fast_path(target: onion.Target) -> tuple[str, str]:
    shift = 64 - target.bits
    return f"{((1 << target.bits) - 1) << shift:016x}", f"{target.value << shift:016x}"


def _onion_cases(rng: random.Random) -> list[Case]:
    cases: list[Case] = []
    for _ in range(24):
        pubkey = E.encode(E.scalar_mult(rng.randrange(1, E.L), E.BASE))
        cases.append({"op": "address", "pubkey": pubkey.hex(),
                      "out": onion.address_from_pubkey(pubkey)})
    for n in range(1, FAST_PATH_CHARS + 1):
        text = "".join(rng.choice(onion.ALPHABET) for _ in range(n))
        mask, value = _fast_path(onion.parse_target(text))
        cases.append({"op": "target", "text": text, "mask64": mask, "value64": value})
    for _ in range(64):
        pubkey = rng.randbytes(32)
        address = onion.address_from_pubkey(pubkey)
        n = rng.randrange(1, 8)
        own = address[:n]
        other = own[:-1] + rng.choice(onion.ALPHABET.replace(own[-1], ""))
        for text in (own, other):
            hit = onion.matches(pubkey, onion.parse_target(text))
            cases.append({"op": "match", "pubkey": pubkey.hex(), "text": text,
                          "out": "1" if hit else "0"})
    return cases


def grind_kat(a0: E.Point, targets: tuple[onion.Target, ...], lo: int, hi: int) -> list[int]:
    """Every offset k in [lo, hi) whose candidate A0 + k*(8B) matches any target."""
    hits, point = [], splitkey.point_at(a0, lo)
    for k in range(lo, hi):
        pubkey = E.encode(point)
        if any(onion.matches(pubkey, t) for t in targets):
            hits.append(k)
        point = E.add(point, splitkey.STEP)
    return hits


def _kat(name: str, a0: E.Point, texts: tuple[str, ...], lo: int, hi: int) -> Case:
    targets = onion.normalize_targets(texts)
    return {"op": "kat", "name": name, "a0": E.encode(a0).hex(),
            "targets": [t.text for t in targets], "lo": str(lo), "hi": str(hi),
            "hits": [str(k) for k in grind_kat(a0, targets, lo, hi)]}


def _grind_cases(rng: random.Random) -> list[Case]:
    def point() -> E.Point:
        return E.scalar_mult(rng.randrange(1, E.L), E.BASE)

    cases = [_kat("kat_single", point(), ("q",), -(2**12), 2**12),
             _kat("kat_multi", point(), ("ab", "z7", "mo"), -(2**14), 2**14)]
    # Long targets cut from real candidates in the window, so every one is certain
    # to be found: a matcher broken past the 12-character fast path fails here.
    a0, lo, hi = point(), -(2**11), 2**11
    picks = rng.sample(range(lo, hi), 5)
    texts = tuple(onion.address_from_pubkey(E.encode(splitkey.point_at(a0, k)))[:n]
                  for k, n in zip(picks, (7, 12, 13, 20, 33), strict=True))
    cases.append(_kat("kat_long", a0, texts, lo, hi))
    return cases


BUILDERS: dict[str, Callable[[random.Random], list[Case]]] = {
    "field": _field_cases,
    "point": _point_cases,
    "decode": _decode_cases,
    "chain": _chain_cases,
    "offset": _offset_cases,
    "onion": _onion_cases,
    "grind": _grind_cases,
}


def _txt_line(case: Case) -> str:
    """Flatten one case: op first, then values in key order; lists become count + items."""
    fields: list[str] = []
    for value in case.values():
        if isinstance(value, list):
            fields += [str(len(value)), *value]
        else:
            fields.append(str(value))
    return " ".join(fields)


def build(seed: int = DEFAULT_SEED) -> dict[str, list[Case]]:
    # One independent stream per file, so adding cases to one file never shifts another.
    return {name: fn(random.Random(f"{seed}:{name}")) for name, fn in BUILDERS.items()}


def render(name: str, cases: list[Case]) -> tuple[bytes, bytes]:
    document = {"schema": SCHEMA, "file": name, "cases": cases}
    json_bytes = (json.dumps(document, indent=1) + "\n").encode()
    columns = sorted({" ".join(c.keys()) for c in cases})
    header = [f"# {SCHEMA} {name}", "# columns per op (lists are length-prefixed):",
              *(f"#   {cols}" for cols in columns)]
    txt_bytes = ("\n".join(header + [_txt_line(c) for c in cases]) + "\n").encode()
    return json_bytes, txt_bytes


def write_all(out_dir: Path, seed: int = DEFAULT_SEED) -> dict[str, str]:
    """Write every vector file plus MANIFEST.json; returns {filename: sha256}."""
    out_dir.mkdir(parents=True, exist_ok=True)
    digests: dict[str, str] = {}
    for name, cases in build(seed).items():
        for suffix, data in zip((".json", ".txt"), render(name, cases), strict=True):
            (out_dir / f"{name}{suffix}").write_bytes(data)
            digests[f"{name}{suffix}"] = hashlib.sha256(data).hexdigest()
    manifest = {"schema": SCHEMA, "seed": seed, "sha256": digests}
    (out_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return digests


def check(out_dir: Path) -> list[str]:
    """Problems found comparing out_dir against a fresh regeneration; empty means clean."""
    manifest = json.loads((out_dir / "MANIFEST.json").read_text())
    problems: list[str] = []
    for name, cases in build(manifest["seed"]).items():
        for suffix, data in zip((".json", ".txt"), render(name, cases), strict=True):
            path = out_dir / f"{name}{suffix}"
            on_disk = path.read_bytes() if path.exists() else b""
            if on_disk != data:
                problems.append(f"{path.name}: differs from regeneration")
            elif manifest["sha256"].get(path.name) != hashlib.sha256(data).hexdigest():
                problems.append(f"{path.name}: manifest hash mismatch")
    return problems
