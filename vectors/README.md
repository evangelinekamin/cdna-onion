# Shared correctness vectors

Generated deterministically by `onionrig vectors` and cross-validated against
libsodium, `cryptography`, and plain integer arithmetic (`tests/test_vectors.py`).
`MANIFEST.json` holds the SHA-256 of every file. `onionrig vectors --check`
regenerates everything and confirms the files on disk match byte for byte.

Each file comes in two forms with the same content:
- `.json`: canonical and self-describing.
- `.txt`: one case per line, space-separated, for C/HIP entries. Lists are written
  as a count followed by the items. Header comments give each op's columns.

Encodings: field elements and points are 32-byte little-endian hex (RFC 8032),
offsets `k` are signed decimal, and 64-bit fast-path masks are 16 big-endian hex
digits over pubkey bytes 0..7.

| File | What it pins down |
|---|---|
| `field` | GF(2^255-19) add, sub, mul, sqr, neg, inv on edge values for every common limb radix (2^25.5, 2^32, 2^51, 2^64), plus decoding of non-canonical inputs |
| `point` | `k*B`, `P+Q`, `P-Q` (for dual-addition-law implementers), and `2P` |
| `decode` | RFC 8032 point decoding, including invalid encodings that must be rejected |
| `chain` | the incremental walk `A0 + i*(8B)` for 64 consecutive `i` |
| `offset` | `A0 + k*(8B)` for signed and large `k` (up to about 2^50): the exact rule the rig verifies hits with |
| `onion` | public key to address, target to 64-bit mask, and match/no-match pairs that catch the little-endian byte-order trap |
| `grind` | **known-answer grinds**: every `k` in a window (negative offsets included) whose candidate matches the targets. `--verify` must find exactly these hits. `kat_long` uses 7- to 33-character targets cut from real candidates in its window, so a matcher broken past the 12-character fast path fails it. |

The known-answer grinds are the end-to-end test that matters most. They do not
care about your limb representation, coordinate system, or batching, only whether
you test the right candidates and match them correctly.
