# Entry Contract: `onionrig-grind/1`

Every competing entry is a single executable that implements this contract.
The rig, the conformance suite, benchmarking, and real searches all drive
entries through it and nothing else, so an entry's internals are entirely its
own business.

## Core idea: proof-of-work scoring

The rig never trusts an entry's own speed claim. It hands the entry a fresh
public start point `A0` and some short target prefixes, and the entry reports
the offsets `k` whose candidate point matches a target. The rig verifies
**every** reported `k` independently. A matching candidate takes, on average,
`1 / p` attempts to find, where `p` is the exact probability that a random key
matches some target. So

    throughput = verified_unique_hits / (p * wall_seconds)

can only be raised by doing more real work. It cannot be inflated by
misreporting, and the fresh `A0` per run makes precomputation useless.

## The candidate rule (the only math an entry must get right)

    B8      = 8 * B                      (B = ed25519 base point)
    P(k)    = A0 + k * B8                (k is a signed integer, |k| < 2^63)
    pubkey  = RFC 8032 encoding of P(k)  (32 bytes: little-endian y; x sign in the top bit of byte 31)
    address = base32(pubkey || checksum || 0x03), lowercase

`k` matches a target `t` when `address` starts with `t`. Targets are 1..49
characters. The x sign bit is the top bit of byte 31, which is stream bit 248,
inside character 49. So characters 0..48 depend only on `y`: no x, no SHA3, and
no checksum is ever needed to test a candidate.

**Byte-order trap.** base32 reads the byte stream MSB-first, and the pubkey is
little-endian. Character `i` is stream bits `[5i, 5i+5)`, so a prefix
constrains the low bytes of `y`. For prefixes of at most 12 characters the test
is one masked compare on `pubkey[0..7]` read as a big-endian u64:
`mask = (2^(5n) - 1) << (64 - 5n)` and `value = target_bits << (64 - 5n)`.
`vectors/onion.txt` lists these masks.

## Invocations

Every invocation runs confined (see Rules). The entry sees exactly one GPU (use
device 0), runs on a couple of pinned CPU cores, has no network, and gets a
private empty `HOME`/`TMPDIR` and only runtime-relevant environment variables.
stderr is free-form logging; the rig keeps the first 64 MiB and never parses it.

### `ENTRY --info`
Print one JSON object on one line, exit 0.

    {"contract": "onionrig-grind/1", "name": "...", "version": "...",
     "model": "<AI model that wrote it, if any>", "device": "<detected GPU>"}

`contract`, `name`, and `version` are required.

### `ENTRY --verify VECTORS_DIR`
Run self-checks against the shared vectors and print one JSON summary line:
`{"passed": N, "failed": M, "checks": [...]}`. Exit 0 if and only if `M == 0`.

At minimum, run both known-answer grinds in `grind.txt`/`grind.json`
(`op = kat`). For each one, test every `k` in `[lo, hi)` (the window includes
negative offsets) and find exactly the listed hits: no misses and no extras.
The field, point, chain, offset, and onion vectors are there to help you
develop.

### `ENTRY --grind`
Newline-delimited JSON on stdin/stdout. One message per line.

1. **Entry → rig:** `{"event": "ready"}` once all setup that does not depend on
   `A0` is done (compiling kernels, allocating memory, building tables). The rig
   allows a setup timeout of 300 s by default.
2. **Rig → entry:**
   `{"cmd": "start", "a0": "<64 hex>", "targets": ["ab", "q7", ...], "seconds": 60.0}`
   - `a0` is a valid RFC 8032 point encoding. If it does not decode, reply with
     `error`.
   - `targets` holds 1..256 prefixes, already lowercase and prefix-free.
   - `seconds` is the grind duration, or `null` to run until stopped.
3. **Entry → rig, any number of times:** `{"event": "hits", "k": ["123", "-45"]}`.
   Offsets are signed decimal strings, to avoid JSON float precision loss. Batch
   freely, but flush at least about once per second while hits are pending.
   **Optional:** `{"event": "progress", "candidates": "<decimal>"}` reports the
   number of candidates tested so far, for example every few seconds. The rig
   only displays it (long real searches can run for days without a hit) and
   never uses it for scoring.
4. **Rig → entry, optional:** `{"cmd": "stop"}`. Stop starting new work, flush
   pending hits, and finish within the grace period (10 s by default). **EOF on
   stdin means the same as `stop`**: the rig has gone away. Keep each unit of GPU
   work short (around 1 s) so a stop is honored promptly on slow GPUs too.
5. **Entry → rig:** `{"event": "done", "candidates": "<decimal>"}`, then exit 0.
   `candidates` is the number of `k` values the entry tested. It is checked
   against the verified hit count as a statistical sanity test (it flags bugs
   such as skipping half of a dual-law pair), but it never sets the score.

On any failure, send `{"event": "error", "message": "..."}` and exit nonzero.

## Rules (enforced by the rig)

- **One bad run fails the entry.** Every reported `k` is verified. A `k` that
  does not match or does not parse voids the run. Any invalid run (a wrong hit, a
  crash, a timeout, a protocol violation) fails the whole benchmark or duel, and
  the entry gets no leaderboard row on that GPU. Averaging only the runs that
  happened to go well would rank an entry that hands users wrong keys. Duplicate
  `k` values are counted once and flagged.
- **Bounded output.** At most 1 MiB from `--info`/`--verify`, at most 16 MiB per
  protocol line, and at most 1,000,000 offsets per session. A matcher that
  reports everything is stopped as a hit flood.
- **The rig owns time.** `wall_seconds` runs from writing `start` to reading
  `done`. Grinding past `seconds` only adds to the denominator. If the entry
  overruns `seconds` plus the grace period, it is stopped and then killed.
- **Fresh randomness every run.** Each run draws a new `A0` and new targets, so
  nothing computed before `start` can produce a valid hit.
- **One GPU, pinned CPU cores, no network, no stragglers.** Grinding on the
  host CPU or on remote machines is not allowed. By default the rig enforces
  this without root:
  - GPU visibility is narrowed to one device.
  - The entry is pinned to 2 of the CPUs the rig may use. Entries on different
    GPUs of one host get different CPUs.
  - It runs in unprivileged user, network, PID, and mount namespaces: no network
    connections, and every process it starts (`setsid` escapees included) dies
    with it.
  - Its environment is scrubbed: credentials and agent sockets are dropped.

  If a host cannot create namespaces, the rig refuses to run entries unless
  `--no-isolation` is given, and records that choice.
- **What the rig cannot enforce.** An entry still sees the host filesystem as
  the invoking user. Never run entries on a machine or account that holds start
  scalars (`start-scalars.secret`) or Tor keys. Hard resource limits (cgroups)
  are the scheduler's job, for example a Slurm allocation.
- **Pre-start work is not timed, but cannot help.** Anything done before
  `ready` is free and must not depend on `A0`. That is impossible to exploit,
  because `A0` arrives only with `start`.

## Why these choices

- *Proof-of-work instead of trusting counters*: honion's methodology, and the
  only measurement that a buggy or dishonest entry cannot inflate.
- *Signed `k`*: dual-addition-law entries walk both directions from a center.
- *Short targets in benchmarks, but few of them and not too many hits*: real
  searches (10+ characters) almost never hit, which gives no signal, so
  benchmarks use short targets. The rig keeps it to at most 4 targets and about
  2,000 hits/s (still around 100k hits per minute-long run, so about 0.3%
  counting error). More would measure per-candidate target compares and per-hit
  cold paths, which a real search never pays for. That bias cost the baseline 30%
  of its measured speed before the limit was added.
