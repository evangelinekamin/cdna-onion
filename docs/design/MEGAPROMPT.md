# Competitor Brief: Maximize ed25519 Vanity-Onion Throughput on AMD CDNA

This is the shared brief handed identically to every competing model. Your
per-model cover note (if any) names your assigned exploration angle; everything
here is common ground. Read it with the corpus in `docs/` (manifest in sec 7).

---

## 0. The task in one paragraph

Build the fastest *correct* Tor v3 vanity `.onion` address generator on AMD
Instinct GPUs. This is a legitimate, long-standing open-source category (mkp224o,
honion, onion-gen). The hot loop is a batched ed25519 keypair grind: walk a point
by repeated additions, test each candidate's address prefix, and emit the key on
a match. You win by producing more verified candidate addresses per second than
every other entry on the same GPU -- nothing else.

**Why speed is everything.** Each extra character of prefix makes a match 32
times rarer, so the work grows exponentially with length, and throughput converts
directly into longer addresses: every 32x of speed buys one more character in the
same time. Your rank is your mean verified rate, and the board also shows it as
the median time to find a 10- and a 13-character prefix on one GPU (the baseline
needs about 6 days and 560 years on the MI210).

## 1. The arena and the hardware

- **Round 1 target (primary): AMD MI210, gfx90a, CDNA2.** Cheap to iterate on.
  Tune here first.
- **Scale target: AMD MI355X, gfx950, CDNA4.** The winner(s) re-tune here later.
- **What has been measured (`docs/design/CDNA-PERF-REFERENCE.md` section 8):**
  on the MI210's vector units, the 32x32+64 integer multiply-add and the FP64
  FMA issue at the same rate, so FP64 holds no vector-unit edge there. The MI210's
  matrix units (FP64 at 45.3 TF, a separate pipe) are untried for this workload.
  The MI355X is not measured yet, and its FP64 matrix rate equals its vector
  rate, so the best design may differ by part: per-arch variants are welcome.
- The workload is compute-bound on the vector pipe and capped by VGPR occupancy,
  not memory-bound. Whether matrix units, FP64 or exact integer, can carry part
  of the field arithmetic is an open question.

## 2. The workload you are optimizing (get this exactly right)

Full detail in `docs/design/ARCHITECTURE.md` sec 2. The essentials:
- Address = `base32(PUBKEY[32] || CHECKSUM[2] || VERSION[1])`. A prefix of up to
  49 characters (the contract's limit) depends **only on y**, because the x sign
  bit sits in character 49. No SHA3 or base32 is needed per candidate. PUBKEY is
  the little-endian compressed point, so the prefix lands in the **low** bytes of
  y; match there.
- Clamping forces valid scalars to differ by 8, so walk `P <- P + 8B` (one point
  addition per candidate, not a scalar multiply).
- Point adds yield projective coords; recovering affine y needs an inversion ->
  amortize with **Montgomery batch inversion** (one inversion per group of N).
  Best published so far, and not known to be optimal: y-only arithmetic; the
  dual addition law (2 candidates/add, ~5.5 muls); affine beats Jacobian for N>20.
- Field is F_p, p = 2^255-19. The limb representation is the main design fork
  (integer radix-2^32 vs FP64 radix-2^51 vs FP64-matrix).

## 3. What you build (the contract)

Your entry is an executable that implements `docs/design/CONTRACT.md` exactly,
so one rig drives every entry identically:
- `--info`: one JSON line naming the entry (put your model name in `model`).
- `--verify VECTORS_DIR`: your self-checks against the shared vectors in
  `vectors/`. At minimum, both known-answer grinds must find exactly the listed
  hits, negative offsets included.
- `--grind`: the newline-delimited JSON protocol. You get a fresh public start
  point `A0` and target prefixes, walk `A0 + k*(8B)`, and report the signed
  offsets `k` that match. Benchmarks, conformance, and real split-key searches all
  use this one protocol. You never see or produce a secret.

`entries/python-ref/entry.py` is the whole contract in executable form (slow but
exact), and `entries/baseline/` is a fast CUDA/HIP implementation of it. Before
any benchmark, `submit check` must pass (section 8): it runs `onionrig check` on
the real GPU.

## 4. How you are scored

- **Proof-of-work, not self-report.** The rig verifies every offset you report
  (with libsodium, not your code) and computes
  `rate = verified hits / (match probability * rig-timed seconds)`. You cannot
  report your way to a better number; you can only do more real work.
- **Real-search conditions.** At most 4 targets and about 2,000 hits/s, so
  per-hit work barely counts, just as in a real 10-character search. Your
  per-hit cold path can be as careful as you like.
- **Honest repetition.** Runs start from a cooled GPU, are thermally sampled, and
  repeat until the mean is precise. The mean is reported, never the min.
  Comparisons are paired ABBA duels on the same card: `onionrig duel BASE YOURS`
  gives your speedup with a 95% interval, and "faster" requires the whole
  interval above 1.
- The rank is the mean verified rate on the round's GPU, also shown as the median
  time to a 10- and a 13-character prefix.

## 5. Hard rules (correctness, anti-cheat, confinement)

- **One bad run fails the entry.** The rig verifies every offset. A single wrong
  hit voids the run, and any invalid run (wrong hit, crash, timeout, protocol
  violation) fails your whole benchmark: no leaderboard row on that GPU. An entry
  that is right most of the time hands users wrong keys the rest of the time.
- **Fresh randomness every run.** A new `A0` and new targets, so precomputed
  answers, hard-coded prefixes, and reused keys cannot score; they are also banned
  by diff audit.
- **Confinement.** Your entry is built and run inside a container that sees
  only its own directory. It runs without network access, inside a PID
  namespace (any process you start dies with you), with a scrubbed environment
  and a private empty `HOME`, one visible GPU, and two pinned CPU cores. Grinding
  on the host CPU or anywhere else is not allowed. Timing and correctness belong
  to the rig, never to your own reports.
- **Write your own code.** Ideas and parameters from anywhere are fair game, and
  you may read the web, but do not paste code from the internet or from GPU MODE
  submissions (most carry no license). Entries must be publishable under MIT;
  say what inspired you.
- **Bounded output.** Stay well under the limits in `CONTRACT.md` (hit lines,
  offsets per session, `--info`/`--verify` output). EOF on stdin means stop.

## 6. Where to be creative (this is the point)

Do **not** just port the reference kernel and stop. The reference is a deliberately
previous-generation baseline; beating it meaningfully means *restructuring*. Open
problems worth attacking (your cover note may point you at one):
- **FP64-matrix (rocWMMA/MFMA) field multiply** on MI210 -- nobody has done this
  for ECC; the 2x matrix rate is sitting there (see `gpumode-craft.md`,
  Berthomieu arXiv 2601.07508).
- **Branchless modular reduction** to kill warp divergence (ZKProphet 2509.22684).
- **Deferred normalization** -- test the prefix on fewer limbs before the full
  affine recovery.
- **More candidates per shared subexpression** -- can a k-add beat the dual law's
  2-per-add?
- **Host/device restructure** -- the batch scheme, work distribution, or the
  grind/emit split are all yours to redesign.
- **Per-arch specialization** -- `#ifdef __gfx90a__ / __gfx950__`; the right
  answer is likely two different kernels.
- **Something else entirely.** This list is a starting point, not a menu. An idea
  of your own, however unusual, is exactly what this contest is for.

Different approaches are explicitly wanted. A field of structurally identical
entries is a failed contest.

## 7. What you are given (corpus manifest)

In this repo (all shareable; ideas and *parameters* from the literature are fair
game, and the baseline is MIT):
- `docs/design/CONTRACT.md`: the exact protocol and rules. Read this first.
- `docs/design/ARCHITECTURE.md`: system design, the crypto, split-key, vectors.
- `docs/design/CDNA-PERF-REFERENCE.md`: the per-generation CDNA performance model
  and **section 8, what has been measured so far**. For example, the faithful
  NVIDIA port runs at occupancy 1 wave/SIMD on gfx90a and gfx950 (256 VGPRs,
  ~31 KB scratch per lane), and the MI210 lowers its effective clock under dense
  64-bit ALU work, more so the more registers each instruction reads.
- `docs/research/literature.md`, `tools.md`, `competitions.md`: the research sweep.
- `docs/research/gpumode-craft.md`: actionable CDNA kernel-tuning craft
  (VGPR/occupancy levers, branchless reduction, the rocprof-compute workflow).
- `docs/*.pdf`: the AMD CDNA2/4/5 ISA references and whitepapers (if missing,
  `tools/fetch-amd-docs.sh` downloads them from AMD).
- `entries/baseline/`: onion-gpu ported to the contract (your calibration floor:
  1.45 G candidates/s measured on the MI210), with `make resource-usage` and
  `make isa` for AMD static analysis.
- `entries/python-ref/`: the contract as a readable reference entry.
- `tools/microbench/`: sustained throughput of each primitive (`v_mad_u64_u32`,
  `v_fma_f64`, ...), with the clock measured inside each kernel. MI210 results
  are in `CDNA-PERF-REFERENCE.md` section 8; datasheets do not give these.
- `vectors/`: the shared correctness vectors (JSON and flat text).

## 8. How you work and submit

Your sandbox has the ROCm toolchain but no GPU. Compile for gfx90a (and gfx950)
locally, read your register use, occupancy, and ISA (`make resource-usage` and
`make isa` in `entries/baseline/` show how), and use the Python reference entry
as a correctness oracle. GPU runs go through the submission runner:
- `submit check` builds your entry on the MI210 and runs `onionrig check`
  (conformance, about a minute).
- `submit bench` builds it and runs `onionrig bench`, the official measurement
  (about five minutes).

Each returns structured results: pass or fail per check, rates, and short error
text. Submissions are rate-limited per model, and the runner's measurements are
the only official ones. Put your model name in `--info` for attribution.
