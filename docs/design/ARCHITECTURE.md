# onionrig - Architecture & Layout

A portable, high-throughput Tor v3 vanity `.onion` generator targeting AMD CDNA
(HIP), built so that multiple independent kernel implementations can be measured
against each other on identical hardware.

Date: 2026-10-15. Companion: `CDNA-PERF-REFERENCE.md`.

---

## 1. Goal & non-goals

**Goal.** Maximize correct candidate ed25519 addresses per second on CDNA2
(MI210, test) and CDNA4 (MI355X, scale), and publish the first rigorous keys/s
numbers for AMD Instinct on this workload. Produce one real 10-character address
as the demonstration artifact.

**Explicit non-goals.**
- Not chasing 12- or 13-char prefixes. That is an entropy wall (65 bits for 13),
  not a code problem; no kernel or hardware we have closes a ~10^4-10^9x gap.
  Target is **10 chars** (50 bits, hours-to-days on a node), with 11 as a
  stretch only if a long run is sanctioned.
- Not beating honion's ~12.5 G/s record by a large multiple via assembly. The
  large multiples (10-350x) come from *algebra over a naive baseline*; assembly/
  ISA tuning is the last ~5-15%. Our contribution is CDNA-first + rigor, not
  magic over the current record.

**Ground rules (from the prior feasibility pass, kept):**
- **Secret key never touches shared hardware.** Split-key search only (sec 7).
- **Alphabet constraint:** base32 here is `a-z` + `2-7`. Any target containing
  `0 1 8 9` or punctuation is impossible. Validate prefixes at input.
- Running personal grinds on Sol likely violates its AUP; the legitimate framing
  is a short benchmarking study with a faculty sponsor, not a week-long search.

---

## 2. The cryptography (precise; get this wrong and the kernel is fast and useless)

**Address encoding** (Tor rend-spec v3):
```
onion_address = base32( PUBKEY[32] || CHECKSUM[2] || VERSION[1] ) + ".onion"
CHECKSUM      = SHA3_256( ".onion checksum" || PUBKEY || VERSION )[:2]
VERSION       = 0x03
```
35 bytes -> 280 bits -> 56 base32 chars. base32 alphabet = RFC4648 lower, `a-z2-7`,
5 bits/char.

**What a prefix actually constrains.**
- `PUBKEY` is the 32-byte compressed ed25519 point: little-endian `y`, with the
  sign bit of `x` in the MSB of byte 31.
- base32 char `i` covers bits `[5i, 5i+5)` of the byte stream, and `PUBKEY` is
  first, so an N-char prefix (N <= 51) depends **only on PUBKEY**, never on the
  checksum. SHA3 + base32 run only on a hit.
- **Byte order trap:** char 0 = top 5 bits of byte 0 = the *least significant*
  bits of `y` (little-endian). The prefix lands in the **low** limbs of the
  encoded point. The matcher compares low bytes of the 32-byte LE encoding. This
  is correctness-vector #4; a kernel that compares the wrong end runs fine and
  finds wrong addresses.

**Why it parallelizes (the source of all the speed).**
- Pubkey `A = a*B`, `a` the clamped scalar, `B` the base point. ed25519 clamping
  clears the low 3 bits of `a`, so consecutive valid scalars differ by **8**.
- So walk `A_{k+1} = A_k + 8B` (precompute `8B`): one **point addition** per
  candidate instead of a full scalar multiply (~250x less work).
- Point add is done in extended twisted-Edwards coords -> projective
  `(X:Y:Z:T)`. Recovering affine `y = Y/Z` needs a field **inversion** (~254
  mul-equivalents). Amortize with **Montgomery batch inversion**: one inversion
  shared across a group of N candidates, + 3(N-1) mults -> inversion cost per
  candidate -> ~0 as N grows.
- **y-only:** the match needs only encoded-`y` low bits, so `x` / the sign bit /
  SHA3 are deferred until the coarse y-prefix matches.
- **Dual addition law (honion):** `y(P+Q)` and `y(P-Q)` share multiplications ->
  two candidates per add, ~5.5 muls/address. onion-gpu's plainer y-only path is
  ~11.7 muls/key. Either is a valid starting algebra.
- **Multi-prefix is ~free:** k acceptable prefixes divide expected time by k.
  Prefer "10 chars + flexibility" (variants / any Nth char) over a rigid 11.

Field is F_p, `p = 2^255 - 19`. Limb representation is the swappable backend
(sec 5); the group/match/emit layers are written once against it.

---

## 3. Module decomposition

Four layers, one deliberately swappable seam (the field backend):

```
                                        owns the number    varies per model
  [4] host harness  ----------------------------------------------  C++/HIPRTC
       CLI, prefix validation, scalar seeding (split-key), launch,
       timing, JSONL streaming, resume, verify
  [3] match + emit  ----------------------------------------------  shared
       masked low-limb compare; multi-prefix table (sorted/Bloom);
       on hit only: full encode + SHA3-256 checksum + base32
  [2] group arithmetic  ------------------------------------------  shared*
       extended-Edwards point add; incremental +8B walk;
       Montgomery batch inversion (group N); y-only; (opt) dual-add
  [1] field arithmetic  ============ THE SEAM =====================  per model
       F_p mul/sqr/add/sub/inv over p=2^255-19
       backend A: integer radix-2^32 on V_MAD_U64_U32
       backend B: FP64 radix-2^51 on V_FMA_F64
```
\*Layer 2's dual-add vs plain y-only is itself a choice a challenger may make; it
is "shared" in that the harness and vectors treat it as a black-box generator.

**Repo layout (as built).** The rig is Python, because it orchestrates
processes, statistics, and verification and never touches the hot loop. Entries
are whatever their authors choose; the baseline is CUDA + HIP from one source.
```
onionrig/
  src/onionrig/        # the rig: reference ed25519, vectors, protocol, verify, score, CLI
  entries/python-ref/  # the contract as a slow, exact reference entry
  entries/baseline/    # onion-gpu ported to the contract (calibration floor)
  tools/microbench/    # primitive throughput per GPU (settles the FP64 fork)
  tools/sol/           # one-shot MI210 / any-GPU job scripts
  vectors/             # shared correctness vectors, JSON + flat text, content-hashed
  docs/design/         # this doc, CONTRACT.md, CDNA-PERF-REFERENCE.md, MEGAPROMPT.md
  tests/               # pytest, including real-GPU and real-Tor integration tests
  bench/, search/      # results (git-ignored), one JSON line per record
```

---

## 4. The contest: a conformance rig, not a plug-in cage

The tension to resolve: a plug-in kernel interface gives perfectly apples-to-
apples numbers but *forbids pipeline restructuring* -- and pipeline restructuring
is exactly the kind of "different models see it differently" idea we most want to
invite. So the seam is **not** a code interface each model plugs into. It is an
**outer conformance + benchmark contract** that each model's *own end-to-end
repo* must satisfy. Separate repos, unified only at the edges.

What is shared (the thin, fixed contract):
- **A CLI surface** every entry must implement, specified in `CONTRACT.md`:
  `--info`, `--verify VECTORS_DIR`, and `--grind`, a single stdin/stdout
  protocol that serves conformance, benchmarking, and real split-key searches
  alike. The rig measures speed by proof-of-work; entries never report it.
  Internals behind this are entirely the model's.
- **The correctness vectors** (sec 6), language/representation-neutral JSON plus
  flat text in `vectors/`. Pass all of them or you are not ranked; a fast wrong kernel
  scores zero.
- **The measurement rig:** same GPU, driver, ROCm version, warm-up, and window
  length for every entry; it runs each entry's own binary and records the number.
- **A shared megaprompt** (see below) carrying identical context to every model.

What is free (everything that makes it a real contest):
- The entire host/device split, batching scheme, coordinate system, field
  representation, language (C++ or Rust host), even the choice of algebra. A
  model that wants to restructure the whole pipeline is *encouraged* to.

This keeps the honesty of a shared harness (identical vectors + identical timing
on identical hardware => a keys/s delta is real) while giving the pipeline
freedom of separate repos. The rig measures the *outcome*; it does not constrain
the *design*.

**The megaprompt** is its own deliverable and its own challenge. It must: hand
over these two design docs + the crypto (sec 2) + the perf reference + the FP64
fork + the ground rules; state the measurable goal (max verified candidates/s on
MI210 and MI355X, must pass all vectors); define the CLI contract above; and
*explicitly invite* pipeline restructuring and novel/ workload-specific algebra
while forbidding correctness shortcuts (dropping the checksum, comparing the
wrong end, skipping a vector). Drafting it well is worth real effort; it is the
one input every competitor shares.

**Design against convergence (solution diversity).** Handing many models one
prompt risks them collapsing onto the same obvious build (HIPIFY the reference,
textbook dual-add, stop) -- the mode-collapse / shared-prior failure. Counter it
deliberately: (a) welcome *good* convergence (Montgomery inversion, extended
coords -- correct and near-optimal; do not fight it for novelty's sake) but
penalize *structural* sameness by scoring solution diversity, not only keys/s;
(b) give each model a *different framing/angle* on the identical task (FP64 path,
integer path, deferred-normalization / cheaper prefix test, host-device
restructure) to de-correlate starting points; (c) prefer a roster of **different
model families** (different pretraining priors explore different regions) over
several snapshots of one lineage; (d) back novelty with process -- require a
"tried and rejected" log and/or keep the most divergent of several attempts. The
deep prior toward known-good algebra is fine for the ~90%; the open onion-specific
tricks are the ~10% where that prior must be actively pushed past, so the
megaprompt isolates those sub-problems and aims different models at different ones.

**Correctness gating & anti-cheat (adapted from GPU MODE / mlx.fast).**
- Correctness is a hard NULL gate, never a penalty: every emitted key must be a
  valid ed25519 point that encodes to a correct v3 address (checksum + version)
  AND matches the requested prefix. One bad key -> zero score for the run.
- **Re-verify on FRESH randomness every ranked timing iteration:** re-seed the
  scalar RNG and/or re-draw the target prefix each iteration (seed-stepped) and
  re-check the hits. This is the decisive defense -- it voids precomputed-answer,
  hard-coded-prefix, and memorized-input cheats outright.
- Banned tactics, enforced by a diff audit (not only the sandbox): precomputing
  or embedding a hit set, hard-coding the target prefix, reusing keys across
  iterations, skipping the checksum/verify. 
- Sandbox the kernel (ephemeral uid, no network, workspace-confined writes, env
  stripped) behind a trusted scorer that owns timing + correctness; score only
  from the scorer's output, never from files the kernel left behind.

**Submission & promotion.**
- An entry submits only its editable paths (tarball); the solver's model name is
  recorded on submit (that is how a result attributes to a model). Freeze the
  rest; re-enforce server-side after upload.
- One metric, one ranked track, a versioned contract: a new contract version
  starts a fresh leaderboard (no cross-version migration).
- Promotion = cumulative baseline: a verified win becomes the new published floor
  every later entry must beat, kept in an inspectable progress record.
- A local estimate mode mirrors the gates but is non-authoritative; the async
  measurement queue on the real MI210 / MI355X is the sole authority.

**Baseline + challenger (your Q2 choice).** Recon (`docs/research/tools.md`)
settles the baseline: **HIPIFY NIKIB0Y/onion-gpu** -- the only reference that is
plain self-contained CUDA C++ (so `hipify-clang` converts near-mechanically to
gfx90a/gfx950), MIT-licensed, and ships a free calibration+correctness harness
(`--bench`, RFC-8032 self-tests, a differential test against double-and-add,
per-hit re-derivation, `verify.py`). Its algebra is deliberately previous-
generation (ref10 10x25.5 limbs, single moving point, ~12 muls/candidate, no
y-only) -- which makes it an *honest* floor: the best published algebra (dual
law, y-only) needs about half the field muls (~12 -> ~5.5), and the gap between
this floor and the entries measures what algebra and tuning buy on the same AMD
card. It is not an entry; it calibrates the rig.
Second, zero-port AMD reference: **0xROOTPLS/prefix32** already runs on AMD via
ROCm's OpenCL runtime (dual-law, 4x u64, benchmarked on an RX 6800 XT), giving an
existing SOTA-class AMD datapoint to calibrate against with no porting (but do
*not* copy its secret hygiene -- it keeps secret material in VRAM and writes keys
world-readable). Each model's clean-room build is then its own conforming repo.
Spec the clean-room kernel from honion's docs (the clearest public dual-law
write-up); on CDNA target an 8x u32 or 4x u64 limb rep (the parts have
`v_add_co`/`v_addc` carry flags), not ref10's 10x25.5.

---

## 5. Field backend interface (recommended internal structure)

Inside a backend, keep the limb layer swappable so one challenger can ship both
representations and let the autotuner pick per arch:
```
// F_p over p = 2^255-19. One header per representation; same names.
fe_mul(fe &r, const fe &a, const fe &b);   // r = a*b mod p
fe_sqr(fe &r, const fe &a);
fe_add/fe_sub(...);
fe_mul_small(fe &r, const fe &a, uint32_t); // for 8B stepping constants
fe_inv(fe &r, const fe &a);                 // only the batch-inversion root
fe_frombytes/fe_tobytes(...);               // 32-byte LE compressed <-> fe
```
- **Backend A (integer radix-2^32):** 8 limbs; `fe_mul` = schoolbook 8x8 with
  `V_MAD_U64_U32` + carry chain, then reduce mod `2^255-19`. Portable CDNA2/4.
- **Backend B (FP64 radix-2^51):** 5 limbs; `fe_mul` via `V_FMA_F64`, carries
  folded in floating point (DPF-ECC/gECC style). Strong on MI210 (22.6 TF vector,
  45.3 TF matrix); weak on MI355X (FP64 regressed, no matrix bonus). An optional
  **FP64-matrix (rocWMMA/MFMA)** variant of this backend targets the 2x matrix
  rate that exists only on MI210/MI300X.

The fork is decided by measurement, not assertion (`CDNA-PERF-REFERENCE.md` s6).

**Evidence as of the 2026-10 literature sweep (`docs/research/literature.md`):**
- The only published head-to-head is **gECC** (arXiv 2501.03245, ACM TACO 2025):
  on NVIDIA A100, **integer radix-2^32 beats FP64** for 256-bit modmul (227 vs
  325 instructions). But that is NVIDIA-specific.
- On CDNA the hardware inverts it: MI300X has **full-rate FP64** (81.7 TF vector /
  163.4 TF matrix) while CDNA 32-bit integer multiply is *not* full-rate, and
  **ZKProphet** (arXiv 2509.22684) shows ECC field arithmetic bottlenecks on the
  32-bit integer pipeline with low ILP, and that conditional modular reductions
  cause warp divergence. So on CDNA the **FP64 radix-2^51 branch is the a-priori
  favorite**, the opposite of NVIDIA -- and keeping reductions **branchless** is a
  shared requirement for both backends.
- **Sharpened by the GPU MODE craft harvest (`docs/research/gpumode-craft.md`,
  vendor specs):** the "FP64 favorite" holds on MI210/MI300X but **not uniformly**.
  On MI355X, FP64 *regressed below MI300X* and matrix == vector (no MFMA bonus),
  so there integer radix-2^32 is the likely winner. The FP64-matrix 2x exists
  only on MI210/MI300X. Net: the test box (MI210) favors FP64/matrix; the scale
  box (MI355X) may favor integer -- so **ship two per-arch variants** and never
  assume MI210 tuning transfers. Microbench both on both cards.
- **Wildcard (novel, nobody has done it for ECC):** Berthomieu et al.
  (arXiv 2601.07508, Feb 2026) give error-bounded multiword FP64 that lifts the
  safe modulus to ~52 bits, and raise casting batched field-muls as **FP64 matrix
  products to hit the 163 TF MFMA units**. A genuine open research angle for a
  challenger. Integer side has a principled generator too: **MoMA** (Zhang &
  Franchetti, CGO 2025) lowers multi-word modular arithmetic correct-by-
  construction.

---

## 6. Correctness vectors (shared ground truth, `vectors/`)

Every backend runs `--verify` and must pass before any keys/s counts:

1. **Field vs reference bignum.** Random `a,b`: assert `a*b mod p`, `a^2`,
   `a^-1` match a known-good big-integer reference.
2. **Base-point multiples.** `k*B` for known `k` matches RFC 8032 / ref10.
3. **Known keypair.** A fixed secret -> fixed pubkey -> fixed `.onion` string
   (pulled from a reference lib / published vector).
4. **Encoding trap.** A fixed 32-byte LE pubkey -> its exact 56-char address,
   pinning the little-endian + x-sign-bit mapping.
5. **Split-key identity.** For `a0, k`: `(a0 + 8k)*B == A0 + k*(8B)`, and the
   recombined secret reproduces the matched address.
6. **Tor round-trip.** A found vanity key loads into Tor's expanded-secret
   format and a signature verifies (the key is actually usable).

---

## 7. Secret safety: split-key search (first-class, not optional)

So the search can run on Sol / national-compute without ever exposing a usable
secret:
1. **Locally:** generate secret offset `a0`, publish only `A0 = a0*B`. `a0` never
   leaves the local machine.
2. **On the shared GPU:** walk `A0 + k*(8B)`, matching prefixes. The GPU knows
   only `A0`, never `a0`. On a hit it returns the integer `k` (+ matched prefix).
3. **Locally:** real scalar = `a0 + 8k`; write the Tor `hs_ed25519_secret_key`
   in expanded form (`scalar || nonce`, nonce = 32 fresh CSPRNG bytes); verify by
   re-deriving `A_found` and comparing to the reported hit. Secret exists only
   here. Recovering `a0` from `A0` is the ECDLP, so the shared job (which sees
   only `A0` and emits only `k`) never holds secret material.

Reference implementation: **AlexanderYastrebov/onion-vanity-address** (`--offset`,
BSD-3, the only end-to-end split-key impl); **honion** documents the identical
additive scheme but left it unimplemented. (Correction to an earlier note:
**mkp224o does NOT** implement split-key/offset -- its batch mode is passphrase-
based deterministic derivation, which is not the same guarantee.) Verified
details, from source (`docs/research/tools.md`), that vector #5 must guard:
- **Signed k:** dual-law tiling produces hits *below* the start too, so `k` may be
  negative. Support it.
- **Clamp invariant:** `a0 + 8k` must stay in `[2^254, 2^255)` to remain a valid
  clamped scalar; assert it and error rather than emitting a key (fails only with
  prob ~2^-219 for bounded `k`, but check).
- **Correct modulus:** do the recovery as a big-integer / mod-L scalar add
  (mod-L is the theoretically correct modulus; the mod-p shortcut only coincides
  while offsets are tiny and non-wrapping). Keep offsets well under 2^250.
- x25519 client-authorization keys are a separate curve path; do not mix them
  with the ed25519 service-key path.

---

## 8. Instrumentation (per CLAUDE.md: stream, never batch)

- **The rig measures; entries never report speed.** Throughput is verified
  hits / (match probability * rig-timed seconds), as in `CONTRACT.md`. Every
  run, calibration, and summary is **appended as one JSON line and fsynced
  immediately** (`bench/<mode>-<time>.jsonl`): a session that dies at 90% keeps
  90% of its data, and progress can be followed with `tail -f`. Each run line
  records the plan, verified hits, rate and its counting error, the self-report
  z-score, thermal samples, flags, and the entry's log path.
- **Benchmarks resemble real searches**: at most 4 targets and about 2,000
  hits/s. More targets or hits would measure per-candidate matching and per-hit
  cold paths that a real 10-character search never exercises. The first
  benchmark of the baseline, at 22 targets, under-measured it by 30%
  (CDNA-PERF-REFERENCE sec 8).
- Real searches append each verified hit to `search/search.jsonl` the instant
  it arrives (point id, k, address, GPU) and keep running for multi-prefix.
- Autotune (grid first, Bayesian later) writes every trial as a JSONL line, so
  the sweep is itself the research dataset and is resumable.
- **Leaderboard metric:** the rank is the run-weighted mean rate of counted runs,
  taken from each benchmark's own summary record. The board never re-derives
  anything from raw runs, and one failed benchmark disqualifies that entry and
  version on that GPU. The headline *display* is "expected time to a 13-char
  prefix at 50% odds" = `0.693 * 2^65 / rate`. It is monotonic in the rate, so it
  is honest, and it turns the impossible 13-char into the scoreboard unit (a
  kernel that moves it from 117 years to 19 years has a vivid trophy). A
  submission that reports an actual 12/13-char *hit* is a correctness failure (a
  bug or a fabrication), not a win, and the rig treats it as one.
- **Honest measurement (adapted from mlx.fast / GPU MODE eval):**
  - **Paired ratio:** run baseline then candidate back-to-back in the same
    session on the same card and rank by the *ratio*, never absolute numbers
    across time (cancels thermal/host drift).
  - **Isolated runner:** one job at a time on a dedicated/ephemeral runner; pin
    the exact card, ROCm/driver version, and clocks.
  - **Thermal gate:** wait for the GPU to cool to its idle baseline before each
    run and sample power, temperature, and clocks throughout. A run with genuine
    thermal or hardware throttling is retried once, and the replaced attempt is
    logged as not counted. Power capping is normal under load and is only
    recorded. Runs without telemetry or still throttled are flagged, and are
    excluded only with `--strict-thermal`.
  - **Parent-owned timing window:** the trusted parent's wall-clock is authority;
    ignore any timing self-reported from inside the sandboxed kernel.
  - **Adaptive repeats:** early-stop when relative std error < 0.1% (min 3, cap
    ~100 runs or 120 s); report the MEAN, not the min (min rewards lucky noise);
    clear the L2 before each timed run.
  - **Calibration band + gain cap:** measure baseline-vs-baseline to get the noise
    CV, reject ranked runs outside the band, and cap per-submission gain (~5%) so
    a freak jump must reproduce as increments.

---

## 9. Open decisions / next steps

Resolved: this pass = layout + perf reference (done: these two docs). Start =
baseline + challenger. Contest = **separate repos unified by a conformance +
benchmark rig and a shared megaprompt** (sec 4), chosen to preserve pipeline-
restructuring freedom while keeping measurement honest.

Decided while building (2026-10-16):
1. **Languages.** The rig is Python, since it orchestrates and never touches the
   hot loop. Entries choose their own language; the baseline is CUDA + HIP from
   one source.
2. **onion-gpu license** is MIT and is vendored with its notice (`entries/baseline`).
3. **Reference ed25519.** Our Python reference is checked against RFC 8032,
   libsodium, and `cryptography`, and the rig verifies hits with libsodium itself,
   so it never trusts code that an entry might share.

Next, in order:
1. First MI210 run: `sbatch tools/sol/mi210-bench.sbatch` (microbench, baseline
   conformance, baseline benchmark). Its microbench column settles sec 5's fork.
2. Hand the corpus and `MEGAPROMPT.md` to the four competitors; round 1 on MI210.

---

## 10. Prior art, corpus & what is actually novel

Full synthesized notes live in `docs/research/` (`literature.md`,
`tools.md`, `competitions.md`), cached from the 2026-10 sweep.

Load-bearing references:
- **gECC** (arXiv 2501.03245) -- GPU ECC framework; integer-vs-FP64 head-to-head
  (NVIDIA); adopt its affine + Montgomery + Gather-Apply-Scatter batching.
- **DPF-ECC** (IPDPS 2020 / TIFS 2021) -- the FP64-radix ECC precedent.
- **ZKProphet** (arXiv 2509.22684) -- why field mul dominates and the integer
  pipeline / branchy-reduction bottleneck; argues for FP64 + branchless.
- **Berthomieu et al.** (arXiv 2601.07508, 2026) -- error-bounded multiword FP64;
  the FP64-matrix-core wildcard.
- **MoMA** (CGO 2025, arXiv 2501.07535) -- correct-by-construction multi-word
  modular arithmetic code-gen (integer side).
- **Bernstein et al.** (EUROCRYPT 2026) -- fast verified inversion mod 2^255-19,
  for the batch-inversion root.
- **honion docs** -- clearest public dual-addition-law write-up (the clean-room
  spec); **NIKIB0Y/onion-gpu** -- the HIPIFY baseline; **AlexanderYastrebov** --
  the split-key reference impl.
- **GPU MODE AMD competition** (MI300/MI355X, >30k submissions, beat AMD AITER) --
  the most transferable harness methodology *and* CDNA kernel-craft library; its
  writeups are the top corpus-harvest target still outstanding.

What is genuinely novel here (the research pitch, all honest):
- **No published ECC/bignum throughput exists for any MI210/250X/300X/355X.** The
  only prior GPU-Curve25519-on-AMD datapoint is 2014 (R9 290X, latency). We
  generate the first.
- **On CDNA the integer-vs-FP64 answer may invert NVIDIA's** -- an unanswered,
  measurable question (sec 5).
- **AI/LLM kernel optimization has never been applied to multi-precision-integer /
  crypto kernels** -- every tool (CUDA-L1, KernelBlaster, Kernel-Smith, ...)
  targets ML operators. Open territory, and Kernel-Smith already runs a non-NVIDIA
  (MACA) backend, so an AMD target is feasible.

### Corpus licensing policy (what we may hand competitors)

The GPU MODE harvest found **no cleanly-licensed winning AMD kernel to ship**:
the grand-prize source is not public, the available ones (Snektron,
luongthecong123) carry **no license = all-rights-reserved**, and the canonical
submission dataset (GPUMODE/kernelbot-data) is under a reciprocity/anti-lockout
license hostile to a closed-model contest. Therefore:
- **Shareable with competitors / publishable:** our own writing and permissively-
  licensed refs -- `docs/research/*.md` syntheses (our words; ideas are not
  copyrightable), the HIPIFY'd onion-gpu baseline (MIT), honion docs (MIT/Apache),
  and a **clean-room MIT reference kernel we write ourselves** from the public
  *writeups*.
- **Internal reference only, never redistributed or published:**
  `docs/research/gpumode-kernel-sources.md` (contains all-rights-reserved source
  excerpts -- fair-use reading-to-learn, not redistribution). Keep it out of any
  shared/published corpus and out of the megaprompt.
