# Reference-tool recon: Tor v3 vanity .onion generators

Research for the clean-room HIP (AMD CDNA) vanity generator. All data below is
from the repos' READMEs, in-tree docs, and the actual source files (fetched via
`raw.githubusercontent.com` / `gh api` on 2026-10-15). Where a number is a
self-report that we cannot independently confirm, it is flagged UNVERIFIED.

The six tools split cleanly into two algebraic generations:

- **Classic / mkp224o-generation** (single moving point in extended coords,
  stepped by 8B, batched Z-inversion): **mkp224o**, **NIKIB0Y/onion-gpu**.
  ~12 field muls/candidate.
- **Dual-addition-law generation** (affine y-only, ± symmetry, a table of fixed
  offsets, one batched vector-division): **honion**, **prefix32**,
  **AlexanderYastrebov**, and **onion-gen** (same 8x32 field + offset model).
  ~5–5.5 field muls/candidate.

The whole field splits GF(2^255-19) the same two ways everyone always has:
redundant **radix-2^25.5, 10 limbs** (ref10 int32/int64) vs **saturated** either
**8 x u32 (radix-2^32)** or **4 x u64 (radix-2^64)** with `2^256 = 38 (mod p)`
folding. CDNA has a hardware carry flag (`v_add_co_u32`/`v_addc_co_u32`), so the
saturated 8xu32 or 4xu64 reps are the ones worth targeting; the 10x25.5 rep only
wins on hardware without a usable carry (which is exactly why honion's own perf
doc lists moving off 10x25.5 as its biggest remaining win).

---

## 1. NIKIB0Y/onion-gpu  — the clean CUDA C++ classic kernel (BASELINE PICK)

- **Repo health:** created 2026-09-27, pushed 2026-09-27, **1 star**, 0 forks,
  **0 open issues**, not archived. Brand new, essentially unknown, single author.
  6 commits. (Low adoption, but the code is self-contained and auditable.)
- **License: MIT** (`LICENSE`; README: "MIT — see LICENSE"). Port + publish OK.
- **Language / backend:** plain **CUDA C++** (`.cu` + `.cuh`), compiled with
  `nvcc` via a Makefile. **No NVRTC, no Rust, no OpenCL.** NVIDIA-only + Linux as
  shipped. This is the single most HIPIFY-friendly codebase of the six.
- **Files:** `main.cu` (947 lines), `ge25519.cuh` (175), `fe25519.cuh` (264),
  `sha512.cuh`, `sha3_256.cuh`, `verify.py`, `Makefile`, `LICENSE`, `SECURITY.md`.

### Crypto internals (read from source)
- **Field:** `typedef int32_t fe[10]` — **radix-2^25.5, 10 limbs** (limbs
  0,2,4,6,8 = 26-bit, 1,3,5,7,9 = 25-bit), 64-bit accumulation. Classic ref10.
- **Curve / coords:** twisted Edwards a=-1 over 2^255-19. Points in extended
  `ge_p3 {X,Y,Z,T}`, `ge_p2`, `ge_p1p1`. `2d` constant hardcoded as 10 limbs.
- **Stepping:** per thread one full `ge_scalarmult_base` (MSB-first
  double-and-add) to a random start, then walks in steps of **8·B** (so it keeps
  landing on clamped — multiple-of-8 — scalars; same trick as mkp224o).
- **Point add:** `ge_madd` = **mixed addition with a cached affine point**
  (Z_q=1), the `madd-2008-hwcd-3` formula. Code + comments make the cost explicit:
  **3 muls in `ge_madd` + 4 in `ge_p1p1_to_p3` = 7 muls to advance the point**,
  which the comment calls "the known optimum for mixed addition on a=-1" (true
  for a *single* moving point — the dual law beats it by amortizing across ±).
- **Batch inversion:** Montgomery's trick fused into forward+backward passes,
  one field inversion per group of **`BINV_N = 384`** (`INNER_BATCH = 44*BINV_N
  = 16896` keys/thread/launch). Amortized inversion ≈ 1 mul/key at N≈256.
- **Muls/candidate (from `main.cu` comment + README):** **≈12 field muls/key**
  (README rounds to ~11.7): 7 to step + Montgomery (1 fwd + 3 back) + ~1
  amortized inversion + packing. **This is ~2.1x the dual-law tools' ~5.5** — it
  is deliberately the previous-generation algebra, which is exactly what makes it
  a good honest baseline to beat.
- **Encoding:** v3 address = base32(pubkey ‖ SHA3-256(".onion checksum"‖pub‖0x03)[:2]
  ‖ 0x03). Prefix test = one masked 64-bit compare for ≤12 chars.
- **NOT y-only:** it computes full compressed pubkeys (`ge_p3_tobytes` does the
  x-sign bit). A clean-room kernel can drop x entirely (honion/AlexanderYastrebov do).

### Bench / verify (strong — gives us a ready calibration harness)
- `--bench`: one kernel launch, reports throughput **and hit count vs expected**.
- **Startup self-tests (always):** (1) scalar=1 → base point; (2) RFC 8032 test
  vector 1; (3) **differential test** — the incremental 8B-chain + batch
  inversion is checked byte-for-byte against independent full double-and-add
  (`chaintest_verify_kernel` recomputes scalar+8i the slow way).
- Every emitted candidate is re-derived from its scalar and cross-checked on the
  host before writing; host refuses to write on mismatch.
- `verify.py`: pure-Python, CUDA-independent re-derivation (scalar→address,
  clamp check, secret-file perms check).
- **Benchmark self-report (UNVERIFIED):** 917 M keys/s sustained on RTX 5060
  (~940 M/s single launch); "~16x mkp224o @ 56 M/s on Ryzen 7 5700X"; wall-clock
  60x 5-char in 2.18 s vs mkp224o 36.0 s.

### Split-key / offset
- **None.** Walks 8B from a per-thread random start; full keypair materialized on
  host. No public-offset mode. (We add split-key ourselves — see §7.)

---

## 2. Splinters-io/honion  — the best SOTA algebra reference (MINE THE DOCS)

- **Repo health:** created 2026-08-20, pushed 2026-09-17, **2 stars**, 1 fork,
  **0 open issues**. 1 commit (squashed import). New, single author.
- **License: MIT OR Apache-2.0** (dual; `LICENSE-MIT`, `LICENSE-APACHE`). Most
  permissive of the set — pick either. Port + publish OK.
- **Language / backend:** **Rust workspace** (1.85+) host; **CUDA device code
  compiled at runtime via NVRTC** (`libnvrtc`, needs only driver). NVIDIA-only
  (cc >= 5.0). Crates: `honion-core`, `honion-gpu`, `honion-keyfile`,
  `honion-cli`. Device code in `cuda/` (`fe25519.cuh`, `fe25519_u32.cuh`,
  `ge25519.cuh`, `search.cu`, `testkernels.cu`, `verify_dual_law.py`).
- **Docs/ are the crown jewel** — `01-ed25519-vanity-search.md`,
  `02-gpu-architecture.md`, `05-security-model.md`, `06-performance.md`,
  `07-benchmarks.md` are the clearest public write-up of the SOTA method anywhere.
  Our clean-room kernel should be specced from these.

### Crypto internals (from docs + tree)
- **Field:** README says current kernel uses **eight 32-bit limbs with hardware
  carry chains** (`fe25519_u32.cuh`, generated by `gen_fe32.py`). An older
  **ten 25.5-bit / 64-bit-accumulator** rep (`fe25519.cuh`) is also in-tree; perf
  doc (07) documents the 10x25.5 → 8x32 move as the ~1.4x win that closes the gap
  to prefix32, so treat 8x32 as the live rep. (Minor inconsistency between docs
  07 "listed as identified" and README "uses"; the u32 file exists, so it landed.)
- **Stepping:** "consecutive valid scalars differ by 8" (clamping clears low 3
  bits) → one point addition per candidate instead of a scalar mult.
- **Dual addition law (the key trick), verbatim from doc 01:**
  `y(P+Q) = (x1·y1 − x2·y2)/(x1·y2 − y1·x2)` and
  `y(P−Q) = (x1·y1 + x2·y2)/(x1·y2 + y1·x2)`.
  Both share `x1·y2` and `y1·x2`; `x1·y1` is once-per-batch (moving point),
  `x2·y2` is precomputed (fixed offset). So **2 muls produce 2 candidates (P+Q and
  P−Q)**. No curve constant d, no projective coords for the result, **y-only**
  (x-sign bit is base32 char 51, past any searchable prefix, so x is never
  computed — saves a mul and a third of per-candidate local memory).
- **Kernel shape:** table of fixed offsets `8B, 2·8B, …, HALF·8B` (affine, with
  x·y) in `__shared__`; each batch emits `2·HALF` candidates `P ± j·8B` plus `P`,
  then advances P by the giant step `(2·HALF+1)·8B`. Batches tile the integers
  exactly; **a match can lie below a thread's start** (signed offset).
- **Batch inversion:** Montgomery's trick turns **513 divisions into one
  inversion** (HALF=256 → 2·256+1). Numerators folded into the *forward* pass, so
  the implied division is free and the backward pass recovers y in 2 muls not 3.
  The next base point's Z is folded into the same product → exactly **one**
  inversion per batch.
- **Muls/candidate (doc 01 table): ≈5.5** (1 shared + 2 fwd fold + 2 back + ~0.5
  amortized inversion). Doc 07 shows the rewrite from the 12-mul extended-coord
  walk to the 5-mul dual law took the kernel 4.9→8.4 G/s (1.7x, matching 12/5).

### Bench / verify
- Commands: `honion estimate --prefix …`, `honion search --prefix … --out … --count N`,
  `honion verify <dir>`. `crates/honion-gpu/examples/bench.rs` is the harness;
  `bench/` holds `study.sh` + raw per-run CSV.
- **Verify is first-class:** every GPU hit is a *claim*, re-derived on CPU with
  `curve25519-dalek` (clamp check incl. overflow invariant, pubkey match, address
  round-trip recomputing SHA3-256, and a signature verify) before any file is
  written; atomic temp-file+rename, 0700/0600, never overwrites.
- `cuda/verify_dual_law.py` checks the dual law vs the standard addition law over
  random pairs (run before any CUDA was written — good practice to copy).
- **Benchmark (self-report, UNVERIFIED but unusually rigorous method):** end-to-end
  **1.2514e10 addr/s** on RTX PRO 6000 Blackwell (96 GB, sm_120), 100 interleaved
  90 s rounds, 95% CI ±0.64%. Kernel-only 8.4e9. Method: `throughput = hits·2^D/T`
  (counts key *dirs actually written*, so no tool's internal counter is trusted —
  **this is the methodology we should adopt**). Same-card comparison:
  honion 1.25e10 (48.5x mkp224o) > prefix32 1.05e10 (40.6x) >> mkp224o 2.58e8
  (CPU, 48 threads, amd64-64-24k). honion also most energy-efficient (2.03e7 addr/W).

### Split-key / offset (DOCUMENTED, NOT IMPLEMENTED)
- The **kernel already keeps all secret material off the GPU**: host draws
  clamped `a0` per thread, computes `A0 = a0·B` on CPU, uploads only `A0`; device
  walks `A0 + k·8B`, returns `(thread, k)`; host reconstructs secret `a0 + 8k`.
  (`honion-gpu` deliberately does not depend on `honion-keyfile`, so no code path
  can push a secret to the device. VRAM never holds a secret → can't/needn't wipe.)
- **True split-key (service / someone else's key)** is spelled out in doc 05 but
  explicitly **"not implemented"** (deliberately deferred): customer sends only
  `P`; service searches `s` with `P + s·B` matching; service returns `s`; customer
  computes final secret `a + s` locally. Identical kernel, just started from `P`.
  This is the scheme we want for shared HPC (see §7) — honion documents it, we
  implement it.

---

## 3. 0xROOTPLS/prefix32  — SOTA algebra, already runs on AMD (OpenCL), MIT

- **Repo health:** created 2026-03-30, pushed 2026-08-07, **4 stars**, 0 forks,
  **0 open issues**. 32 commits, v2.2.0. Oldest of the new GPU tools; the most
  "real" of the dual-law GPU set.
- **License: MIT** ("Copyright (c) 2026 0xROOTPLS … just keep the copyright
  notice"). Port + publish OK.
- **Language / backend:** **Rust host + OpenCL kernel** (`kernel/vanity.cl`,
  auto-tuned at runtime). CUDA appears only as a source of `OpenCL.lib` on
  Windows. **Already benchmarked on AMD** (RX 6800 XT). OpenCL on CDNA runs under
  ROCm's OpenCL runtime with no port — so prefix32 is our **zero-port on-device
  AMD baseline** (complementary to the HIPIFY'd NIKIB0Y baseline).
- **Files:** `src/{main,search,gpu,fe25519,ge25519}.rs`, `kernel/vanity.cl`,
  `build.rs`, `LICENSE`, `README.md`.

### Crypto internals (from source)
- **Field:** `Fe25519(pub [u64; 4])` — **radix-2^64, 4 x 64-bit limbs**, `2^256 =
  38 (mod p)` folding, explicit `overflowing_add`/`adc`/`sbb` carry chains. (This
  is the rep honion's perf doc compares itself against; the CPU/host rep. The
  `.cl` kernel carries the device rep — not fully inspected here.)
- **Curve:** `GeP3`, `Affine { x, y, xy }` (precomputed x·y), `from_p3_zinv`
  (shared Z-inverse), `mult_by_cofactor` (8B). This is the **dual-law affine-y**
  structure (same as honion / AlexanderYastrebov).
- **Batch inversion:** `--batch` = "candidates per inversion", **default 2048**.
- **Muls/candidate:** honion's doc 07 (independent reading of prefix32) puts it at
  **~5** via the dual law — i.e., same generation as honion, and on the same card
  prefix32's *kernel* was actually faster than honion's until honion switched to
  8x32 limbs. So prefix32 is the strongest existing dual-law kernel on AMD-capable
  OpenCL. (Field arithmetic is unit-tested against `curve25519-dalek`.)

### Bench / verify
- **Bench: strong.** `--bench [SECS]` (median/mean/spread on the real search),
  `--bench-mul` (field-multiply micro-bench), `--gpu --bench`.
- **Verify: none.** Does NOT re-derive keys before writing (honion confirms), and
  **inherits umask — observed 0755/0644 key-file perms** (a real secret-hygiene
  bug we must not copy). Writes non-atomically.
- **Benchmark self-report (UNVERIFIED):** v2 ~**3135 M/s RX 6800 XT**, ~2430 M/s
  RTX 5060 Ti; CPU 222.9 M/s (5900X, 24 thr). (honion measured it at 1.05e10 on a
  PRO 6000 — much faster card.)

### Split-key / offset
- **None, and NOT secret-safe:** honion's table lists prefix32 as holding **secret
  material in device memory: yes**. It generates full keypairs through the GPU. No
  public-offset mode. Would need rework before any shared-HPC use.

---

## 4. mihalych-pro/onion-gen  — multi-backend (incl. AMD via OpenCL/Vulkan), Rust-CUDA

- **Repo health:** created 2026-10-08, pushed 2026-10-09, **1 star**, 0 forks,
  **0 open issues**. 20 commits. Newest repo; most feature-rich; single author;
  bilingual (en/ru) docs.
- **License: Apache-2.0.** Port + publish OK (adds an explicit patent grant).
- **Language / backend:** **Rust**, one binary reaching **CUDA, Metal, Vulkan,
  OpenCL**. The CUDA kernel is **written in Rust** (`cuda-kernel/src/{field,curve,
  field_asm}.rs`, rust-gpu/rust-CUDA → PTX) — **there is no `.cu` to HIPIFY.** The
  OpenCL and Vulkan backends already run on AMD. Vendors proven in benches: NVIDIA
  (RTX 4060) and Apple (M1 Pro).
- **Files of interest:** `cuda-kernel/src/field.rs`, `curve.rs`, `field_asm.rs`;
  `docs/implementation/{cuda-path,opencl,portable-gpu,search-engine}.md`;
  `docs/design/distributed-search.md`; many `benches/*.rs`.

### Crypto internals (from source)
- **Field (verbatim from `field.rs`):** "radix 2^32 in eight 32-bit limbs … held
  below 2^256 and folded with 2^256 = 38 (mod p). … the same representation the
  processor path and the other two kernels use." → **8 x u32 / radix-2^32**, same
  as honion's u32 path. Doc explicitly notes 64 partial products vs 100 for
  10x25.5, element 32 B vs 40 B. (Good portable choice; validates 8xu32 for us.)
- **Curve / muls:** `curve.rs` present; README/docs do not state a muls/candidate
  number. Uses the same 8B-offset stepping (`expanded_secret_at_offset`), so it is
  the dual-law/offset generation but the exact per-candidate count is UNVERIFIED.
- **Batch inversion:** present (search engine works in "blocks"); size not pinned
  in the material read.

### Bench / verify
- `verify <dir>` (format, clamping, key↔address consistency, via Arti's
  `tor-hscrypto` — independent of its own arithmetic). `guess` (estimate without
  searching). `--evaluate` (measure machine, cost per prefix length 4–12).
  `--retune` (re-measure batch size / launch shape).
- **Benchmark self-report (UNVERIFIED), median of 3:** CUDA RTX 4060 **1861.9
  M/s**, Vulkan 1791.0, OpenCL 1758.9; Metal M1 Pro 255.5, OpenCL M1 Pro 249.8;
  CPU i9-9900K 150.4. "12.6x onionloom on the 4060."

### Distributed master/worker (relevant to HPC, but NOT split-key)
- Master hands each worker **a seed + a range of blocks + the filter set**;
  worker reports **where** (an offset/location) it found a hit, **not the key**;
  master re-derives the key via the same `expanded_secret_at_offset`.
- **IMPORTANT — not secret-safe:** the master *hands the seed to the worker*
  ("The master handed out the seed, so it already has it"), so **a worker can
  derive the keys**. This is a work-distribution optimization (don't ship the
  32-byte key back), **not** a public-offset split-key. On shared HPC where the
  workers are untrusted, this model leaks. Use the additive split-key (§7) instead.

### Split-key / offset
- No public-offset / secret-safe mode. `--seed` (repeatable) and `--from-block`
  (resume) only.

---

## 5. cathugger/mkp224o  — the CPU reference everyone benchmarks against (CC0)

- **Repo health:** created 2017, pushed **2024-02-15** (commit `5172c0f`),
  **1633 stars**, 178 forks, **31 open issues**. The established, battle-tested
  baseline; by far the most-used of the six. Not archived but low recent activity.
- **License: CC0-1.0** (public-domain dedication, in `COPYING.txt` — there is no
  `LICENSE` file). The most permissive possible: port + publish with **no
  attribution required**. (Bundled ed25519 backends are SUPERCOP/ref10, also
  public domain / unencumbered.)
- **Language / backend:** **C (C99), CPU only.** No GPU. ed25519 backends:
  `ref10` (10x25.5), `amd64-51-30k` (5x51 asm), `amd64-64-24k` (4x64 asm),
  `ed25519-donna`. `--enable-amd64-51-30k` etc. at configure time.

### Crypto internals (this is the origin of the whole technique)
- Generates v3 (ed25519) addresses. `worker_fast()` (idea from horse25519) and
  `worker_batch()` (passphrase/deterministic + the batched path; details in
  `OPTIMISATION.txt`). The **8·B stepping + `fe25519_batchinvert` (Montgomery's
  trick)** originate here — `ed25519/*/fe25519_batchinvert.c`,
  `ge25519_batchpack.c`. This is the *classic* single-moving-point algebra
  (the generation NIKIB0Y reimplements on GPU), **not** the dual law.
- Batch size configurable at build (`BATCHNUM`/`BATCHMUL` in `OPTIMISATION.txt`).
- **No muls/candidate figure published**; it is the ~12-mul-class single-point
  walk. `-s` prints stats for benchmarking the ed25519 backends.

### Bench / verify
- `-s` stats (keys/s per backend). No dedicated `--bench`/`--verify` subcommands.
- **Benchmark (3rd-party, from honion's same-card run, closest to verified):**
  2.58e8 addr/s on a Threadripper 9960X, 48 threads, amd64-64-24k backend (21%
  ahead of amd64-51-30k). README itself gives only rough "tens of minutes for 6
  chars" guidance.

### Split-key / offset
- **NONE.** `grep -rniE 'offset|split.?key'` over the whole tree returns nothing.
  **The task brief's claim that mkp224o "has split-key/--offset-style secret-safe
  generation" is INCORRECT for mainline cathugger/mkp224o.** Its `worker_batch`
  passphrase path is *deterministic key derivation from a passphrase*, a different
  thing (and arguably worse for secrecy). If a split-key mkp224o exists it is a
  third-party fork, not this repo. **The real split-key reference is
  AlexanderYastrebov (§6), which implements it end-to-end.**

---

## 6. AlexanderYastrebov/onion-vanity-address  — the split-key reference impl (Go, BSD)

- **Repo health:** created 2025-09-11, pushed 2025-09-30, **84 stars**, 4 forks,
  **3 open issues** (1 PR). Most-adopted of the modern tools; clean, tested Go.
- **License: BSD-3-Clause** (`LICENSE`). Port + publish OK (keep copyright +
  no-endorsement clause). Depends on `filippo.io/edwards25519` (BSD-3) and the
  author's `github.com/AlexanderYastrebov/vanity25519` field pkg.
- **Language / backend:** **Go, CPU** (CLI; also Docker / k8s Jobs for distribution).
- **Files:** `main.go` (CLI + the canonical `--offset` examples), `search.go`
  (the dual-law + offset math), `prefix.go`, `service.go` (Tor key-file I/O),
  `client.go`.

### Crypto internals (read from `search.go`)
- Uses `filippo.io/edwards25519` (`Point`) + a custom `field.Element`. `_B8 =
  8·B` via `MultByCofactor(NewGeneratorPoint())`.
- **Dual addition law, affine, y-only** — identical algebra to honion/prefix32.
  Verbatim comment + code:
  `y3  = (x1·y1 − x2·y2)/(x1·y2 − y1·x2)` (P+Q),
  `y3' = (x1·y1 + x2·y2)/(x1·y2 + y1·x2)` (P−Q, via x2→−x2 symmetry).
  Each `Affine{X,Y,XY}` caches x·y. Offsets table `offsets[i] = (i+1)·8B`.
- **Batch:** `batchSize = 4096` (so 2048 offsets → 4097 candidates incl. center),
  one iteration tests `{P±j·8B} ∪ {P}`.
- **Vector division** (`vectorDivision`, Harris' extension of Montgomery's trick,
  eprint 2008/199): `4(n−1)+1` muls + 1 invert (~265 muls) → **262M + 4M·n**.
  The moving point's own `1/Z` is piggybacked onto the same vector division.
- **Muls/candidate (doc'd + matches code): amortized (5M + 2A) per candidate.**
  Inner loop comment: `(5M+2A)·batchSize + 278M + 9A` per iteration.

### Bench / verify
- Prints attempts + attempts/s on completion (no separate `--bench`). No
  `--verify` subcommand, but every path re-derives the pubkey (`publicKeyFor`
  with `SetBytesWithClamping`) so output is self-consistent. `main_test.go`,
  `search_test.go`, `prefix_test.go` give real unit tests.
- **Benchmark (self-report, UNVERIFIED):** ~45–48.5 M/s on a laptop; ~2x mkp224o;
  k8s fan-out ~7.2e8 attempts/s aggregate.

### Split-key / offset (FULLY IMPLEMENTED — the reference for §7)
- `--from PUBLIC_KEY`: search starting from a given pubkey, output an **`offset`**
  (base64), not a key. Secret never present.
- `--offset OFFSET`: read start secret from stdin, output the final keypair.
- Exact recovery (`add` in `search.go`): `final_scalar = start_scalar + 8·offset`.
  See §7 for the full derivation and the safety conditions.

---

## 7. The split-key / public-offset math  (for secret safety on shared HPC)

Confirmed from **two independent implementations/specs**: AlexanderYastrebov
`search.go::add` (implemented) and honion `docs/05-security-model.md` (spec'd).

### Setup
- Ed25519 base point `B`, group order `L`. Clamping forces every valid secret
  scalar `a` to be a **multiple of 8** with `a ∈ [2^254, 2^255)`.
- Define the cofactor-cleared base point **`B8 = 8·B`** (`MultByCofactor(B)`).
  Walking in units of `B8` keeps every scalar a valid (multiple-of-8) clamp.

### Protocol (what we run on shared/untrusted HPC)
1. **Locally** (trusted machine) draw a clamped start scalar `a0` and compute the
   start public key `A0 = a0·B`. Keep `a0` secret; it never leaves the box.
2. **Ship only `A0`** (a public key) to the GPU / HPC job.
3. HPC walks candidates `A0 + k·B8 = (a0 + 8k)·B` (k signed, dual-law tiling),
   tests the address prefix, and on a hit **returns only the integer `k`
   (the "offset")** — never a key. (honion returns `(thread, k)`; AlexanderYastrebov
   returns `offset`.)
4. **Locally**, recover the final secret scalar:

   **`a_found = a0 + 8·k`**

   and the final public key `A_found = A0 + k·B8 = a_found·B`. Build the Tor
   `hs_ed25519_secret_key` from `a_found` (expanded form: `a_found ‖ nonce`,
   nonce = 32 fresh CSPRNG bytes or `SHA-512(a_found)[32:]`; clamp is a no-op by
   construction — see below).

### Why it is secret-safe
The HPC job sees only `A0` (public) and emits only `k` (a public integer). Both
`A0` and `k` reveal nothing about `a0` (recovering `a0` from `A0` is the ECDLP).
An observer can compute `A_found` but not `a_found`. VRAM never holds secret
material, so it needn't be wiped and can be profiled/dumped freely.

### Exact arithmetic and the safety invariant (important for our impl)
- AlexanderYastrebov computes `a_found = a0 + 8·offset` using **field.Element add
  (mod p = 2^255−19)**, with `offset` a non-negative big.int and start scalars
  seeded from a random `uint64` (`startOffset < 2^64`). This is only correct
  because the values never wrap: `8·offset ≈ 2^67 ≪ 2^254`, so `a0 + 8·offset`
  stays in `[2^254, 2^255) ⊂ [0, p)` and mod-p reduction is a no-op, i.e. it
  equals the true **integer** sum, which is already a validly-clamped scalar
  (low 3 bits stay 0; bit 254 stays 1; bit 255 stays 0; re-clamping is a no-op).
- **Invariant to enforce (honion `SecretScalar::offset` checks it, returns an
  error rather than a key):** `a0 + 8k` must stay `< 2^255` (and `≥ 2^254`) so it
  remains a valid clamp. With `|k|` bounded by ~2^32–2^64 this fails with
  probability ~2^-219 — astronomically unlikely, but **check it, don't assume**.
- **Our recommendation for the HIP tool:** do the recovery as a **big-integer /
  mod-L scalar add** (not mod-p), and then assert the clamp invariant and verify
  by re-deriving `A_found` and comparing to the reported hit. mod-L is the
  theoretically correct modulus (the point depends on scalar mod L); mod-p merely
  happens to coincide in the non-wrapping regime. Keeping offsets bounded well
  under 2^250 makes all three (integer, mod-p, mod-L) identical, which is the
  safest regime to operate in.
- Signed `k`: dual-law tiling means a hit can be *below* the start, so support
  negative `k` (store/transmit as signed, recover `a0 + 8k` with signed `k`).

### Client-authorization keys (x25519) are different
AlexanderYastrebov's `--client` path uses `vanity25519.Add` on x25519 keys — a
separate curve/representation. Our tool targets service (Ed25519) keys; keep the
two code paths distinct.

---

## 8. Comparison table

| Tool | Lang / backend | Vendors | Field rep | Point add | Muls/cand | Batch-inv N | License | Bench | Verify | Split-key |
|---|---|---|---|---|---|---|---|---|---|---|
| **NIKIB0Y/onion-gpu** | CUDA C++ (nvcc) | NVIDIA | radix-2^25.5, 10-limb int32 (ref10) | mixed madd (extended, single pt), 7/step | **~12** | 384 (`BINV_N`) | **MIT** | `--bench` + self-tests + differential | **yes** (verify.py + per-hit re-derive) | no |
| **Splinters-io/honion** | Rust + CUDA/NVRTC | NVIDIA | 8x u32 (current) / 10x25.5 (legacy) | **dual law, affine y-only** | **~5.5** | 513 (HALF=256) | **MIT OR Apache-2.0** | `estimate` + `bench.rs` + `hits·2^D/T` method | **yes** (CPU re-derive, sig check, atomic 0600) | **doc'd, NOT impl** |
| **0xROOTPLS/prefix32** | Rust + OpenCL | **AMD** + NVIDIA | radix-2^64, 4x u64 (host) | **dual law, affine** | **~5** | 2048 (`--batch`) | **MIT** | `--bench`, `--bench-mul` | no (umask 0644 keys!) | no (secret in VRAM) |
| **mihalych-pro/onion-gen** | Rust; CUDA(Rust-GPU)/Metal/Vulkan/OpenCL | **AMD**+NVIDIA+Apple | radix-2^32, 8x u32 | dual-law/offset (count n/s) | n/s | "blocks" | **Apache-2.0** | `--evaluate`, `guess`, `--retune` | `verify` (via Arti) | no (seed shared w/ workers) |
| **cathugger/mkp224o** | C, CPU only | CPU | ref10 / 5x51 / 4x64 / donna | single-pt + batchinvert (classic) | ~12-class | build-time `BATCHNUM` | **CC0-1.0** | `-s` stats | no subcmd | **no** (brief's claim is wrong) |
| **AlexanderYastrebov/onion-vanity-address** | Go, CPU | CPU | filippo edwards25519 + custom field | **dual law, affine y-only** | **5M+2A** | 4096 (`batchSize`) | **BSD-3-Clause** | attempts/s | self-consistent (re-derive) | **YES, fully impl** (`--from`/`--offset`) |

n/s = not stated in source/README read.

## 9. License verdict — NONE forbids porting/publishing
All six are permissive and safe to port and publish: CC0 (mkp224o), MIT
(NIKIB0Y, prefix32), MIT-OR-Apache-2.0 (honion), Apache-2.0 (onion-gen),
BSD-3-Clause (AlexanderYastrebov). **No GPL/AGPL/CC-BY-SA/"none" anywhere.** The
only obligations are trivial attribution (keep copyright notices for MIT/BSD;
Apache adds a patent grant + NOTICE propagation). We can HIPIFY any of them and
release under our own permissive license. Algorithms themselves are not
copyrightable, so the clean-room dual-law kernel carries no license burden from
honion/prefix32 even if we only read their docs.

## 10. Unverifiable / caveats
- **All throughput numbers are author self-reports** except mkp224o/prefix32 as
  measured inside honion's benchmark doc (which is itself a self-report, but uses
  the soundest method — counting key dirs written, `hits·2^D/T` — and is the one
  to replicate). Treat every G/s figure as UNVERIFIED until we reproduce on our
  own AMD hardware.
- **honion field rep:** README says 8x u32 is live; perf doc 07 frames it as
  "identified/costed". The `fe25519_u32.cuh` file exists in-tree, so 8x u32 is
  present; whether `search.cu` compiles the u32 or the 10x25.5 path by default was
  not confirmed from source (only the `.cuh` headers + docs were read).
- **prefix32 / onion-gen device field rep:** confirmed from host Rust
  (`fe25519.rs` 4x64) and `field.rs` (8x u32) respectively; the actual `.cl` /
  PTX device arithmetic was not line-read.
- **onion-gen muls/candidate** and batch-inversion size not stated; its distributed
  model shares the seed with workers (NOT secret-safe on untrusted HPC).
- **mkp224o split-key:** confirmed ABSENT by grep over the full tree.
- Repos NIKIB0Y, honion, onion-gen are days-to-weeks old with 1–2 stars and
  single authors — low external validation; read as untested-at-scale.
