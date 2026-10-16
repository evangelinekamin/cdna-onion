# GPU MODE / AMD Competition Kernel Craft — Transferable CDNA Systems Lessons

Harvested from the GPU MODE AMD $100K competition (MI300, 2025) and the $1.1M "E2E Model Speedrun"
(MI355X, 2026) winning-kernel writeups + authoritative ROCm/CDNA docs.

**READ THIS FIRST — framing.** Every GPU MODE AMD kernel is matmul-family (FP8 GEMM / MLA decode /
MXFP4 MoE). OUR workload is a **batched ed25519 / twisted-Edwards point-addition grinder** with Montgomery
batch inversion — field arithmetic over p = 2^255-19, no matrix structure. So this doc deliberately
**discards the GEMM algorithm/tiling math** and keeps only the **CDNA systems craft that transfers to an
instruction-bound integer/FP64 compute kernel bottlenecked by VGPR occupancy and branchy modular
reductions**. Every section is labeled with how it maps to our hot loop.

Verification legend: [V] read from primary/authoritative source; [K] established architectural fact
(stated from knowledge, cross-checked where possible); [?] could not verify — flagged.

Our targets (coordinator): **gfx90a = MI210 (CDNA2)** and **gfx950 = MI355X (CDNA4)**.

---

## 0. Hardware ground truth for our two targets

| | MI210 (gfx90a, CDNA2) | MI355X (gfx950, CDNA4) |
|---|---|---|
| Wavefront | **64** (wave64) [V] | **64** (wave64) [V] |
| CUs | 104 [V] | 256 (32/XCD × 8 XCD) [V] |
| SIMD/EUs per CU | 4, each ≤8 waves → **32 waves/CU** [V] | 4 (64 stream-proc/CU) [V] |
| LDS per CU | 64 KiB [K] | **160 KiB** [V] |
| VGPR file | 256 arch VGPRs/lane (gfx90a) [V] | 512 KiB VGPR file/CU [V] |
| Matrix cores/CU | 4 | 4 [V] |

- **Wave size is 64 on BOTH targets.** ROCm spec tables list only `64` for CDNA4/gfx950; wave32 is an
  RDNA-only feature. The coordinator's "CDNA5 wave32 shift" is **NOT supported by any current ROCm spec
  I can find** — CDNA1-4 are wave64-only. Treat any MI400/"CDNA-next" wave32 as **unconfirmed/future [?]**.
  Design for wave64: our batch grinder should map one scalar-field element (or one curve point) per lane
  and run 64-wide.

### 0.1 The FP64 vector-vs-matrix wildcard — VERIFIED, and it flips between our two targets [V]
Peak FP64 (TFLOPS), vector (VALU/FMA) vs matrix (MFMA):

| GPU | FP64 vector | FP64 matrix (MFMA) | matrix advantage? |
|---|---|---|---|
| MI210 (CDNA2) | 22.6 | 45.3 | **2x — yes** |
| MI250X (CDNA2) | 47.9 | 95.7 | **2x — yes** |
| MI300X (CDNA3) | 81.7 | 163.4 | **2x — yes** |
| **MI355X (CDNA4)** | **78.6** | **78.6** | **NONE — equal**, and *lower* than MI300X |

**Verdict for our FP64-matrix branch:** the FP64 MFMA path (`v_mfma_f64_16x16x4` / `4x4x4`) doubles FP64
throughput on **MI210 and MI300** — worth a serious prototype there if we can cast field-mul limb products
as tiny f64 matrix ops. On **MI355X it buys nothing** (matrix == vector, and absolute FP64 regressed), so
on gfx950 stay on the VALU FP64 (or integer) path. This asymmetry means the fastest kernel is likely
**per-arch different** — plan to ship two tuned variants. Sources: glennklockwood MI355X; AMD MI200/MI300X
spec sheets; AnandTech MI210.

---

## 1. Winning-solver craft (what beat AMD's AITER baselines) — transferable parts only

Note on availability: the comp kernels are "single-file" HIP. Writeups are public; **source is only
partially public and mostly unlicensed** — the per-kernel license
ledger is in `gpumode-kernel-sources.md` (internal, not distributed). ColorsWind (grand prize) published **neither source nor writeup** that I could find [?].

### 1.1 Snektron — `solution.hip` (AMD FP8 MM) [V, read the raw file]
Transferable systems/ISA craft (ignore the MFMA math):
- **Branchless out-of-bounds handling:** OOB lanes are given `voffset = 0xFFFFFFFF` so the hardware buffer
  load returns 0 instead of taking a branch. → **Directly our pattern:** handle "lane has no work this
  round" or edge limbs with a poison offset / mask, never a `if`.
- **Intrinsics over hand asm:** no `__asm volatile` GCN asm; instead binds LLVM intrinsics by symbol
  (`__asm("llvm.amdgcn.raw.buffer.load.i32")`) and uses `__builtin_amdgcn_*`. Portable and lets the
  scheduler work. Uses `__builtin_amdgcn_perm` (byte-permute) and `__builtin_amdgcn_update_dpp` (cross-lane
  DPP) for a register transpose — **DPP/permute are how you move data between lanes without LDS**, relevant
  to our Montgomery batch-inversion prefix/suffix product scan.
- **Explicit `s_waitcnt` control:** `__builtin_amdgcn_s_waitcnt(vmcnt(0) | exp_cnt(7)<<4 | lgkmcnt(0xf)<<8)`
  to hand-tune memory-vs-compute overlap when the compiler's is suboptimal.
- **Buffer descriptor config word `0x00020000`** (CK's 32-bit format) for raw buffer loads; cache policy
  chosen per-access via a coherency enum (`cache_global` / `cache_stream` / `non_temporal`).
- **LDS bank-conflict padding:** `+4` elements per row on every `__shared__` tile.
- **`__launch_bounds__(256)`** on every kernel to cap registers/occupancy; MFMA gated on `__gfx942__` with
  an fp32 fallback "so we can test on MI210" — **good pattern: compile-time arch dispatch** (`#ifdef
  __gfx90a__` / `__gfx950__`) so one source ships both our tuned variants.

### 1.2 Akash Karnatak — AMD challenge writeup [V]
- **LDS instruction-WIDTH gotcha (high-value):** a conflict-free 4-byte-aligned swizzle still lost perf
  because the compiler emitted `ds_read2_b32` instead of `ds_read_b64`. **Lesson: verify the emitted LDS
  instruction width in the ISA, not just the conflict count.** For us: when we stage limbs/points in LDS
  for batch inversion, confirm we get wide `ds_read_b64/b128`.
- **Persistent kernel:** launch exactly one block per CU (304 on MI300) and loop over work internally to
  kill launch overhead. → **Our grinder is embarrassingly parallel and long-running: use a persistent,
  grid-stride design, one resident wave-group per CU chewing through the candidate stream.**
- **Match accumulator width to the reference** (they used fp32 intermediate, not low-precision atomics).
  → For us: full-width (256-bit / 5×f64) intermediates for field products; only narrow at the very end.

### 1.3 luongthecong123 — `fp8-quant-matmul` (`mn_large.hip`, `mn_small.hip`) [V, README]
- **Occupancy math stated plainly:** two 1024-thread blocks fill the 2048-thread CU; dropping to 50%
  occupancy **doubles per-thread registers 64→128**. `#pragma unroll 1` holds usage at 64 VGPRs.
  → **This is our central tradeoff** (see §2): more VGPRs per lane (fat field state) vs more resident waves.
- **`ds_read_b64` for contiguous data is "much faster than eight `ds_read_u8`"**; `ds_read_b128` for the
  unrolled case. → always widen LDS/global access.
- **Rotating-index LDS layout reduced VGPR pressure vs XOR swizzle** — swizzle choice affects *registers*,
  not just conflicts.
- rocWMMA produced **fewer register spills than raw MFMA intrinsics** (only relevant if we pursue FP64-MFMA
  on MI210/MI300 — prefer the library wrapper there to keep VGPRs down).

### 1.4 Seb V — RDNA3 SGEMM writeup [V] — **flagged: RDNA3, not CDNA**
Useful for the *ISA-editing workflow* (§5.4) and general ideas, but several headline tricks are
**RDNA3-only and do NOT apply to our CDNA targets**: VOPD **dual-issue** (`v_dual_fmac_f32`), WGP-vs-CU LDS
modes (`-mcumode`), the 4-bank cached-read VGPR model, wave32. Transferable bits: watch `scratch_load` in
ISA as the spill signal; `__launch_bounds__` to stop the compiler spilling under double-buffering; use
SGPR base + per-lane VGPR offset to cut address-math VALU; split global loads into small chunks across the
loop to reduce SIMD contention; microbench showed a **~75% VALU-utilization practical ceiling** (don't chase
100%) for their MI300 kernels. That is their observation, not a limit here: our MI210 microbenchmarks issue
at about 97% of peak per effective clock (`docs/design/CDNA-PERF-REFERENCE.md` section 8).

### 1.5 HF "Creating custom kernels for the AMD MI300" [V]
- MI300X: **256 VGPRs/thread max, ≤16 waves/CU** in that article's framing, wave64, 64 KiB LDS + 32 KiB L1,
  **128-bit max global load**. → size our vector loads at 128-bit (`uint4` / `__uint128_t`).
- **`exp2` instead of `exp`** (scale by log2e) for latency — generalize to: **prefer the cheap
  hardware-native op**; for us that means `v_mul_hi_u32`/`v_mad_u64_u32` over any library bignum path.
- **"MI300X has no warp-level barriers, only a thread-block barrier"** → they used inline **ASM to make
  waves wait at barriers** and an int16 LDS semaphore for producer/consumer warp specialization. → If our
  batch-inversion needs a prefix-product handoff between waves, expect to hand-roll the sync.

---

## 2. Occupancy & VGPR pressure — the #1 lever for our kernel [V, CDNA2 register-pressure guide]

Our field state is fat (8×u32 or 5×f64 per element, ×several temporaries for a point-add + the running
batch-inversion product), so **VGPR pressure will gate occupancy**. This is the most important section.

- **gfx90a occupancy ladder:** 4 SIMD/CU × ≤8 waves = **32 waves/CU** max. VGPRs are allocated per-SIMD in
  banks; crossing a threshold drops a wave. Measured example: **102→96 VGPRs bought a 5th wave/SIMD**
  (4→5). Above ~**256 VGPRs/lane → 1 wave and scratch spills** [K].
- **Detect spills two ways:**
  - `hipcc -Rpass-analysis=kernel-resource-usage` → per-kernel remark with `SGPRs, VGPRs, AGPRs,
    ScratchSize [bytes/lane], Occupancy [waves/SIMD], SGPRs Spill, VGPRs Spill, LDS Size [bytes/block]`.
  - or `hipcc --save-temps` → grep `.vgpr_spill_count` / `.sgpr_spill_count` in
    `*hip-amdgcn-amd-amdhsa-gfx90a.s`. (Plain disasm can't distinguish spill from legit scratch.)
- **Control knobs (apply then re-measure — effects shift across ROCm versions):**
  - [ ] `__launch_bounds__(maxThreadsPerBlock, minWavesPerEU)` — 2nd arg is a **minimum waves/EU**, the
    primary occupancy lever; default assumes block size 1024 (over-allocates regs if our block is smaller).
  - [ ] LLVM `amdgpu-waves-per-eu` attribute for finer control.
  - [ ] Declare temporaries **near first use**; avoid stack arrays and large by-value kernel args (they
    land in VGPRs/scratch); `restrict` on pointers (cut SGPRs in the example, may raise VGPRs — measure).
  - [ ] `#pragma unroll 1` (or bounded) on the grind loop — full unroll of a multi-limb schoolbook multiply
    explodes VGPRs; partial unroll trades ILP vs occupancy.
  - [ ] **Spilling is sometimes the right call** (higher occupancy can beat a spill-free low-occupancy
    config) — the guide's worst case is 1 wave/CU, so **always benchmark the spilled vs unspilled build**,
    never assume.
- **Our decision to autotune:** limbs-per-lane radix (8×u32 vs 5×f64), unroll factor, and
  `minWavesPerEU` form a small grid. Sweep it (§5.5), rank by measured throughput, keep per-arch winners.

---

## 3. LDS layout & bank conflicts — for Montgomery batch inversion staging [V]

Point-add itself is register-resident, but **batch inversion** (prefix products → one inverse → suffix
unwind, Montgomery's trick across a batch) needs a shared running product across lanes/waves.

- CDNA LDS has **32 banks of 4 bytes**; conflict = two lanes hit different addresses in the same bank [K].
- **Padding** (`+4`/row, or pad 64→72) is the first fix; an **XOR swizzle** is the conflict-free fix for
  2D tiles — but luongthecong found a **rotating index** cheaper in VGPRs than XOR.
- **Check the emitted width** (`ds_read_b64`, not `ds_read2_b32`) — Akash's gotcha.
- Prefer **DPP / `__builtin_amdgcn_perm` / cross-lane ops** to shuttle the running product *within* a
  wave64 with **no LDS at all** (Snektron's transpose pattern) — likely the fastest batch-inversion scan.
- MI355X has **160 KiB LDS/CU** (vs 64 on MI300/MI210) — more room to stage larger batches on gfx950.

---

## 4. Instruction selection for the field-multiply hot loop [K + V-pointers]

Core op: 256×256→512-bit schoolbook (or 5×5 f64) multiply-accumulate, then reduce mod 2^255-19.
ROCm perf-model confirms **INT32 ADD/LOGIC/MAD and FP64 are supported on CDNA/2/3/4** [V].

### 4.1 Integer radix-2^32 path (8 limbs)
- **`v_mad_u64_u32`** — 32×32→64 fused multiply-add producing a 64-bit result with carry; the single most
  important instruction for schoolbook limb products (one per partial product instead of
  `v_mul_lo_u32`+`v_mul_hi_u32`+adds) [K — confirm exact form/throughput in the **CDNA3 ISA PDF** and
  `amdgpu_isa_cdna3.xml`].
- **Carry chains:** `v_add_co_u32` (add, carry-out to VCC/SGPR) + `v_addc_co_u32` (add with carry-in) build
  the multi-limb accumulation. Keep the carry in a dedicated SGPR/VCC and chain limbs in order; avoid
  recomputing carries [K].
- **`v_lshl_add_u32` / `v_lshl_or_b32`** fuse shift+add/or — useful for limb repacking and the ×19 and ×38
  folds in the 2^255-19 reduction (19 = (16+2+1), or just `v_mad`/`v_mul_lo` by the constant 19) [K].
- **`v_bitop3_b32`** (CDNA3/4) — a LUT3 "any boolean of 3 inputs in one op" (like NV's LOP3); folds
  mask/select/xor sequences in the reduction and in field negation/conditional logic into single ops
  [K — availability per gen in the ISA XML; **flag:** confirm on gfx90a, it may be CDNA3+ only].

### 4.2 FP64 radix-2^51 path (5 limbs, "float-float")
- Uses **`v_fma_f64`** with 51-bit limbs so products fit in f64's 53-bit mantissa; carry handled by
  add-with-rounding tricks. Full-rate FP64 FMA on all our targets [K].
- **FP64-MFMA wildcard (§0.1):** on MI210/MI300 the matrix path is 2x — a prototype casting the 5×5 limb
  outer-product as `v_mfma_f64_16x16x4f64` could win there; **pointless on MI355X**. Prefer a rocWMMA
  wrapper to keep VGPR/AGPR spills down (luongthecong). Accumulators live in **AGPRs** on CDNA — budget
  them separately from VGPRs in occupancy math [K]. Reference model for cycle counts: arXiv 2609.14845
  "Accurate Models of AMD Matrix Cores" [V-exists, not deep-read].

### 4.3 Branchless modular reduction — kills warp divergence [K, directly our bottleneck]
The "branchy modular reductions" the coordinator named are the conditional subtract of p (and the ×19
fold). On a 64-wide SIMD, a per-lane `if (a >= p) a -= p;` **serializes divergent lanes** — forbid it.
- **Pattern:** compute `t = a - p` with borrow via `v_sub_co_u32`/`v_subb_co_u32`; the final borrow bit
  says whether `a < p`; then **`v_cndmask_b32`** selects `a` or `t` per lane with **zero divergence**
  (both results already computed). Generalize: **compute all branches, select with a mask.**
- For the last conditional subtract in 2^255-19 arithmetic, fold the "+19 if it wrapped" the same way.
- Avoid data-dependent `s_cbranch` inside the grind loop entirely; the only acceptable branch is the rare
  "vanity prefix hit → write result," which is uniform-ish and handled by a compacted output queue, not a
  per-lane store (keeps EXEC full).
- EXEC-mask awareness: when lanes do finish early, let them idle under the mask rather than branching;
  a persistent grid-stride loop keeps EXEC densely full.

---

## 5. Feeding the pipes, scheduling & the ROCm tuning WORKFLOW

### 5.1 How CDNA issues (and what does NOT exist)
- **CDNA has NO RDNA3-style VOPD dual-issue.** Do not design around `v_dual_*`. CDNA throughput comes from
  **4 SIMD16 per CU** each running a wave64 over 4 cycles, **many resident waves** hiding latency, and the
  **MFMA units running asynchronously alongside VALU** [K]. So our ILP strategy = enough independent
  field-mul chains in flight (across waves and within a lane) to saturate the VALU, not instruction pairing.
- **Packed ops** `v_pk_add_u16/i16`, `v_pk_mul_*` exist (2-wide 16-bit) — minor relevance (our limbs are
  32/64-bit), but worth a look for the 16-bit-limb variant.
- **Overlap MFMA with VALU:** if we take the FP64-MFMA path on MI210/300, keep VALU busy with carry/reduce
  work while the matrix pipe runs — that overlap is where the 2x actually materializes.
- `s_waitcnt` (vmcnt/lgkmcnt/exp_cnt) and `__builtin_amdgcn_sched_barrier` / `s_setprio` are the manual
  scheduling levers if the compiler mis-schedules (Snektron used explicit `s_waitcnt`; Seb V warned inline
  asm needs explicit deps or the scheduler moves it wrongly).

### 5.2 Compiler flags (hipcc / clang) [V]
- [ ] `--offload-arch=gfx90a` and `=gfx950` (build both).
- [ ] `--save-temps` → inspect `*.s` (`.vgpr_count`, `.vgpr_spill_count`, `.sgpr_spill_count`).
- [ ] `-Rpass-analysis=kernel-resource-usage` → occupancy/spill remarks at compile time.
- [ ] `__launch_bounds__` + `-mllvm -amdgpu-waves-per-eu=N` to pin the occupancy target.
- [ ] `-ffast-math` only on the FP64 path and only if it preserves the exact limb arithmetic (it will NOT
  for float-float carry tricks — **keep FP64 strict**; flag per-op).
- [ ] Consider `-mllvm -amdgpu-unroll-threshold` / `#pragma unroll N` to bound unrolling.

### 5.3 Counter-driven profiling with rocprof-compute (omniperf) [V]
Workflow: get a Speed-of-Light first, then drill into the CU block.
- **Wavefront analysis:** launch stats (`grid`, `workgroup`, **`VGPR`/`AGPR`/`SGPR` usage**) + runtime stats
  (**wavefront occupancy**, active cycles). First question: are we VGPR/LDS/SGPR-limited on occupancy?
- **Instruction mix:** counts of **VALU / VMEM / SALU / LDS / MFMA (matrix)** instructions issued →
  confirms we're VALU/MAC-bound (expected) not memory-bound.
- **VALU utilization / busy** → how close to peak (Seb V saw ~75% on MI300 kernels; simple MI210 loops
  reach ~97% per effective clock, CDNA-PERF-REFERENCE section 8); low VALU busy with low
  occupancy = raise waves; high VALU busy = we're instruction-bound and must cut *instructions* (better
  reduction, `v_bitop3`, fewer carries).
- **LDS bank-conflict rate** = cycles in LDS scheduler due to conflicts / uncontended base cycles → tune §3.
- **Speed-of-Light / roofline** to see headroom vs the (integer or FP64) peak.
- Counter reference: ROCm `rocprofiler-compute` CDNA performance model + MI300/MI200 counter list [V].

### 5.4 Reading / hand-editing ISA [V]
- `llvm-objdump -d --mcpu=gfx90a solution.hsaco` (or `roc-obj`) to disassemble; count `v_mad_u64_u32`,
  carries, `v_cndmask`, and look for `scratch_load/store` (spills) and `ds_read2_b32` (narrow LDS).
- `--save-temps` `.s` carries the metadata disasm lacks; edit `.s`, reassemble
  (`hipcc -target amdgcn-amd-amdhsa -mcpu=gfx90a -c`), link (`ld.lld -shared`), load the `.hsaco` via HIP
  Module API — Seb V's loop for when you must beat the register allocator by hand (update `.vgpr_count`
  metadata after). `rocgdb` for stepping if correctness diverges.

### 5.5 Autotuning loop (adopt this) [synthesis]
Persistent-kernel grinder + a sweep over {radix (u32 vs f64), unroll, `minWavesPerEU`, LDS layout,
arch}. For each config: compile, read spill/occupancy remark, run the throughput benchmark on the real
card, **append one JSONL line per config to disk immediately** (so a crashed sweep keeps its data), rank
by measured candidate-keys/sec, keep the per-arch winner. Mirror the GPU MODE eval discipline (see §6).

---

## 6. Correctness/anti-overfit discipline borrowed from the GPU MODE eval harness [V, from eval.py]

Even though this is our *build* side, the GPU MODE ranked-eval design is the gold standard and we should
self-impose it while tuning so we never ship a fast-but-wrong kernel:
- **Re-verify on FRESH random inputs every timed iteration** (GPU MODE bumps the seed +13 each iter) — for
  us, re-randomize the candidate seed/prefix each benchmark iteration and re-check that every emitted
  keypair is a valid ed25519 point encoding to the correct v3 .onion matching the target. Prevents a
  kernel that accidentally special-cases one input.
- **Adaptive timing:** warmup once, then loop to a relative-std-error floor (GPU MODE: `err/mean < 0.1%`,
  min 3 / cap ~100 runs / 120 s), report the **mean**, clear caches between runs.

---

## 7. AMD-specific gotchas that bite

- **VGPR allocation cliffs** (§2): perf is a step function of occupancy; a 2-VGPR diet can buy a whole wave.
  Always check the resource-usage remark, not just runtime.
- **LDS instruction-width regression:** conflict-free layout can still emit narrow `ds_read2_b32` and lose —
  verify width in ISA (Akash).
- **No warp-level barrier on MI300-class** — only block-level; cross-wave handoff needs hand-rolled
  ASM/semaphore (HF MI300 blog). Affects a multi-wave batch-inversion scan.
- **Inline asm & the scheduler:** without explicit deps the compiler relocates asm incorrectly (Seb V) —
  prefer `__builtin_amdgcn_*` intrinsics (Snektron) and only drop to asm with `volatile` + `s_waitcnt`.
- **Scratch spills are near-invisible in plain disasm** — need `.vgpr_spill_count` metadata or the remark.
- **FP64 is NOT uniform across our targets** (§0.1): MI355X has *less* FP64 than MI300X and no MFMA bonus;
  a kernel tuned FP64-heavy for MI210/300 may be the wrong choice on gfx950.
- **`-ffast-math` will break float-float carry arithmetic** — keep the FP64 field path strict.
- **hipRTC vs offline compile:** hipRTC (runtime compile) is convenient for the autotuner (compile each
  config in-process, no toolchain shell-out) but **does not emit the `--save-temps` `.s` / resource-usage
  remarks** the same way — do the *analysis* builds offline with `hipcc`, use hipRTC only for the fast
  sweep once flags are fixed [K — verify remark availability under hipRTC on our ROCm version [?]].
- **wave32 assumption would be a bug** on CDNA — hardcode/confirm 64 (§0).

---

## 8. Could-not-verify / honesty flags

- **ColorsWind (grand prize): no public source and no public writeup found** [?]. Named only in the GPU MODE
  recap. Our best actual winning-kernel corpus is the HF dataset (license-restricted — see ledger).
- **Exact `v_mad_u64_u32` / `v_bitop3_b32` encodings, per-gen availability, and throughput** not read from
  the ISA here — stated from architectural knowledge; **confirm in the CDNA3 ISA PDF and
  `amdgpu_isa_cdna3.xml` / cdna2 XML**, especially whether `v_bitop3` exists on gfx90a (likely CDNA3+ only).
- **FP64-MFMA cycle counts** for `v_mfma_f64_16x16x4` not pulled (arXiv 2609.14845 models them; not
  deep-read). The 2x peak ratios ARE verified from vendor specs.
- The **HLRS omniperf workshop PDF** would not extract as text (binary); omniperf metric names here come
  from the ROCm docs + search, cross-checked, but panel-exact names should be confirmed by running
  `rocprof-compute` on our card.
- **"CDNA5 wave32 shift": unconfirmed** — no current ROCm spec shows wave32 on any CDNA part.
- Seb V's techniques are **RDNA3**, re-measure any parameter on CDNA.

## 9. Source URLs
- GPU MODE recap / writeup links (kernelboard PR): https://github.com/gpu-mode/kernelboard/pull/49/files
- Snektron FP8 MM (source, **no license**): https://github.com/Snektron/gpumode-amd-fp8-mm
- luongthecong123 FP8 quant matmul (source, **no license**): https://github.com/luongthecong123/fp8-quant-matmul
- Seb V RDNA3 SGEMM: https://seb-v.github.io/optimization/update/2025/01/20/Fast-GPU-Matrix-multiplication.html
- Akash Karnatak AMD challenge: https://akashkarnatak.github.io/amd-challenge/
- Fan Wenjie (Bilibili): https://www.bilibili.com/read/cv41954307/
- HF "Creating custom kernels for the AMD MI300": https://huggingface.co/blog/mi300kernels
- GPUMODE/kernelbot-data (actual submissions, **Researcher Reciprocity License**): https://huggingface.co/datasets/GPUMODE/kernelbot-data
- CDNA register-pressure guide (gfx90a): https://rocm.blogs.amd.com/software-tools-optimization/register-pressure/README.html
- rocprof-compute CDNA perf model: https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/conceptual/cdna/cdna-performance-model.html
- MI300/MI200 counters: https://rocm.docs.amd.com/.../gpu-arch/mi300-mi200-performance-counters.html
- CDNA3 ISA (confirm instruction encodings): linked from https://rocm.docs.amd.com/.../reference/gpu-arch/index.md ; machine-readable XML: https://gpuopen.com/machine-readable-isa/
- AMD matrix-core model (FP64 MFMA): arXiv 2609.14845
- ROCm GPU specs (wave64, LDS, VGPR): https://rocm.docs.amd.com/en/latest/reference/gpu-specs.html
- MI355X FP64: https://glennklockwood.com/garden/processors/mi355x ; MI210: https://www.anandtech.com/show/17326
