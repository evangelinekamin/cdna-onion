# CDNA Performance Reference for ed25519 Vanity Generation

Distilled from the AMD sources in `docs/*.pdf` (two whitepapers, three ISA
reference guides, ~1,700 pages total) down to only what governs throughput for
this one workload: batched ed25519 point addition with Montgomery batch
inversion. Read this instead of the PDFs; the PDFs remain the authority when a
row says "confirm".

Date distilled: 2026-10-15.

---

## 1. Generation -> hardware map

| Gen | GPUs | gfx target | Wave | Our role |
|---|---|---|---|---|
| CDNA2 | MI210, MI250, MI250X | gfx90a | wave64 | **Test box** (Sol AMD partition is MI210) |
| CDNA3 | MI300X, MI300A, MI325X | gfx942 | wave64 | Optional interim target |
| CDNA4 | MI350X, MI355X | gfx950 | wave64 | **Scale target** (the 8x MI355X node) |
| CDNA5 | MI400, MI455X | gfx9?? | **wave32** | Future only; we have no access |

"MI200x" on Sol = MI210, CDNA2, gfx90a, same ISA as MI250X (single die vs dual).
The prior feasibility report's "MI355X = CDNA4" is correct.

---

## 2. What governs throughput for THIS workload

Established from the onion-gpu/honion write-ups, reconfirmed against these docs:

- **Instruction-bound on integer multiply-add, not memory-bound.** The cost per
  candidate is a fixed count of 255-bit field multiplications, each built from
  32-bit (or 64-bit) integer MACs with carry. onion-gpu measured ~147 GB/s of
  ~448 GB/s local traffic: the kernel is ALU-bound.
- **Irrelevant here:** HBM bandwidth (so H200-vs-H100, MI300X-vs-MI325X buys
  nothing), tensor/matrix cores, and FP4/FP6/FP8. The MI350/MI400 marketing axis
  (low-precision AI formats) does not touch this kernel.
- **The two levers that matter:** (a) the throughput of the integer-MAC (or
  FP64-FMA) primitive the field multiply is built on, and (b) VGPR occupancy,
  which caps how large the batch-inversion group N can be before spilling.

---

## 3. Bignum-relevant instructions (presence confirmed by grep of the ISA docs)

Counts are mentions in each ISA text; presence, not rate. "-" = absent.

| Instruction | Semantics (key ones) | CDNA2 | CDNA4 | CDNA5 |
|---|---|---|---|---|
| `V_MAD_U64_U32` | `{vcc,D.u64}=S0.u32*S1.u32+S2.u64` (the radix-2^32 workhorse) | yes | yes | yes |
| `V_MUL_HI_U32` / `V_MUL_LO_U32` | 32x32 -> hi/lo 32 | yes | yes | yes |
| `V_ADD_CO_U32` / `V_ADDC_CO_U32` | add with carry-out / carry-in (manual carry chains) | yes | yes | **ADDC absent** |
| `V_MAD_CO_U64_U32` / `V_MAD_NC_U64_U32` | 64-bit MAC with / without explicit carry | - | - | **yes (new)** |
| `V_LSHL_ADD_U64` | fused shift+add (limb packing) | - | yes | yes |
| `V_BITOP3_B32` | 3-input LUT bitwise (NVIDIA LOP3 analog) | - | yes | yes |
| `V_FMA_F64` | fused FP64 multiply-add (radix-2^51 path) | yes | yes | yes |
| `V_MFMA_F64_*` | FP64 matrix (not useful here) | yes | yes | - |
| `V_PERMLANE*` / `DS_BPERMUTE_B32` | cross-lane (only if a design shares limbs across lanes) | bperm only | +swap | full family |

Reading of the deltas that matter to us:

- **CDNA2 -> CDNA4:** same integer-MAC core. CDNA4 adds `V_LSHL_ADD_U64`,
  `V_BITOP3_B32`, and lane-swap permutes. `V_BITOP3_B32` mainly helps the SHA3
  (Keccak) checksum, which runs only on a hit, so it is a minor win. No new
  integer-mul primitive -> the field multiply is written the same way on both.
- **CDNA4 -> CDNA5:** the real ISA jump for bignum. Native 64-bit MAC with
  explicit carry (`V_MAD_CO/NC_U64_U32`) cleans up the carry chain, and the part
  goes wave32. Not our target, but it is why a future MI400 port would be
  re-tuned, not re-ported.

---

## 4. The FP64 fork (the single most design-shaping fact)

The field multiply has two viable limb representations, and the better one is
**hardware-dependent across our two targets**:

- **Integer radix-2^32** (eight 32-bit limbs) on `V_MAD_U64_U32` + carry chain.
  This is honion's representation. Portable, carry-managed.
- **FP64 radix-2^51** (five 51-bit limbs) on `V_FMA_F64`. This is the
  DPF-ECC / gECC approach: field multiplies become FP64 FMAs, no explicit carry.

Why it splits. Verified FP64 peak per part (vendor specs via
`docs/research/gpumode-craft.md`; the whitepaper's 47.9 TF is the dual-die
MI250X, so do not apply it to MI210):

| Part | Gen | FP64 vector | FP64 matrix (MFMA) | Matrix bonus |
|---|---|---|---|---|
| MI210 (test box) | CDNA2 | 22.6 TF | 45.3 TF | **2x** |
| MI250X | CDNA2 | 47.9 TF | 95.7 TF | 2x |
| MI300X | CDNA3 | 81.7 TF | 163.4 TF | 2x |
| MI355X (scale box) | CDNA4 | 78.6 TF | 78.6 TF | **none** |

- **MI210 (test box):** FP64 is first-class (22.6 TF vector, ~1:1 with FP32) and
  the matrix path **doubles** it -> the radix-2^51 FP64 multiply is a strong
  contender, and the FP64-matrix limb idea (sec below) is worth prototyping here.
- **MI355X (scale box):** the MI350 series deprioritized FP64 -- absolute FP64
  *regressed below MI300X*, and matrix == vector (no MFMA bonus). So on gfx950
  the FP64-matrix wildcard is dead and integer radix-2^32 or plain VALU-FP64 is
  the likely winner. (Figures are the agent's vendor-spec read; confirm by
  microbench.)

**The sharpened consequence:** the FP64-matrix (rocWMMA/MFMA) path is a potential
2x *only on MI210/MI300X, not on MI355X* -- so the thing that may win on the test
box is explicitly the wrong horse on the scale box. Expect to **ship two
per-arch-tuned variants** (`#ifdef __gfx90a__ / __gfx950__`) and never assume
MI210 tuning transfers.

**Consequence for the whole project:** the fastest kernel is probably not the
same representation on MI210 and MI355X. Tuning only on the test box and scaling
blind would mislead. Therefore field arithmetic is a swappable backend, and the
bench harness runs *both* backends on *both* archs. See `ARCHITECTURE.md` sec 5.

---

## 5. Occupancy budget (register file, LDS, wave)

| Resource | CDNA2 | CDNA4 | CDNA5 | Relevance |
|---|---|---|---|---|
| Arch VGPR / lane | 256 | 256 | 512 | Caps batch-inversion group N before spill |
| Acc VGPR / lane | 256 | 256 | (unified) | MFMA-only; unused here unless we borrow them |
| Wave size | 64 | 64 | 32 | Scheduling, not algorithm (kernel is per-lane independent) |
| LDS / CU(WGP) | 64 KB | ~64 KB (confirm) | 320 KB/WGP (documented) | Minor; batch state lives in VGPR/global, not LDS |

- A point in extended coords at radix-2^32 is ~16 x u32 = 64 B; its Z (for batch
  inversion) is 32 B. 256 VGPR/lane (1 KB/lane) is the real constraint on how
  large N can be held in registers before spilling to global, which is why
  onion-gpu reserves VRAM for batch state rather than keeping it in LDS.
- Wave64 vs wave32 does not change the math: every lane walks its own
  independent point. It changes only occupancy arithmetic and any (optional)
  cross-lane limb sharing.

---

## 6. Documented vs must-measure

These PDFs are ISA/architecture references. They give instruction **semantics
and operand forms**, register-file sizes, LDS sizes, wave size, and peak FLOPS
(whitepapers). They do **not** give integer-ALU **issue rates** / cycles per
instruction. So the following are unknown from the docs and are exactly what the
first microbenchmarks must produce (per gen, per backend):

1. Sustained `V_MAD_U64_U32` throughput (the integer field-mul ceiling).
2. Sustained `V_FMA_F64` throughput (the FP64 field-mul ceiling) -> ratio to (1)
   on CDNA2 vs CDNA4; this decides the fork empirically.
3. Achieved field-multiplies/second for each backend inside the real point-add
   loop (not in isolation), at the best (N, threads, blocks) found by autotune.
4. VGPR pressure / occupancy at each N, and the N where spilling starts.

Everything downstream (keys/s projections for MI210 and MI355X) derives from
these four measurements. Until they exist, any per-GPU keys/s number for Instinct
is an extrapolation and must be labeled as one.

---

## 7. Source map (so nothing is lost)

| File in `docs/` | What to go back to it for |
|---|---|
| `instinct-mi200-cdna2-instruction-set-architecture.pdf` (275 p) | CDNA2 opcode semantics, operand encodings, LDS banking |
| `amd-instinct-cdna4-instruction-set-architecture.pdf` (608 p) | CDNA4 (gfx950) opcodes; new `V_BITOP3_B32`, `V_LSHL_ADD_U64`, permute-swap |
| `amd-instinct-cdna5-instruction-set-architecture.pdf` (832 p) | CDNA5 wave32, `V_MAD_CO/NC_U64_U32` (future MI400) |
| `amd-cdna2-white-paper.pdf` (17 p) | CDNA2 FP64 = 47.9 TF 1:1 claim; CU array; cache |
| `amd-cdna5-whitepaper.pdf` (24 p) | wave32 rationale, LDS 320 KB/WGP, MI355X/MI455X positioning |

These are AMD's copyrighted publications, so git does not carry them:
`tools/fetch-amd-docs.sh` downloads them from amd.com and checks each against
the edition cited here. For grepping, `pdftotext -layout` extracts the text.

---

## 8. Measured so far (2026-10-16)

### The faithful port on CDNA (static, ROCm 7.2.4 compiler)

`entries/baseline` is onion-gpu (ref10 10x25.5-bit limbs, int64 accumulators)
with upstream's tuning unchanged. Its `grind_kernel` resource usage:

| | gfx90a (MI210) | gfx950 (MI355X) |
|---|---|---|
| Arch VGPRs / AccVGPRs | 256 / 6 | 256 / 14 |
| Scratch per lane | 31,340 B | 31,316 B |
| SGPR spills | 143 | 145 |
| **Occupancy** | **1 wave/SIMD** | **1 wave/SIMD** |

The ref10 multiply fits NVIDIA's ~170-register budget because NVIDIA has a
native wide multiply-add. On CDNA every int64 accumulator costs two VGPRs, so a
straight port maxes the register file and leaves no second wave to hide the
latency of the ~30 KB of batch arrays in scratch. **Occupancy is the first lever**
(this matches the GPU MODE craft notes).

Hot-loop instruction mix (`make -C entries/baseline isa`):

| | gfx90a | gfx950 |
|---|---|---|
| `v_mad_u64_u32` / `v_mad_i64_i32` | 1751 / 812 | 1751 / 812 |
| `v_mul_lo_u32` | 1310 | 1310 |
| 64-bit add | 208 `v_add_co_u32` + 190 `v_addc_co_u32` | **208 `v_lshl_add_u64`** |
| `v_readlane` / `v_writelane` (SGPR spill) | 151 / 143 | 151 / 145 |

gfx950 fuses each 64-bit add pair into the CDNA4-only `v_lshl_add_u64`, which
section 3 found in the ISA guide. Here it shows up in real codegen.

### Primitive throughput (`tools/microbench`)

Each kernel was confirmed in the ISA to compile to exactly the instruction it
measures. Units: operations per compute unit per nominal clock. Each figure is
the mean of 5 samples of about 100 ms, taken after the clocks settle; on the
4060 Ti, three runs from an idle GPU agree within 0.5%. A figure above the
part's lane width (64 integer lanes per SM on Ada) means it boosted above its
nominal clock.

| Primitive | RTX 4060 Ti (sm_89), reference | MI210 | MI355X |
|---|---|---|---|
| `v_mad_u64_u32` (32x32+64) | 31.8 | **TBD (run on Sol)** | TBD |
| `v_mul_lo_u32` | 67.0 | TBD | TBD |
| `v_mul_hi_u32` + add | 33.6 | TBD | TBD |
| `v_fma_f64` | 1.78 (Ada consumer FP64 is 1/64 rate) | TBD (expect about 64, full rate) | TBD |
| FP32 FMA (`v_pk_fma_f32` on CDNA) | 113.5 | TBD (expect about 128, packed) | TBD |
| ref10 `fe_mul` | 0.166 (14.7 G/s) | TBD | TBD |

On the 4060 Ti, the baseline's hot loop runs at about 78% of the isolated
`fe_mul` rate: field-multiply bound, as expected. The MI210 column decides the
integer-vs-FP64 fork (section 4).

### Benchmark methodology finding

The first benchmark of the baseline used 22 targets at about 15,000 hits/s and
measured 734 M/s. With real-search conditions (2 targets, about 1,800 hits/s) the
same kernel measured **954 M/s (95% CI 946 to 962)**. The 30% gap was per-candidate
target compares plus onion-gpu's per-hit self-check, a full scalar
multiplication during which the rest of that lane's SIMD group sits idle. Real
10-character searches almost never hit, so the rig now caps benchmarks at 4
targets and about 2,000 hits/s (`plan_for`). Per-hit cost is real, but it must
not decide a competition about search throughput.
