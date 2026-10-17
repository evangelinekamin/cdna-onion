# onion-walk: working notes

A log of what was tried and what was measured, newest last. "VALU/pair" counts
VALU instructions in a loop body from `make isa` (static); rates are the
arena's MI210 measurements unless marked otherwise.

## Design (2026-10-16)

- **Algebra.** Dual addition law, y-only, renormalized by x1*x2 so that a pair's
  two denominators multiply to a *free* q_j = s_j^2 - lam^2 (table constant minus
  a per-batch constant; no multiply). Numerators become two-term inner
  products alpha_j, beta_j of per-batch values with table constants, each reduced
  once: y(C +- T_j) = (alpha_j +- beta_j) / q_j. Per pair of candidates: 1 forward
  multiply (prefix product), 2 backward (Montgomery), 2 inner products (4 limb
  products, 2 reductions), 2 partial leaves. ~3.6 multiply-equivalents per
  candidate vs ~5.5 for the published dual-law kernels.
- **Scratch.** Montgomery's trick normally stores 2 field elements per candidate
  (prefix product and denominator). Here the denominators are free to recompute,
  so the forward pass stores only P_{j-1}: one element per *pair*, 36 bytes, a
  quarter of the usual traffic.
- **Prefix filter ("leaf").** The final multiply y = (alpha +- beta) * (1/q) only
  needs y's low 64 bits. Computing the 4 columns that reach them plus the 2 top
  columns (to estimate the fold quotient Q within +1) takes 41 of 81 limb
  products; the true low 64 bits are base, base+19 or base+38, so the GPU tests
  three values (never a miss; ~3x false positives, all rejected on the host).
- **Field.** 9 x 29-bit limbs: products accumulate carry-free in 64 bits (one
  v_mad_u64_u32 each), reduction folds with 2^261 = 1216 and carries
  sequentially, each carry seeding the next column's accumulator.
- **Correctness net.** Every GPU-flagged offset is re-derived on the host with an
  independent radix-2^51 implementation before it is reported.

## Static ISA (gfx90a, ROCm 7.2.4, H = 1024)

| Step | Forward VALU/pair | Backward VALU/pair | VGPRs / spills | Note |
|---|---|---|---|---|
| plain C++ MADs | 175 | 948 | 128 / 178 (4 waves) | LLVM reassociates carries: +2 instr/limb |
| asm MADs, SGPR table operands, buffer scratch | 131 | 866 | 128 / 74 | spills only outside the hot loops |

## MI210 measurements

| Version | What changed | check (5 s) | bench (60 s, 95% CI) |
|---|---|---|---|
| 0.1 | first GPU build: asm MADs, SGPR table operands, buffer scratch, 3-value filter, target loop | 13.34 G/s | **13.137 G/s (13.126-13.149)**, 3 runs |
| 0.3 | 32-bit exact filter + rare full path, SGPR targets, streaming high columns, layout-2 scratch | 14.21 G/s | **14.250 G/s (14.125-14.375)**, 3 runs; 102,269 flagged = 102,269 verified (exact filter) |
| 0.5 | CKPT2 (half the scratch traffic), H=2048, two lane sets on two streams | 16.03 G/s | **15.834 G/s (15.611-16.057)**, 3 runs |
| 0.6 | signed-limb filter, fold_top without the zero multiply-add, one addressing VGPR (BLOCK=64), H=4096 | 16.34 G/s | **16.091 G/s (15.996-16.185)**, 3 runs |

v0.1 VALU accounting: ~1082 VALU per pair (forward 131, backward 866, target loop
~60 more at 4 targets, inversion ~25) = ~541 per candidate. At 13.14 G/s that is
7.1e12 lane-ops/s, about 76% of the 9.3e12 VALU peak at a 1.4 GHz effective clock.
So the kernel is VALU-bound and well fed; further speed has to come mostly from
fewer instructions per candidate.

## A/B probes on the MI210 (probe/: interleaved rounds, self-timed, 6 s each)

The arena's `check` only surfaces the last 200 characters of a non-JSON
`--verify` stdout, so `probe/run.sh` prints one compact line of mean rates.
Self-timed rates read ~0.5% above the rig's (no protocol overhead); only the
ratios matter.

probe3 (3 rounds x 6 s, 4 targets x 5 chars):

| Variant | G/s | spread | vs v0.1 |
|---|---|---|---|
| v0.1 (benchmarked 13.137) | 13.177 | 1.5% | 1.000 |
| v0.2 (3-value filter gone, 64-bit exact filter, SGPR targets, [j][lane][9] scratch, prefetch) | 13.064 | 1.4% | 0.991 |
| current: 32-bit filter, streaming high columns, [j][lane][9] scratch | 13.842 | 0.9% | 1.050 |
| current with limb-major scratch (layout 0) | **14.337** | 0.7% | **1.088** |
| current without the P prefetch | 13.761 | 0.7% | 1.044 |
| current with the 64-bit filter | 13.176 | 0.6% | 1.000 |

Lessons: the [j][lane][9] layout (two dwordx4 + one dword per element, 36-byte
lane stride) is ~3.5% slower than nine fully coalesced dword accesses; the
32-bit filter is worth ~5%; the one-step P prefetch ~0.6%.

probe4 (base = limb-major scratch, everything else as probe3 "current"):

| Variant | G/s | vs base |
|---|---|---|
| base | 14.432 | 1.000 |
| no P prefetch | 14.257 | 0.988 |
| H = 2048 | 14.425 | 1.000 (inversion savings ~1.3% eaten by something else) |
| 3 waves/SIMD (168 VGPRs) | 14.178 | 0.982 |
| carry via v_alignbit_b32 + v_lshrrev_b32 instead of v_lshrrev_b64 | 13.917 | 0.964 |
| PROFILE build (clock reads + atomics) | 13.109 | 0.908 |

So v_lshrrev_b64 issues at full rate (one instruction beats two), and four
waves per SIMD beat three. Profile, s_memtime cycles per wave per pair of
candidates: setup 33, forward 1920, inversion 281, backward 11749, center 21.

probe5: scratch layout 2 (per-wave limb-major blocks, immediate offsets, no
SGPR offsets) 14.160 vs layout 0 14.071 G/s (+0.6%, within noise); both with
32-bit fast targets. Absolute rates drift ~2% between probes (GPU state), so
compare only within a probe. PROFILE build: effective shader clock (s_memtime
vs s_memrealtime) **1180 MHz** under this kernel, against 1400 MHz for the
plain v_mad_u64_u32 microbenchmark: the card is power-limited here.

Spill-induced serialization (found in the ISA, fixed after v0.3): a VGPR spill
reload in the backward loop (nLam limb 8) was issued after the nine prefetch
loads, and its `s_waitcnt vmcnt(0)` therefore waited for the prefetch too: a
full memory latency per iteration. Fixes: the center offset k moved out of
the register-resident lane state into memory (only the rare hit path and the
once-per-batch advance touch it), and nLam is parked in LDS (9 dwords per lane,
36 KiB per CU), which LDS reads (lgkmcnt) can never serialize against vector
memory. The backward loop now issues the next element's loads at its top and
waits for them only in the next iteration.

probe6 (PROFILE build, v0.3 code + timing): effective clock 1169 MHz; host wall
8.40 s = GPU kernel time 8.39 s (no launch gaps), but each wave is alive (first
to last instruction, s_memrealtime) only 6.33 s on average: 75% of the kernel.
Every v0.x build so far used scratch (spills, 200-700 B/lane). Hypothesis: the
scratch-wave limit keeps fewer waves resident than the register occupancy (4
per SIMD). The kernel now has no scratch at all; probe8 tests it.

probe7: k in memory + nLam in LDS (backward loop free of spill reloads, but
prologue spills remained): 14.369 vs v0.3 14.324 G/s (+0.3%, noise). The
serialized prefetch was mostly hidden by the other waves.

Tried and dropped: software-pipelining the table entry (load j-1 at the end of
step j) needs ~45 more SGPRs: 152 SGPR spills. MFMA (FP64 matrix) offload of
the inner products: exact FP64 products need <= 24-bit limbs (11 per element),
so the four per-batch operands would take 88 VGPRs in MFMA layout instead of
36; not viable at 4 waves/SIMD.

probe8: zero-scratch build 14.327 vs v0.3 14.324 G/s: the scratch-wave-limit
hypothesis is wrong. The 68-75% "alive" ratio fits oldest-first wave
arbitration instead: older waves on a SIMD finish first while the younger ones
stay resident, so the SIMD stays busy. Kernel time equals wall time. So the
kernel is VALU-bound at about 89% utilization at ~1175 MHz, and fewer
instructions remain the lever.

Instruction trimming after v0.3 (static, backward main path per pair):
- one rare branch for both leaves + `-fno-slp-vectorize`: 811 -> 794 (the SLP
  vectorizer had packed the second leaf's four target compares into a bool
  vector in VGPRs);
- `BWD_UNROLL=2` removes the nine prefetch-rotation copies: 794 -> 785 (bigger
  code; to be measured).

probe9 (zero-scratch build): 0.3 s launches 14.350, 1.0 s launches 14.304,
H = 2048 14.358, H = 768 14.144 G/s. Launch length does not matter; H = 2048
saves ~12 VALU/pair on paper but gains nothing (bigger table and scratch
footprint); H = 768 costs 1.4%. Keep H = 1024, 0.3 s.
Tried: holding nLam in registers through the forward loop (saves the LDS
reads) creates 24 VGPR spills elsewhere; reverted.

probe10 (PROFILE builds): scratch aliased to one element (wrong results, same
instruction stream, almost no HBM traffic) runs **16.86 vs 14.37 G/s (+17%)**,
with the shader clock at 1304 vs ~1175 MHz. So the 72 B/pair of scratch
traffic (~500 GB/s) costs ~11% in clock (power) and ~6% in stalls. Caveat:
aliased data could switch less. probe13 measures the marginal cost by
doubling the real traffic instead.

probe11: v0.3 14.289, zero-scratch 14.290, + single rare branch / no SLP
14.418 G/s (+0.9%).

probe12: BWD_UNROLL=2 14.585 vs 14.500 G/s (+0.6%, within noise; the nine
prefetch-rotation copies per pair are gone).

CKPT2 (store P only every other step; backward does two steps from P_{j-2}
with A = P_{j-2} I, 1/q_j = A q_{j-1}, 1/q_{j-1} = A q_j, I <- I q_j q_{j-1}):
static backward main path 1700 VALU per two steps (850/step vs 794), i.e. about
+6% instructions for half the scratch traffic. Measured in probe14.

probe13 (marginal cost of scratch traffic, real data): doubling it (every
element written and read twice) 11.945 vs 14.510 G/s (-17.7%); PROFILE builds
1115 vs 1170 MHz, so only ~4.6% is clock and ~13% is memory stalls. A likely
reason: the forward loop stores 2304 bytes per wave per ~132 VALU. With every
wave in its forward pass, that is ~2 TB/s, above HBM bandwidth, so the forward
phase is bandwidth-starved in bursts even though the average is ~500 GB/s.
Remedies: fewer bytes (CKPT2, CKPT4), or spreading the stores over time (would
need the next batch's forward pass interleaved with this batch's backward pass,
which needs ~18 more VGPRs or LDS that is not there).
Caveat found afterwards: the doubled loads were consumed at once (XOR into an
asm barrier), not one iteration later like the real prefetch, so the variant
also added a memory-latency stall per iteration; -17.7% overstates the cost of
the traffic itself. CKPT2 vs base (probe14) is the direct test.

probe14: **CKPT2 15.055 vs one-step 14.493 G/s (+3.9%)**; unroll-2 of the
one-step loop 14.554 (+0.4%). Halving the scratch traffic is worth ~10% even
after paying ~6% more instructions; the traffic costs ~20% at full volume.
CKPT2 is the default from v0.4.

CKPT4 (tree: B = P_{j-4} I, B01 = B Q23, B23 = B Q01, then 1/q from B01/B23;
11 multiplies per 4 steps): spills (25 VGPRs even after reordering), because
B23 must live across two pairs and the pair working set already fills 128
VGPRs. LDS holds only one 9-dword value per lane (nLam) at 1024 lanes per CU.

probe15 (CKPT2 unless noted): H=1024 15.110, H=2048 15.261 (+1.0%), CKPT4
(spilling) 14.370, CKPT4 H=2048 14.519 G/s. With half the scratch footprint
a larger batch now pays (inversion amortization). CKPT4 dropped: its spills
cost more than the traffic it saves.

v0.4 (CKPT2, H=1024) conformance check: all pass, grind 15.17 G/s.

Hypothesis for CKPT2's large gain: the forward loop issues nine dword stores
per element, and a wave can hold only 63 outstanding vector memory operations
(vmcnt), so with long HBM write latency the forward pass may be bound by the
number of stores in flight rather than by bandwidth. Layout 3 keeps
coalescing but uses three stores per element (two lane-contiguous 16-byte
groups and one dword group); probe17 tests it.

Idea under evaluation (needs 3 waves/SIMD for registers): put W (the giant
step's factors) at the end of the chain instead of the start, so the next
center is known right after the inversion; then run batch b+1's forward pass
interleaved with batch b's backward pass (double-buffered scratch). Stores and
loads would be spread evenly instead of bursting in forward passes, and each
wave would have two independent chains for ILP. Extra registers: the forward
chain's P and nLam (18), within 168 at 3 waves. probe18 measures the 3-wave
cost first.

probe16 (CKPT2): H=1024 15.107, H=2048 15.266, H=4096 15.315 G/s; PROFILE
build at H=2048: **1240 MHz** effective clock (~1170 before CKPT2). Most of
CKPT2's gain is clock: HBM traffic eats the power budget. H=2048 default.

probe17: layout 3 (three memory ops per element instead of nine) 15.259 vs
layout 2 15.308 G/s (CKPT2, H=2048); with CKPT1 14.318. Not the number of
memory instructions; the bytes are what cost.

PIPE (walk_pipe.h): 139 VGPRs, no spills, 3 waves/SIMD (LDS: two nLam banks,
18 KiB per 256-lane block). Interleaved loop 1963 VALU per two steps (1700
backward + 263 forward), the same work as CKPT2, reordered.

probe18 (H=2048): CKPT2 at 4 waves/SIMD 15.293, CKPT2 at 3 waves 14.913
(-2.5%), CKPT4 at 3 waves (no spills) 14.146 G/s. The tree's extra multiplies
cost more than its traffic saving, and a 3-wave pipeline must win back 2.5%.

probe19/20 (H=2048): CKPT2 at 4 waves 15.289 / 15.280; pipelined CKPT2 at 3
waves 14.814 / 14.966; pipelined H=4096 14.846; pipelined CKPT4 13.824 G/s.
Spreading the stores evenly over time gains nothing over CKPT2 at 3 waves:
the traffic costs clock (power), not burst stalls. PIPE removed (git history
has it: walk_pipe.h).

v0.4 (CKPT2, H=2048) conformance check: all pass, grind 15.45 G/s.

Launch tails: waves finish a launch at different times (oldest-first issue;
"alive" 68-75% of the kernel), and the next launch in the same stream starts
only when the last wave of this one ends, so each launch ends with SIMDs
holding fewer waves. STREAMS=2 runs two independent lane sets on two streams,
launched alternately: each set's queued launch fills the CUs as the other's
waves retire. Costs a second scratch area (H/2+4 elements per lane).

probe21 (CKPT2, H=2048, 3 rounds x 8 s): one stream 15.281, **two streams
15.911 G/s (+4.1%)**; PROFILE builds 15.229 at ~1240 MHz vs 15.826 at
~1195 MHz. Work per clock +8%: the launch-boundary drain was costing that
much (the "91% VALU utilization" estimate was too high; the SIMDs now run
near 98% by the same count). The clock drops because the SIMDs are busier:
the card is power limited, so energy per candidate is the currency now.

probe22 (two streams): base 15.953, BLOCK=64 15.914, H=4096 15.985, CKPT1
15.337 G/s. Block size and H do not matter; CKPT2 still beats CKPT1 by 4%.

Filter trims (static, per pair, main path): the leaves start column 8's
multiply-add chain from column 7's carry (no 64-bit add), and take A+ and
A- = alpha - beta in signed limbs (v_mad_i64_i32; no 128p bias for A-, so 9
fewer adds). Signed columns make V and Q possibly negative, so the "unsure"
window is two-sided (N mod 2^23 within 128 of 0 or 2^23, about 2^-15 per
leaf). 545 -> 534 VALU per pair section. With only the A- leaf signed, LLVM
spilled the scratch offset and reloaded it (with a vmcnt(0) wait) at the top
of every backward iteration; both leaves signed avoids that.

fold_top: written as a 64-bit expression, t += (1216 hi) << 32 made LLVM
emit a v_mad_u64_u32 with a zero multiplier (64 of them in the kernel). Now
fold_top takes column 8's accumulator and adds 9728 hi + 1216 (lo >> 29) with
two multiply-adds: 7 fewer VALU per pair (forward 263 -> 259 per two steps,
backward 1678 -> 1659).

probe24 (5 rounds x 8 s): signed leaves 16.005, + first fold_top fix 16.016,
+ H=4096 16.095 G/s (spread 1.4-2.3%).

Register pressure is at the edge: every trim shifts LLVM's allocation, and
twice the scratch offset VGPR got spilled and reloaded (with vmcnt(0)) at the
top of each backward iteration. With one wave per block (BLOCK=64) the wave's
part of the scratch address moves into the scalar resource base and 4t is both
the scratch voffset and the LDS address: one VGPR for all addressing, no
reload in the loop (SGPR spills instead, eight lane moves per batch).

probe25 (4 rounds x 8 s): signed leaves 16.030, + fold_top 16.055, + one
addressing VGPR (BLOCK=64) 16.062, + H=4096 16.160 G/s. H=4096 is +0.6% in
both probes (the per-batch inversion, ~29k VALU, is spread over twice the
pairs); default from here. Scratch: (H/2 + 4) elements of 36 bytes per lane,
7.9 GB per lane set, 15.7 GB for the two.

probe26 (H=4096): 16.242 G/s; streaming cache hints on the scratch traffic
(stores slc, loads glc+slc or glc) 16.246 / 16.217: no effect, dropped.

v0.6 candidate (signed filter, fold_top, BLOCK=64 addressing, H=4096)
conformance check: all pass, grind 16.34 G/s, long targets 15.73 G/s.

## State at v0.6 (static counts, gfx90a, H=4096)

Per two steps (two candidate pairs): backward loop 1660 VALU (1347 MAD),
forward loop 259 (198 MAD), so ~960 VALU per pair (~777 v_mad_u64/i64_32);
per-batch work (inversion: 254 squarings + 11 multiplies, giant step, center)
adds ~7 per pair. With two streams the SIMDs issue VALU work ~98% of cycles
(PROFILE clock times VALU count), so the rate is clock x 1/VALU-per-pair; the
clock (~1.2 GHz) is set by power, of which the 36 bytes per pair of scratch
traffic costs ~5%.

Ideas weighed and not pursued (reasons in the entries above): MFMA (FP64
matrix) inner products (per-lane bignums would need cross-lane transposes of
every output), table constants pre-shifted by 2^(29i) to drop the folds
(-32 MAD per pair, but 9x the SGPR constants per step and SMEM returns out of
order, so they cannot be prefetched), CKPT=4 (tree spills at 4 waves; the
recompute variant costs +6% VALU for ~2.5% clock), pipelined batches (3
waves), four candidates per inversion element (regroups the same products).

probe27 (5 x 8 s): v0.6 16.224, backward loop unrolled by two (no prefetch
copies, 4.5 fewer VALU per pair) 16.101 G/s: slower, likely the doubled loop
body (~25 KB) in the instruction cache. Dropped.

probe28: base 16.125, scratch as [wave][j][9][64] (each wave's elements
contiguous, for TLB locality) 16.091, the same unrolled 16.002 G/s. Page
locality is not a cost; layout kept.

Fast filter for up to four targets of any lengths: prefix masks nest, so all
are pre-tested under the shortest one's 32-bit mask (never rejects a match;
longer targets' near-misses pass at the shorter length's rate and the exact
test settles them). Before, any length mix fell back to the general path (a
loop over constant memory). --verify gains a mixed-length random window
(two 1- and two 2-character targets). Check: 9 self-checks pass, grind 16.30
G/s. The rig's long-targets check sends six targets (four short, two long), so
it still takes the general path (15.71 G/s over 4 s).

General target form (more than four targets): the per-target loop over
constant memory (about three instructions per target per candidate, so 256
targets would roughly halve the rate) is replaced by a branchless binary
search over the sorted masked values in LDS (1 KB per block): log2(size)
dependent LDS reads. --verify adds a 24-target window (2 and 3 characters).
Check: 10 self-checks pass; long-targets (six targets) 15.58 vs grind 16.18
G/s, the same ratio as the loop at six targets.

probe29 (64 targets of 7 characters, self-timed --bench): per-target loop
9.342, LDS binary search 15.382 G/s (+65%; ~5% below the four-target fast
form).

## Where it stands (2026-10-17)

Official: v0.6 16.091 G/s (11.1x the baseline). The hot loops are at ~960
VALU per candidate pair with the SIMDs ~98% busy, and the clock is set by
power, so further gains need fewer instructions per pair or fewer bytes of
scratch. What is left is small: H=8192 (inversion amortization, ~0.3%, but 31
GB of scratch), a divstep (Bernstein-Yang) inversion instead of Fermat
(~0.3-0.5%), a narrower unsure window (+-32 is provably enough, ~0.1%). The
host side was also made sturdier after v0.6 without touching the benchmark
path: any four targets use the fast filter, larger sets a binary search in
LDS, and a GPU without room for the second lane set runs one.

## Round-1 merge: onion-walk-combined 0.7 (2026-10-17)

Base: this entry as of b03ce98 (v0.6 kernel, 16.091 G/s official). The four
round-1 entries were reviewed side by side (`results/arena-round1-mi210/review/`
in the onionrig repo). Gemini 4 Argon's entry had found the same free
denominator product, but normalized its numerators so one term is a table
constant: y1 (a_j +- u1 b_j) / (c_j -+ w1). Applied here by letting the
backward inverses carry mu = 1/x1:
y(C +- T_j) = (alpha' +- beta') mu/q_j with alpha' = (x1 y1)(r s) - lam y2 and
beta' = y1^2 r - (y2 s). beta' is one product plus an addend (`mul_add_k`, the
addend joining each column as one `v_mad_u64_u32` times 1), so the numerators
take three limb products instead of four. The chain starts at W mu and the
inverse is multiplied by mu (two multiplies per batch), so the backward pass
still ends at exactly 1/W for the giant step. The table is unchanged. The
center is now tested before the loops, so y1 holds no registers through them.

Other round-1 ideas checked and not needed: GPT's launch sizing (lanes at an
exact multiple of resident capacity) is what this entry already does via
hipOccupancyMaxActiveBlocksPerMultiprocessor; GPT's and Qwen's designs are
memory-bound (128 B per candidate) and fix problems this design doesn't have.

Static ISA (gfx90a, ROCm 7.2.4, H = 4096), backward loop main path per two
pairs: v0.6 1660 VALU (1347 MAD) -> 0.7 1516 VALU (1203 MAD), -72 VALU per pair
(81 MADs removed, 9 one-instruction adds added): ~888 VALU per pair instead of
~960 (-7.5%). VGPR spills 6 -> 2, scratch 28 -> 12 bytes per lane, 4 waves/SIMD.
CPU tests: fe29_test adds mul_add at the limb bounds (600,305 checks);
walk_test unchanged and passing, and fails under two deliberate mutations
(wrong addend, dropped mu).

MI210 measurements of 0.7 (Sol, tools/sol/probe.sbatch; interleaved rounds
of 8 s, 4 targets x 5 characters, self-timed, ~0.5% above the rig):

| Probe | Variants (G/s, spread over rounds) |
|---|---|
| probe 1 (5 rounds; the conformance check ran on the node's other MI210 meanwhile) | v0.6 16.175 (2.8%), 0.7 17.234 (5.5%); PROFILE clocks equal (~1170-1190 MHz) |
| probe 2 (6 rounds, node to itself) | v0.6 16.144 (2.4%), **0.7 17.340 (1.8%)**, 0.7 H=8192 17.360 (1.8%), 0.7 CKPT=1 16.588 (2.0%) |

0.7 is +7.4% over v0.6, the instruction cut almost exactly, at the same
clock. H=8192 is within noise of 4096 (kept 4096, half the scratch), and
CKPT=2 still beats CKPT=1 by 4.5%. Lesson from probe 1: jobs on sg250's two
MI210s slow each other (rounds swung 20-34% while the check ran, and the
check's own rates read 7 G/s), so measure with the node to yourself.

Official bench (judge.sbatch, rig 0.1.0, plan 4 targets x 5 characters after
a 17.603 G/s calibration): **17.294 G/s (95% CI 17.265-17.322)**, 3 runs of
60 s at ~1360 MHz reported sclk, 289 W, 84-86 C; 124,199 hits flagged, all
verified, no degenerate batches. Conformance check: all pass.

## One asm statement per chain: F29_CHAIN_ASM (2026-10-17, 0.7.1)

On gfx940 and later, LLVM's hazard recognizer assumes any inline asm may hide
a dst-forwarding (OPSEL/SDWA) hazard. It puts an `s_nop 0` after a statement
whenever the next instruction reads or overwrites a register the statement
wrote. Every multiply-add here was its own statement, and each writes the
same dead carry-out pair, so nearly every one got a nop. gfx942 and gfx950
compile to identical hot loops, with 163 + 1,053 `s_nop` per two pairs on
259 + 1,516 VALU. The hardware needs none of them: `v_mad_u64_u32` uses
neither OPSEL nor SDWA (CDNA4 ISA section 4.5, table 11).

fe29.h now forms every column of a product as one chain (`column`, `columni`,
`sqr_column`, `folds`), and mul, mul_add and ip share one body (`mul_core`).
By default a chain is still one statement per multiply-add. The default
build's hot loops have the same instruction mix as 0.7's, with the same count
per opcode and a slightly different order (one nop fewer on gfx942/gfx950).
`-DF29_CHAIN_ASM=1` makes each chain one statement (fe29_chains.h).

Static counts per two pairs (ROCm 7.2.4):

| gfx942 / gfx950 | VALU fwd + bwd | s_nop fwd + bwd | VGPRs, VGPR spills |
|---|---|---|---|
| default | 259 + 1,516 | 163 + 1,053 | 128, 10 |
| F29_CHAIN_ASM=1 | 261 + 1,525 | 22 + 174 | 127, 10 |

On gfx90a neither build has nops. The chain build spills 8 VGPRs instead of 2,
all outside the loops.

**MI210 results** (gfx90a, where there are no nops to remove):
- The chain build passed the rig's conformance check (judge.sbatch, job 65960515).
- Probe 3 ran 6 rounds of 20 s in a fixed order: 0.7, then the restructured
  default, then chain. Means: 17.196, 17.144 and 17.108 G/s.
- The GPU warmed from 17.5 to 17.0 G/s over the job, so later slots read low.
  By the last round the gaps to 0.7 were -0.06% and -0.15%, about the warm-up
  drift per slot.
- So the restructure costs nothing measurable, and chaining costs at most
  about 0.2% on a GPU it cannot help.

Whether 1,020 fewer nops pay for 11 more VALU on CDNA3/4 is a question for the
rental sweep (tools/rental, build `chain`). That sweep rotates the order every
round, pairs each build with the base of the same round, and warms the GPU
first.
