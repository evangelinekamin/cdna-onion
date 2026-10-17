# onion-walk-combined

A Tor v3 vanity-address grinder for AMD CDNA GPUs, implementing the
`onionrig-grind/1` contract (`docs/design/CONTRACT.md`). It merges the best of
the four round-1 arena entries: Claude Opus 5.5's onion-walk 0.6 (16.09 G/s on
the MI210, the round's best) is the base, with its pair numerators normalized
the way Gemini 4 Argon's entry normalized its own, which saves one of the four
limb products per pair. Written by Claude Opus 5.5. MIT licensed (see
`LICENSE`).

    make entry GPU_ARCH=gfx90a     # the arena build (gfx950 also compiles)
    make test                      # CPU tests: reference math, GPU field code, walk emulation
    ./entry --info | --verify VECTORS_DIR | --grind

## How it works

Every GPU lane walks its own sequence of *centers* C = A0 + k*8B. One batch
tests C and the 2H points C +- j*8B (j = 1..H, H = 4096) with a single field
inversion, then steps to the next center.

**Algebra (src/walk.h).** The dual addition law gives y(C +- T) for T = j*8B
with two denominators x1*y2 -+ y1*x2. Dividing through by x1*x2 turns them into
s_j -+ lam (s = y2/x2 from a table, lam = y1/x1 once per batch), whose product
q_j = s_j^2 - lam^2 needs no multiplication. Both candidates of the pair are
then (alpha_j +- beta_j) / q_j, numerators with four terms between them, each a
per-batch value times a table constant. The inverses carry a factor mu = 1/x1
(the forward chain starts at W mu and the inverse is multiplied by mu, two
multiplies per batch), which turns one term into a pure table constant:
alpha' = (x1 y1)(r s) - lam y2 is a two-term inner product and
beta' = y1^2 r - (y2 s) one product plus an addend, three limb products
instead of four. Montgomery's trick over the q_j needs one stored field element
per *pair* (not two per candidate), and the walk keeps only every other one
(see Memory). Per pair of candidates: one forward multiply, two and a half
backward, three products for the numerators, two partial products for the
filter.

**Filter (src/fe29.h, `leaf32`).** The last multiply only needs y's low bits.
Forming the low columns (mod 2^32) and the top columns of the product (35 limb
products of 81) gives y's low 32 bits exactly, except when the top columns
cannot settle the reduction quotient (about 1 in 2^15), which the function
reports. Those candidates, and the rare ones passing the 32-bit test, get the
full multiply. The test is exact for targets up to 6 characters and a
pre-filter for longer ones; it never rejects a match. Its second operand is in
signed limbs (`v_mad_i64_i32`), so alpha - beta needs no bias added. Targets
are compared under the shortest target's mask (prefix masks nest): up to four
values in SGPRs, or for larger sets a branchless binary search over the sorted
values in LDS (eight dependent reads for 256 targets instead of 256 compares).

**Field (src/fe29.h).** Nine 29-bit limbs: a limb product is below 2^58, so a
column of 9 (or 18, for inner products) products accumulates carry-free in one
64-bit register: each product is one `v_mad_u64_u32`. The reduction folds the
high columns with 2^261 = 1216 (mod p) and carries sequentially, each carry
seeding the next column's accumulator. The multiply-adds are inline asm, so LLVM
cannot reassociate the carry to the end of a column (which costs a 64-bit add).
Table constants are wave-uniform and enter as SGPR operands. Each column is
one chain of multiply-adds; `-DF29_CHAIN_ASM=1` emits a chain as one asm
statement instead of one per multiply-add, which on gfx940 and later saves the
`s_nop` LLVM inserts after each statement (`NOTES.md`).

**Memory.** Montgomery's trick stores a prefix product per step, read back in
reverse; on the MI210 that HBM traffic costs about 20% at full volume. The
walk stores only every other prefix product (`CKPT=2`) and rebuilds the
inverses of a step pair from it: A = P_{j-2}/P_j gives 1/q_j = A q_{j-1} and
1/q_{j-1} = A q_j, five multiplies per two steps instead of four, for half the
traffic. The traffic costs clock rather than stalls: the card is power
limited, and halving the bytes raised the effective shader clock from ~1170 to
~1240 MHz. The scratch layout makes every access a coalesced 256-byte row per
wave, with the wave's part of the address in the scalar buffer resource and the
limb offsets in the instructions, so one VGPR (4 x lane) serves all scratch and
LDS addressing.

**Launches.** Two independent lane sets, each sized to fill the GPU, run on
two streams and are launched alternately. Within a launch the SIMDs favor
older waves, so waves finish at different times; the other set's queued launch
fills the CUs as they retire instead of the GPU draining at every launch
boundary (+4%).

**Trust.** Every offset the GPU flags is re-derived on the host with an
independent radix-2^51 implementation (src/ref25519.h) before it is reported.
`--verify` checks the GPU field code against the shared vectors, the known-answer
grinds, and four randomized windows (short, mixed-length, and 24 targets)
against an exhaustive host computation.

## Performance (AMD Instinct MI210, the arena's official benchmark)

| Version | Rate | vs baseline (1.45 G/s) |
|---|---|---|
| 0.1 | 13.137 G/s | 9.1x |
| 0.3 | 14.250 G/s | 9.8x |
| 0.5 | 15.834 G/s | 10.9x |
| 0.6 | 16.091 G/s | 11.1x |
| 0.7 (combined) | **17.294 G/s** (95% CI 17.265-17.322) | 11.9x |

On an AMD Instinct MI350X (CDNA4, gfx950, 256 CUs, 1,000 W), 0.7.1 benchmarks
at **54.78 G/s** (95% CI 54.12-55.45, the rig's proof-of-work rate, 4 runs).
On an 8x AMD Instinct MI355X node (1,400 W each) a real search ran at
**575 G/s** (about 72 G/s per GPU), finding three 10-character `evangeline`
addresses in 84 minutes.

Versions 0.1-0.6 are Claude's round-1 entry; 0.7 is the merge. Per pair of
candidates the hot loops now execute about 888 VALU instructions (about 700 of
them 32x32+64 multiply-adds) at 4 waves per SIMD, down from about 960, and the
SIMDs issue VALU work nearly every cycle; the card is power limited (about
290 W of its 300 W) and holds an effective ~1.2 GHz shader clock under this
load. `NOTES.md` has the measurement log.

## Credits and inspiration

Ideas, not code: the numerator normalization from Gemini 4 Argon's round-1
entry (its pair-factored law leaves one table constant per numerator); the
dual addition law and y-only testing as described in the honion
documentation; walking by 8B with Montgomery batch inversion (mkp224o);
the split-key offset scheme (AlexanderYastrebov/onion-vanity-address, honion);
Hisil, Wong, Carter and Dawson's extended twisted Edwards formulas;
Montgomery's simultaneous-inversion trick. The protocol harness structure
follows the repository's baseline entry (onion-gpu port, MIT).
