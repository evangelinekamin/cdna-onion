// GF(2^255 - 19) for the GPU hot loop: nine 29-bit limbs in 32-bit registers.
//
// Why 29 bits: a limb product is below 2^58, so a column of nine products (or
// eighteen, for the two-term inner products the walk needs) accumulates in one
// 64-bit register with no carry handling at all. Every product is a single
// v_mad_u64_u32 on CDNA: 81 per multiply, with no add-with-carry chains. The
// price is the reduction, which turns 17 64-bit columns into 29-bit limbs; it
// folds the high half with 2^261 = 1216 (mod p) and carries sequentially, using
// each column's carry as the starting accumulator of the next column so the
// carry add is free.
//
// Limb bounds (all functions here are written against these):
//   tight: every limb < 2^29 + 2^17. Output of mul/sqr/ip, and canon.
//   loose: every limb < 2^30.6.     Output of add/sub on tight inputs.
//   signed: |limb| < 2^30 + 2^18, two's complement. Output of dif, and add of
//           tight inputs read as signed: the second operand of leaf32.
//   mul needs one tight operand and one loose operand (or tighter); mul_add
//   the same plus a tight addend; ip needs all four operands tight; leaf32 one
//   tight and one signed.
// Every function compiles for both host and device, so the CPU tests exercise
// exactly the arithmetic the GPU runs.
#pragma once
#include <cstdint>

// F29_CHAIN_ASM=1: one inline-asm statement per multiply-add chain (see chain
// below); the default is one per multiply-add.
#ifndef F29_CHAIN_ASM
#define F29_CHAIN_ASM 0
#endif
#if F29_CHAIN_ASM && defined(__HIP_DEVICE_COMPILE__) && !defined(F29_NO_ASM)
#define F29_CHAINED 1
#include "fe29_chains.h"
#else
#define F29_CHAINED 0
#endif

#if defined(__HIPCC__)
#define HD __host__ __device__ __forceinline__
#else
#define HD inline
#endif

namespace f29 {

static constexpr uint32_t M29 = (1u << 29) - 1;
static constexpr uint64_t F261 = 1216;      // 2^261 mod p = 2^6 * 19
static constexpr uint64_t F264 = 9728;      // 2^264 mod p (hi word of a column, one limb up)
// 128p in this radix, for subtraction: limb 0 = 2^30 - 2432, limbs 1..8 = 2^30 - 2.
static constexpr uint32_t Z0 = (1u << 30) - 2432;
static constexpr uint32_t ZI = (1u << 30) - 2;

struct fe {
    uint32_t l[9];
};

// One 32x32+64 multiply-add: v_mad_u64_u32. On the device it is inline asm,
// which keeps each column's accumulation in the order written: LLVM would
// otherwise reassociate a column so the incoming carry is added last, with a
// separate 64-bit add (two instructions per limb per multiply). The asm stays
// schedulable; only its order within a dependency chain is fixed. mads takes
// its second operand in an SGPR (table constants and the fold factors, which
// are wave-uniform); mul0 has no addend.
#if defined(__HIP_DEVICE_COMPILE__) && !defined(F29_NO_ASM)
__device__ __forceinline__ uint64_t mad(uint32_t a, uint32_t b, uint64_t c) {
    uint64_t r, dead;
    asm("v_mad_u64_u32 %0, %1, %2, %3, %4" : "=v"(r), "=s"(dead) : "v"(a), "v"(b), "v"(c));
    return r;
}
__device__ __forceinline__ uint64_t mads(uint32_t a, uint32_t b, uint64_t c) {
    uint64_t r, dead;
    asm("v_mad_u64_u32 %0, %1, %2, %3, %4" : "=v"(r), "=s"(dead) : "v"(a), "s"(b), "v"(c));
    return r;
}
__device__ __forceinline__ uint64_t mul0(uint32_t a, uint32_t b) {
    uint64_t r, dead;
    asm("v_mad_u64_u32 %0, %1, %2, %3, 0" : "=v"(r), "=s"(dead) : "v"(a), "v"(b));
    return r;
}
__device__ __forceinline__ uint64_t mul0s(uint32_t a, uint32_t b) {
    uint64_t r, dead;
    asm("v_mad_u64_u32 %0, %1, %2, %3, 0" : "=v"(r), "=s"(dead) : "v"(a), "s"(b));
    return r;
}
// Signed forms (v_mad_i64_i32) for leaf32: a is below 2^31, so it reads the
// same as a signed operand.
__device__ __forceinline__ int64_t madi(uint32_t a, int32_t b, int64_t c) {
    int64_t r;
    uint64_t dead;
    asm("v_mad_i64_i32 %0, %1, %2, %3, %4" : "=v"(r), "=s"(dead) : "v"(a), "v"(b), "v"(c));
    return r;
}
__device__ __forceinline__ int64_t madis(int32_t a, uint32_t b, int64_t c) {
    int64_t r;
    uint64_t dead;
    asm("v_mad_i64_i32 %0, %1, %2, %3, %4" : "=v"(r), "=s"(dead) : "v"(a), "s"(b), "v"(c));
    return r;
}
__device__ __forceinline__ int64_t mul0i(uint32_t a, int32_t b) {
    int64_t r;
    uint64_t dead;
    asm("v_mad_i64_i32 %0, %1, %2, %3, 0" : "=v"(r), "=s"(dead) : "v"(a), "v"(b));
    return r;
}
// acc + c as c * 1 + acc: one v_mad_u64_u32 instead of an add and an
// add-with-carry. adds takes c in an SGPR (the 1 is an inline constant, so it
// stays within the one-SGPR constant bus limit).
__device__ __forceinline__ uint64_t addv(uint32_t c, uint64_t acc) {
    uint64_t r, dead;
    asm("v_mad_u64_u32 %0, %1, %2, 1, %3" : "=v"(r), "=s"(dead) : "v"(c), "v"(acc));
    return r;
}
__device__ __forceinline__ uint64_t adds(uint32_t c, uint64_t acc) {
    uint64_t r, dead;
    asm("v_mad_u64_u32 %0, %1, %2, 1, %3" : "=v"(r), "=s"(dead) : "s"(c), "v"(acc));
    return r;
}
#else
HD uint64_t mad(uint32_t a, uint32_t b, uint64_t c) { return (uint64_t)a * b + c; }
HD uint64_t mads(uint32_t a, uint32_t b, uint64_t c) { return (uint64_t)a * b + c; }
HD uint64_t mul0(uint32_t a, uint32_t b) { return (uint64_t)a * b; }
HD uint64_t mul0s(uint32_t a, uint32_t b) { return (uint64_t)a * b; }
HD int64_t madi(uint32_t a, int32_t b, int64_t c) { return (int64_t)(int32_t)a * b + c; }
HD int64_t madis(int32_t a, uint32_t b, int64_t c) { return (int64_t)a * (int32_t)b + c; }
HD int64_t mul0i(uint32_t a, int32_t b) { return (int64_t)(int32_t)a * b; }
HD uint64_t addv(uint32_t c, uint64_t acc) { return acc + c; }
HD uint64_t adds(uint32_t c, uint64_t acc) { return acc + c; }
#endif

// Operand kind as a template switch: S = the second operand is wave-uniform.
template <bool S> HD uint64_t madk(uint32_t a, uint32_t b, uint64_t c) { return S ? mads(a, b, c) : mad(a, b, c); }
template <bool S> HD uint64_t mul0k(uint32_t a, uint32_t b) { return S ? mul0s(a, b) : mul0(a, b); }
template <bool S> HD uint64_t addk(uint32_t c, uint64_t acc) { return S ? adds(c, acc) : addv(c, acc); }

// Chains: acc + a_0 b_0 + ... + a_{n-1} b_{n-1}, added in that order (chain0
// starts from zero; chaini and chaini0 are the signed forms). Every column of
// a product is one chain. By default each multiply-add in it is its own asm
// statement. F29_CHAIN_ASM=1 makes the whole chain one statement
// (fe29_chains.h), for gfx940 and later: there LLVM assumes any inline asm may
// hide a dst-forwarding hazard and puts an s_nop after a statement whenever the
// next instruction touches a register it wrote, which with one multiply-add
// per statement means about 0.7 s_nop per VALU instruction in the hot loops
// (gfx90a needs none). One statement per chain removes most of them, but it
// also stops the scheduler from interleaving independent chains. n is a
// compile-time constant once the callers' loops are unrolled.
template <bool S>
HD uint64_t chain(const uint32_t *a, const uint32_t *b, int n, uint64_t acc) {
#if F29_CHAINED
    return chain_asm<S, false>(a, b, n, acc);
#else
#pragma unroll
    for (int t = 0; t < n; t++) acc = madk<S>(a[t], b[t], acc);
    return acc;
#endif
}

template <bool S>
HD uint64_t chain0(const uint32_t *a, const uint32_t *b, int n) {
#if F29_CHAINED
    return chain_asm<S, true>(a, b, n, 0);
#else
    uint64_t acc = 0;
#pragma unroll
    for (int t = 0; t < n; t++) acc = t == 0 ? mul0k<S>(a[t], b[t]) : madk<S>(a[t], b[t], acc);
    return acc;
#endif
}

HD int64_t chaini(const uint32_t *a, const int32_t *b, int n, int64_t acc) {
#if F29_CHAINED
    return chaini_asm<false>(a, b, n, acc);
#else
#pragma unroll
    for (int t = 0; t < n; t++) acc = madi(a[t], b[t], acc);
    return acc;
#endif
}

HD int64_t chaini0(const uint32_t *a, const int32_t *b, int n) {
#if F29_CHAINED
    return chaini_asm<true>(a, b, n, 0);
#else
    int64_t acc = 0;
#pragma unroll
    for (int t = 0; t < n; t++) acc = t == 0 ? mul0i(a[t], b[t]) : madi(a[t], b[t], acc);
    return acc;
#endif
}

// Column c (0..16) of a product: the limb products x_i y_{c-i} for every i
// with both limbs in range, in increasing i, as one chain (from zero, or added
// to acc). With NP = 2 (inner products) each i contributes x[0]_i y[0]_{c-i}
// and then x[1]_i y[1]_{c-i}.
template <bool S, int NP>
HD uint64_t column(const uint32_t (&x)[NP][9], const uint32_t (&y)[NP][9], int c, bool zero,
                   uint64_t acc) {
    uint32_t ta[9 * NP], tb[9 * NP];
    int n = 0;
#pragma unroll
    for (int i = c > 8 ? c - 8 : 0; i <= (c < 8 ? c : 8); i++) {
#pragma unroll
        for (int p = 0; p < NP; p++) {
            ta[n] = x[p][i];
            tb[n] = y[p][c - i];
            n++;
        }
    }
    return zero ? chain0<S>(ta, tb, n) : chain<S>(ta, tb, n, acc);
}

// column's signed form (one product), for leaf32.
HD int64_t columni(const uint32_t (&x)[9], const int32_t (&y)[9], int c, bool zero, int64_t acc) {
    uint32_t ta[9];
    int32_t tb[9];
    int n = 0;
#pragma unroll
    for (int i = c > 8 ? c - 8 : 0; i <= (c < 8 ? c : 8); i++) {
        ta[n] = x[i];
        tb[n] = y[c - i];
        n++;
    }
    return zero ? chaini0(ta, tb, n) : chaini(ta, tb, n, acc);
}

// Column c of a square, with x2 = 2x: the cross terms x_i x2_{c-i} for
// i < c - i, then x_{c/2}^2 for even c, as one chain.
HD uint64_t sqr_column(const uint32_t (&x)[9], const uint32_t (&x2)[9], int c, bool zero,
                       uint64_t acc) {
    uint32_t ta[5], tb[5];
    int n = 0;
#pragma unroll
    for (int i = c > 8 ? c - 8 : 0; 2 * i < c; i++) {
        ta[n] = x[i];
        tb[n] = x2[c - i];
        n++;
    }
    if ((c & 1) == 0) {
        ta[n] = x[c / 2];
        tb[n] = x[c / 2];
        n++;
    }
    return zero ? chain0<false>(ta, tb, n) : chain<false>(ta, tb, n, acc);
}

// The carry out of a column: acc >> 29, a 64-bit value below 2^35 (one
// v_lshrrev_b64, which issues at full rate: two 32-bit instructions measured
// slower).
HD uint64_t carry29(uint64_t acc) { return acc >> 29; }

// Low column m's folds, as one chain: 1216 lo(h) for m <= 7, with h the high
// column m+9, and 9728 hi(hprev) for m >= 1, with hprev the high column m+8.
HD uint64_t folds(uint64_t acc, uint64_t h, uint64_t hprev, int m) {
    uint32_t a[2], b[2];
    int n = 0;
    if (m <= 7) {
        a[n] = (uint32_t)h;
        b[n] = (uint32_t)F261;
        n++;
    }
    if (m >= 1) {
        a[n] = (uint32_t)(hprev >> 32);
        b[n] = (uint32_t)F264;
        n++;
    }
    return chain<true>(a, b, n, acc);
}

// The end of a reduction: limb 8 from column 8's accumulator acc (< 2^63),
// and the carry out of it (bits 261 and up), folded into limbs 0 and 1:
// 1216 (acc >> 29) = 9728 hi + 1216 (lo >> 29) for acc = hi 2^32 + lo, two
// multiply-adds without forming the 64-bit carry.
HD void fold_top(fe &r, uint64_t acc) {
    const uint32_t lo = (uint32_t)acc, hi = (uint32_t)(acc >> 32);
    r.l[8] = lo & M29;
    const uint32_t a[2] = {hi, lo >> 29}, b[2] = {(uint32_t)F264, (uint32_t)F261};
    const uint64_t t = chain<true>(a, b, 2, (uint64_t)r.l[0]);  // t < 2^45
    r.l[0] = (uint32_t)t & M29;
    r.l[1] += (uint32_t)(t >> 29);
}

// r = x[0] y[0] + ... + x[NP-1] y[NP-1] (+ z when ADD), reduced once: the
// common body of mul, mul_add and ip below. S: every y (and z) is wave-uniform
// (SGPR operands).
template <bool S, int NP, bool ADD>
HD void mul_core(fe &r, const uint32_t (&x)[NP][9], const uint32_t (&y)[NP][9],
                 const uint32_t *z) {
    // Streaming order: high column c_{m+9} is formed just before low column m
    // folds it in, so only two high columns are ever live (register pressure).
    uint64_t hprev = 0, acc = 0;
#pragma unroll
    for (int m = 0; m <= 8; m++) {
        const uint64_t h = m <= 7 ? column<S, NP>(x, y, m + 9, true, 0) : 0;
        acc = column<S, NP>(x, y, m, m == 0, acc);
        if constexpr (ADD) acc = addk<S>(z[m], acc);
        acc = folds(acc, h, hprev, m);
        if (m == 8) break;
        r.l[m] = (uint32_t)acc & M29;
        acc = carry29(acc);
        hprev = h;
    }
    fold_top(r, acc);
}

// r = a * b. a tight, b loose (or both tight). r tight. r may alias a or b.
// S: b is wave-uniform (SGPR operands).
template <bool S>
HD void mul_k(fe &r, const fe &a, const fe &b) {
    uint32_t x[1][9], y[1][9];
#pragma unroll
    for (int i = 0; i < 9; i++) { x[0][i] = a.l[i]; y[0][i] = b.l[i]; }
    mul_core<S, 1, false>(r, x, y, nullptr);
}

HD void mul(fe &r, const fe &a, const fe &b) { mul_k<false>(r, a, b); }

// r = a*b + c, reduced once. a tight, b loose (or tight), c tight. r tight.
// r may alias any input. S: b and c are wave-uniform (SGPR operands). The
// addend joins each low column's accumulator before the reduction: nine
// one-instruction adds where a second limb product would cost 81 (the walk's
// beta, whose other term became a table constant). Column bound: nine
// tight-by-loose products plus the carry, folds and addend stay below 2^63.
template <bool S>
HD void mul_add_k(fe &r, const fe &a, const fe &b, const fe &c) {
    uint32_t x[1][9], y[1][9], z[9];
#pragma unroll
    for (int i = 0; i < 9; i++) { x[0][i] = a.l[i]; y[0][i] = b.l[i]; z[i] = c.l[i]; }
    mul_core<S, 1, true>(r, x, y, z);
}

HD void mul_add(fe &r, const fe &a, const fe &b, const fe &c) { mul_add_k<false>(r, a, b, c); }

// r = a^2. a tight. r tight. r may alias a.
HD void sqr(fe &r, const fe &a) {
    uint32_t x[9], x2[9];
#pragma unroll
    for (int i = 0; i < 9; i++) { x[i] = a.l[i]; x2[i] = a.l[i] << 1; }
    uint64_t h[8];
#pragma unroll
    for (int k = 9; k <= 16; k++) h[k - 9] = sqr_column(x, x2, k, true, 0);
    uint64_t acc = 0;
#pragma unroll
    for (int m = 0; m <= 8; m++) {
        acc = sqr_column(x, x2, m, m == 0, acc);
        acc = folds(acc, m <= 7 ? h[m] : 0, m >= 1 ? h[m - 1] : 0, m);
        if (m == 8) break;
        r.l[m] = (uint32_t)acc & M29;
        acc = carry29(acc);
    }
    fold_top(r, acc);
}

// r = a*b + c*d, reduced once. All inputs tight. r tight. r may alias any input.
// S: b and d are wave-uniform (SGPR operands).
template <bool S>
HD void ip_k(fe &r, const fe &a, const fe &b, const fe &c, const fe &d) {
    uint32_t x[2][9], y[2][9];
#pragma unroll
    for (int i = 0; i < 9; i++) {
        x[0][i] = a.l[i];
        y[0][i] = b.l[i];
        x[1][i] = c.l[i];
        y[1][i] = d.l[i];
    }
    mul_core<S, 2, false>(r, x, y, nullptr);
}

HD void ip(fe &r, const fe &a, const fe &b, const fe &c, const fe &d) { ip_k<false>(r, a, b, c, d); }

// r = a + b (no carry). Tight inputs give r < 2^30.01 per limb.
HD void add(fe &r, const fe &a, const fe &b) {
#pragma unroll
    for (int i = 0; i < 9; i++) r.l[i] = a.l[i] + b.l[i];
}

// r = a - b + 128p. b tight (limbs <= 2^30 - 2432). r loose.
HD void sub(fe &r, const fe &a, const fe &b) {
    r.l[0] = a.l[0] + (Z0 - b.l[0]);
#pragma unroll
    for (int i = 1; i < 9; i++) r.l[i] = a.l[i] + (ZI - b.l[i]);
}

// r = 128p - b. b tight. r loose (< 2^30).
HD void neg(fe &r, const fe &b) {
    r.l[0] = Z0 - b.l[0];
#pragma unroll
    for (int i = 1; i < 9; i++) r.l[i] = ZI - b.l[i];
}

// Canonical form: the value mod p, limbs < 2^29, value < p. Any limbs < 2^32.
HD void canon(fe &r, const fe &a) {
    uint32_t t[9];
    uint64_t acc = 0;
#pragma unroll
    for (int i = 0; i < 9; i++) {
        acc += a.l[i];
        t[i] = (uint32_t)acc & M29;
        acc = carry29(acc);
    }
    // Bits from 255 up: limb 8 from bit 23, plus the overflow past 2^261.
    acc = 19 * ((uint64_t)(t[8] >> 23) + (acc << 6));
    t[8] &= (1u << 23) - 1;
#pragma unroll
    for (int i = 0; i < 9; i++) {
        acc += t[i];
        t[i] = (uint32_t)acc & M29;
        acc = carry29(acc);
    }
    // Now below 2^255 + 2^12; one more fold leaves it below 2^255.
    acc = 19 * (uint64_t)(t[8] >> 23);
    t[8] &= (1u << 23) - 1;
#pragma unroll
    for (int i = 0; i < 9; i++) {
        acc += t[i];
        t[i] = (uint32_t)acc & M29;
        acc = carry29(acc);
    }
    // Subtract p when value >= p, i.e. when value + 19 reaches 2^255.
    uint32_t u[9];
    acc = 19;
#pragma unroll
    for (int i = 0; i < 9; i++) {
        acc += t[i];
        u[i] = (uint32_t)acc & M29;
        acc = carry29(acc);
    }
    const uint32_t ge_p = u[8] >> 23;
    u[8] &= (1u << 23) - 1;
#pragma unroll
    for (int i = 0; i < 9; i++) r.l[i] = ge_p ? u[i] : t[i];
}

// Low 64 bits of a canonical element (what the address prefix is made of).
HD uint64_t low64(const fe &c) {
    return (uint64_t)c.l[0] | ((uint64_t)c.l[1] << 29) | ((uint64_t)c.l[2] << 58);
}

HD bool is_zero_canon(const fe &c) {
    uint32_t acc = 0;
#pragma unroll
    for (int i = 0; i < 9; i++) acc |= c.l[i];
    return acc == 0;
}

// The prefix filter's multiply. For y = a*b mod p, with a tight and b in
// signed limbs (|b_i| < 2^30 + 2^18: an add() or a dif() of tight values),
// sets low to the low 32 bits of y and returns false; or returns true
// ("unsure", about one call in 2^15), when the caller must compute y in full.
// Only the columns that reach the low 32 bits and the top columns are formed:
// 35 limb products instead of 81. Six target characters (30 stream bits) lie
// in y's low 32 bits, so this decides any target of up to six characters
// outright and pre-filters longer ones.
//
// Why it is exact. With d_m the folded columns of a*b (as in mul, before any
// carry; signed here, each below 2^62.2 in size), V = sum d_m 2^(29m) is
// congruent to y. With Q = floor(V / 2^255) and R = V mod 2^255,
// y = R + 19Q unless that leaves [0, p). The low 32 bits of V need only d_0
// and d_1 mod 8 (the fold factors 1216 and 9728 vanish mod 8). For Q:
// floor(V / 2^232) = N + e with N = d_8 + floor(d_7 / 2^29) and |e| < 20 (the
// carries out of d_0..d_6). So Q = floor(N / 2^23) unless N mod 2^23 is within
// 128 of 0 or of 2^23. Outside that window R + 19Q is also in [0, p): above
// p - 1 needs R's bits 232..254 all ones, below 0 needs Q < 0 and
// R < 19|Q| < 2^44, so those bits all zeros; either puts N mod 2^23 inside the
// window.
HD bool leaf32(const fe &a, const fe &b, uint32_t &low) {
    uint32_t x[9];
    int32_t y[9];
#pragma unroll
    for (int i = 0; i < 9; i++) { x[i] = a.l[i]; y[i] = (int32_t)b.l[i]; }
    const int64_t c9 = columni(x, y, 9, true, 0);
    // Mod 2^32 the products and folds do not care about signs.
    const uint32_t d0 = (uint32_t)mads((uint32_t)c9, (uint32_t)F261, (uint64_t)mul0i(x[0], y[0]));
    const uint32_t d1 = x[0] * (uint32_t)y[1] + x[1] * (uint32_t)y[0];
    const int64_t c7 = columni(x, y, 7, true, 0);
    const int64_t c15 = columni(x, y, 15, true, 0);
    const int64_t c16 = columni(x, y, 16, true, 0);
    // A signed column c = hi 2^32 + lo with hi = c >> 32 (floor) and lo the
    // unsigned low word: the lo fold is unsigned, the hi fold signed.
    const int64_t d7 = madis((int32_t)(c15 >> 32), (uint32_t)F264,
                             (int64_t)mads((uint32_t)c16, (uint32_t)F261, (uint64_t)c7));
    // n = d8 + floor(d7 / 2^29) with d8 = c8 + 9728 hi(c16): c8's chain starts
    // from the carry, which saves a 64-bit add.
    int64_t n = columni(x, y, 8, false, d7 >> 29);
    n = madis((int32_t)(c16 >> 32), (uint32_t)F264, n);
    low = d0 + (d1 << 29) + 19u * (uint32_t)(n >> 23);
    return (((uint32_t)n + 0x80u) & 0x7FFFFFu) < 0x100u;
}

// r = a - b in signed limbs (two's complement in the uint32 limbs), for
// leaf32. a and b tight.
HD void dif(fe &r, const fe &a, const fe &b) {
#pragma unroll
    for (int i = 0; i < 9; i++) r.l[i] = a.l[i] - b.l[i];
}

// Signed limbs (from dif) back to the unsigned loose form sub() returns.
HD void undif(fe &r, const fe &a) {
    r.l[0] = a.l[0] + Z0;
#pragma unroll
    for (int i = 1; i < 9; i++) r.l[i] = a.l[i] + ZI;
}

// Host conversions between canonical 32-byte little-endian and limbs.
inline fe from_bytes(const uint8_t s[32]) {
    fe r;
    for (int i = 0; i < 9; i++) {
        uint32_t v = 0;
        for (int b = 0; b < 29; b++) {
            int bit = 29 * i + b;
            if (bit < 255 && ((s[bit / 8] >> (bit % 8)) & 1)) v |= 1u << b;
        }
        r.l[i] = v;
    }
    return r;
}

inline void to_bytes(uint8_t s[32], const fe &a) {
    fe c;
    canon(c, a);
    for (int i = 0; i < 32; i++) s[i] = 0;
    for (int i = 0; i < 9; i++)
        for (int b = 0; b < 29; b++) {
            int bit = 29 * i + b;
            if (bit < 256 && ((c.l[i] >> b) & 1)) s[bit / 8] |= (uint8_t)(1u << (bit % 8));
        }
}

}  // namespace f29
