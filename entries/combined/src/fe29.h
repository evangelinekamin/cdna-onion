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
//   mul needs one tight operand and one loose operand (or tighter); ip needs
//   all four operands tight; leaf32 one tight and one signed.
// Every function compiles for both host and device, so the CPU tests exercise
// exactly the arithmetic the GPU runs.
#pragma once
#include <cstdint>

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
#else
HD uint64_t mad(uint32_t a, uint32_t b, uint64_t c) { return (uint64_t)a * b + c; }
HD uint64_t mads(uint32_t a, uint32_t b, uint64_t c) { return (uint64_t)a * b + c; }
HD uint64_t mul0(uint32_t a, uint32_t b) { return (uint64_t)a * b; }
HD uint64_t mul0s(uint32_t a, uint32_t b) { return (uint64_t)a * b; }
HD int64_t madi(uint32_t a, int32_t b, int64_t c) { return (int64_t)(int32_t)a * b + c; }
HD int64_t madis(int32_t a, uint32_t b, int64_t c) { return (int64_t)a * (int32_t)b + c; }
HD int64_t mul0i(uint32_t a, int32_t b) { return (int64_t)(int32_t)a * b; }
#endif

// Operand kind as a template switch: S = the second operand is wave-uniform.
template <bool S> HD uint64_t madk(uint32_t a, uint32_t b, uint64_t c) { return S ? mads(a, b, c) : mad(a, b, c); }
template <bool S> HD uint64_t mul0k(uint32_t a, uint32_t b) { return S ? mul0s(a, b) : mul0(a, b); }

// The carry out of a column: acc >> 29, a 64-bit value below 2^35 (one
// v_lshrrev_b64, which issues at full rate: two 32-bit instructions measured
// slower).
HD uint64_t carry29(uint64_t acc) { return acc >> 29; }

// The end of a reduction: limb 8 from column 8's accumulator acc (< 2^63),
// and the carry out of it (bits 261 and up), folded into limbs 0 and 1:
// 1216 (acc >> 29) = 9728 hi + 1216 (lo >> 29) for acc = hi 2^32 + lo, two
// multiply-adds without forming the 64-bit carry.
HD void fold_top(fe &r, uint64_t acc) {
    const uint32_t lo = (uint32_t)acc, hi = (uint32_t)(acc >> 32);
    r.l[8] = lo & M29;
    uint64_t t = mads(hi, (uint32_t)F264, (uint64_t)r.l[0]);
    t = mads(lo >> 29, (uint32_t)F261, t);  // t < 2^45
    r.l[0] = (uint32_t)t & M29;
    r.l[1] += (uint32_t)(t >> 29);
}

// r = a * b. a tight, b loose (or both tight). r tight. r may alias a or b.
// S: b is wave-uniform (SGPR operands).
template <bool S>
HD void mul_k(fe &r, const fe &a, const fe &b) {
    uint32_t x[9], y[9];
#pragma unroll
    for (int i = 0; i < 9; i++) { x[i] = a.l[i]; y[i] = b.l[i]; }
    // Streaming order: high column c_{m+9} is formed just before low column m
    // folds it in, so only two high columns are ever live (register pressure).
    uint64_t hprev = 0, acc = 0;
#pragma unroll
    for (int m = 0; m <= 8; m++) {
        uint64_t h = 0;
        if (m <= 7) {
            const int k = m + 9;
            h = mul0k<S>(x[k - 8], y[8]);
#pragma unroll
            for (int i = k - 7; i <= 8; i++) h = madk<S>(x[i], y[k - i], h);
        }
        acc = (m == 0) ? mul0k<S>(x[0], y[0]) : madk<S>(x[0], y[m], acc);
#pragma unroll
        for (int i = 1; i <= m; i++) acc = madk<S>(x[i], y[m - i], acc);
        if (m <= 7) acc = mads((uint32_t)h, (uint32_t)F261, acc);
        if (m >= 1) acc = mads((uint32_t)(hprev >> 32), (uint32_t)F264, acc);
        if (m == 8) break;
        r.l[m] = (uint32_t)acc & M29;
        acc = carry29(acc);
        hprev = h;
    }
    fold_top(r, acc);
}

HD void mul(fe &r, const fe &a, const fe &b) { mul_k<false>(r, a, b); }

// r = a^2. a tight. r tight. r may alias a.
HD void sqr(fe &r, const fe &a) {
    uint32_t x[9], x2[9];
#pragma unroll
    for (int i = 0; i < 9; i++) { x[i] = a.l[i]; x2[i] = a.l[i] << 1; }
    uint64_t h[8];
#pragma unroll
    for (int k = 9; k <= 16; k++) {
        // Cross terms x_i * 2x_{k-i} for i < k - i, then the square for even k.
        uint64_t s = (k == 16) ? mul0(x[8], x[8]) : mul0(x[k - 8], x2[8]);
#pragma unroll
        for (int i = k - 7; 2 * i < k; i++) s = mad(x[i], x2[k - i], s);
        if ((k & 1) == 0 && k != 16) s = mad(x[k / 2], x[k / 2], s);
        h[k - 9] = s;
    }
    // Column 0 is the square x0^2; then fold, carry, and the rest.
    uint64_t acc = mul0(x[0], x[0]);
    acc = mads((uint32_t)h[0], (uint32_t)F261, acc);
    r.l[0] = (uint32_t)acc & M29;
    acc = carry29(acc);
#pragma unroll
    for (int m = 1; m <= 8; m++) {
#pragma unroll
        for (int i = 0; 2 * i < m; i++) acc = mad(x[i], x2[m - i], acc);
        if ((m & 1) == 0) acc = mad(x[m / 2], x[m / 2], acc);
        if (m <= 7) acc = mads((uint32_t)h[m], (uint32_t)F261, acc);
        acc = mads((uint32_t)(h[m - 1] >> 32), (uint32_t)F264, acc);
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
    uint32_t x[9], y[9], u[9], v[9];
#pragma unroll
    for (int i = 0; i < 9; i++) { x[i] = a.l[i]; y[i] = b.l[i]; u[i] = c.l[i]; v[i] = d.l[i]; }
    uint64_t hprev = 0, acc = 0;
#pragma unroll
    for (int m = 0; m <= 8; m++) {
        uint64_t h = 0;
        if (m <= 7) {
            const int k = m + 9;
            h = mul0k<S>(x[k - 8], y[8]);
            h = madk<S>(u[k - 8], v[8], h);
#pragma unroll
            for (int i = k - 7; i <= 8; i++) {
                h = madk<S>(x[i], y[k - i], h);
                h = madk<S>(u[i], v[k - i], h);
            }
        }
        acc = (m == 0) ? mul0k<S>(x[0], y[0]) : madk<S>(x[0], y[m], acc);
        acc = madk<S>(u[0], v[m], acc);
#pragma unroll
        for (int i = 1; i <= m; i++) {
            acc = madk<S>(x[i], y[m - i], acc);
            acc = madk<S>(u[i], v[m - i], acc);
        }
        if (m <= 7) acc = mads((uint32_t)h, (uint32_t)F261, acc);
        if (m >= 1) acc = mads((uint32_t)(hprev >> 32), (uint32_t)F264, acc);
        if (m == 8) break;
        r.l[m] = (uint32_t)acc & M29;
        acc = carry29(acc);
        hprev = h;
    }
    fold_top(r, acc);
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
    int64_t c9 = mul0i(x[1], y[8]);
#pragma unroll
    for (int i = 2; i <= 8; i++) c9 = madi(x[i], y[9 - i], c9);
    // Mod 2^32 the products and folds do not care about signs.
    const uint32_t d0 = (uint32_t)mads((uint32_t)c9, (uint32_t)F261, (uint64_t)mul0i(x[0], y[0]));
    const uint32_t d1 = x[0] * (uint32_t)y[1] + x[1] * (uint32_t)y[0];
    int64_t c7 = mul0i(x[0], y[7]);
#pragma unroll
    for (int i = 1; i <= 7; i++) c7 = madi(x[i], y[7 - i], c7);
    const int64_t c15 = madi(x[8], y[7], mul0i(x[7], y[8]));
    const int64_t c16 = mul0i(x[8], y[8]);
    // A signed column c = hi 2^32 + lo with hi = c >> 32 (floor) and lo the
    // unsigned low word: the lo fold is unsigned, the hi fold signed.
    const int64_t d7 = madis((int32_t)(c15 >> 32), (uint32_t)F264,
                             (int64_t)mads((uint32_t)c16, (uint32_t)F261, (uint64_t)c7));
    // n = d8 + floor(d7 / 2^29) with d8 = c8 + 9728 hi(c16): c8's chain starts
    // from the carry, which saves a 64-bit add.
    int64_t n = madi(x[0], y[8], d7 >> 29);
#pragma unroll
    for (int i = 1; i <= 8; i++) n = madi(x[i], y[8 - i], n);
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
