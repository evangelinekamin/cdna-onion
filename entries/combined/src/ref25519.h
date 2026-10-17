// Exact host-side ed25519 arithmetic: radix 2^51 limbs with unsigned __int128.
//
// This is the entry's independent second opinion. It builds the tables the GPU
// uses (before "ready"), decodes A0, and re-derives every candidate the GPU flags
// before that offset is reported. It shares no code with the GPU field arithmetic
// (fe29.h), so a bug there cannot hide itself here.
#pragma once
#include <cstdint>
#include <cstring>
#include <string>

namespace ref {

typedef unsigned __int128 u128;
static const uint64_t M51 = (1ULL << 51) - 1;

struct fe {
    uint64_t v[5];
};

inline fe fe_small(uint64_t x) {
    fe r = {{x & M51, x >> 51, 0, 0, 0}};
    return r;
}

// Limbs back under 2^51 (limb 1 may keep a few extra bits).
inline void fe_carry(fe &a) {
    uint64_t c;
    c = a.v[0] >> 51; a.v[0] &= M51; a.v[1] += c;
    c = a.v[1] >> 51; a.v[1] &= M51; a.v[2] += c;
    c = a.v[2] >> 51; a.v[2] &= M51; a.v[3] += c;
    c = a.v[3] >> 51; a.v[3] &= M51; a.v[4] += c;
    c = a.v[4] >> 51; a.v[4] &= M51; a.v[0] += c * 19;
    c = a.v[0] >> 51; a.v[0] &= M51; a.v[1] += c;
}

inline fe fe_add(const fe &a, const fe &b) {
    fe r;
    for (int i = 0; i < 5; i++) r.v[i] = a.v[i] + b.v[i];
    fe_carry(r);
    return r;
}

// a - b + 4p, so no limb underflows for inputs with limbs below 2^53.
inline fe fe_sub(const fe &a, const fe &b) {
    fe r;
    r.v[0] = a.v[0] + 0x1FFFFFFFFFFFB4ULL - b.v[0];
    for (int i = 1; i < 5; i++) r.v[i] = a.v[i] + 0x1FFFFFFFFFFFFCULL - b.v[i];
    fe_carry(r);
    return r;
}

inline fe fe_neg(const fe &a) { return fe_sub(fe_small(0), a); }

inline fe fe_mul(const fe &a, const fe &b) {
    const uint64_t *x = a.v, *y = b.v;
    const uint64_t y19_1 = y[1] * 19, y19_2 = y[2] * 19, y19_3 = y[3] * 19, y19_4 = y[4] * 19;
    u128 t0 = (u128)x[0] * y[0] + (u128)x[1] * y19_4 + (u128)x[2] * y19_3 +
              (u128)x[3] * y19_2 + (u128)x[4] * y19_1;
    u128 t1 = (u128)x[0] * y[1] + (u128)x[1] * y[0] + (u128)x[2] * y19_4 +
              (u128)x[3] * y19_3 + (u128)x[4] * y19_2;
    u128 t2 = (u128)x[0] * y[2] + (u128)x[1] * y[1] + (u128)x[2] * y[0] +
              (u128)x[3] * y19_4 + (u128)x[4] * y19_3;
    u128 t3 = (u128)x[0] * y[3] + (u128)x[1] * y[2] + (u128)x[2] * y[1] +
              (u128)x[3] * y[0] + (u128)x[4] * y19_4;
    u128 t4 = (u128)x[0] * y[4] + (u128)x[1] * y[3] + (u128)x[2] * y[2] +
              (u128)x[3] * y[1] + (u128)x[4] * y[0];
    fe r;
    t1 += (uint64_t)(t0 >> 51); r.v[0] = (uint64_t)t0 & M51;
    t2 += (uint64_t)(t1 >> 51); r.v[1] = (uint64_t)t1 & M51;
    t3 += (uint64_t)(t2 >> 51); r.v[2] = (uint64_t)t2 & M51;
    t4 += (uint64_t)(t3 >> 51); r.v[3] = (uint64_t)t3 & M51;
    uint64_t c = (uint64_t)(t4 >> 51); r.v[4] = (uint64_t)t4 & M51;
    r.v[0] += c * 19;
    c = r.v[0] >> 51; r.v[0] &= M51; r.v[1] += c;
    return r;
}

inline fe fe_sq(const fe &a) { return fe_mul(a, a); }

inline fe fe_sqn(fe a, int n) {
    for (int i = 0; i < n; i++) a = fe_sq(a);
    return a;
}

// a^(2^255 - 21) = 1/a (0 for a = 0), via the usual chain of 254 squarings.
inline fe fe_inv(const fe &z) {
    fe z2 = fe_sq(z);
    fe z9 = fe_mul(fe_sqn(z2, 2), z);
    fe z11 = fe_mul(z9, z2);
    fe z5_0 = fe_mul(fe_sq(z11), z9);              // z^(2^5 - 1)
    fe z10_0 = fe_mul(fe_sqn(z5_0, 5), z5_0);
    fe z20_0 = fe_mul(fe_sqn(z10_0, 10), z10_0);
    fe z40_0 = fe_mul(fe_sqn(z20_0, 20), z20_0);
    fe z50_0 = fe_mul(fe_sqn(z40_0, 10), z10_0);
    fe z100_0 = fe_mul(fe_sqn(z50_0, 50), z50_0);
    fe z200_0 = fe_mul(fe_sqn(z100_0, 100), z100_0);
    fe z250_0 = fe_mul(fe_sqn(z200_0, 50), z50_0);
    return fe_mul(fe_sqn(z250_0, 5), z11);
}

// a^((p - 5) / 8) = a^(2^252 - 3), for square roots.
inline fe fe_pow22523(const fe &z) {
    fe z2 = fe_sq(z);
    fe z9 = fe_mul(fe_sqn(z2, 2), z);
    fe z11 = fe_mul(z9, z2);
    fe z5_0 = fe_mul(fe_sq(z11), z9);
    fe z10_0 = fe_mul(fe_sqn(z5_0, 5), z5_0);
    fe z20_0 = fe_mul(fe_sqn(z10_0, 10), z10_0);
    fe z40_0 = fe_mul(fe_sqn(z20_0, 20), z20_0);
    fe z50_0 = fe_mul(fe_sqn(z40_0, 10), z10_0);
    fe z100_0 = fe_mul(fe_sqn(z50_0, 50), z50_0);
    fe z200_0 = fe_mul(fe_sqn(z100_0, 100), z100_0);
    fe z250_0 = fe_mul(fe_sqn(z200_0, 50), z50_0);
    return fe_mul(fe_sqn(z250_0, 2), z);
}

// Canonical little-endian encoding of the value mod p.
inline void fe_tobytes(uint8_t out[32], fe a) {
    fe_carry(a);
    fe_carry(a);
    // Now a < 2^255 + small. q = 1 iff a >= p.
    uint64_t q = (a.v[0] + 19) >> 51;
    q = (a.v[1] + q) >> 51;
    q = (a.v[2] + q) >> 51;
    q = (a.v[3] + q) >> 51;
    q = (a.v[4] + q) >> 51;
    a.v[0] += 19 * q;
    uint64_t c;
    c = a.v[0] >> 51; a.v[0] &= M51; a.v[1] += c;
    c = a.v[1] >> 51; a.v[1] &= M51; a.v[2] += c;
    c = a.v[2] >> 51; a.v[2] &= M51; a.v[3] += c;
    c = a.v[3] >> 51; a.v[3] &= M51; a.v[4] += c;
    a.v[4] &= M51;  // drops 2^255, completing the subtraction of p
    uint64_t w[4];
    w[0] = a.v[0] | (a.v[1] << 51);
    w[1] = (a.v[1] >> 13) | (a.v[2] << 38);
    w[2] = (a.v[2] >> 26) | (a.v[3] << 25);
    w[3] = (a.v[3] >> 39) | (a.v[4] << 12);
    for (int i = 0; i < 4; i++)
        for (int b = 0; b < 8; b++) out[8 * i + b] = (uint8_t)(w[i] >> (8 * b));
}

// Reads 255 bits (bit 255 ignored); the value may be >= p (non-canonical).
inline fe fe_frombytes(const uint8_t in[32]) {
    uint64_t w[4];
    for (int i = 0; i < 4; i++) {
        w[i] = 0;
        for (int b = 0; b < 8; b++) w[i] |= (uint64_t)in[8 * i + b] << (8 * b);
    }
    w[3] &= 0x7FFFFFFFFFFFFFFFULL;
    fe r;
    r.v[0] = w[0] & M51;
    r.v[1] = ((w[0] >> 51) | (w[1] << 13)) & M51;
    r.v[2] = ((w[1] >> 38) | (w[2] << 26)) & M51;
    r.v[3] = ((w[2] >> 25) | (w[3] << 39)) & M51;
    r.v[4] = (w[3] >> 12) & M51;
    return r;
}

inline bool fe_iszero(const fe &a) {
    uint8_t s[32];
    fe_tobytes(s, a);
    uint8_t acc = 0;
    for (int i = 0; i < 32; i++) acc |= s[i];
    return acc == 0;
}

inline bool fe_eq(const fe &a, const fe &b) { return fe_iszero(fe_sub(a, b)); }

inline int fe_isodd(const fe &a) {
    uint8_t s[32];
    fe_tobytes(s, a);
    return s[0] & 1;
}

// The canonical value as four little-endian 64-bit words.
inline void fe_words(uint64_t w[4], const fe &a) {
    uint8_t s[32];
    fe_tobytes(s, a);
    for (int i = 0; i < 4; i++) {
        w[i] = 0;
        for (int b = 0; b < 8; b++) w[i] |= (uint64_t)s[8 * i + b] << (8 * b);
    }
}

// ---------------------------------------------------------------------------
// Curve: -x^2 + y^2 = 1 + d x^2 y^2, points in extended coordinates.
// ---------------------------------------------------------------------------
struct ge {
    fe X, Y, Z, T;
};

struct Consts {
    fe d, d2, sqrtm1;
};

inline const Consts &consts();

inline ge ge_identity() {
    ge r = {fe_small(0), fe_small(1), fe_small(1), fe_small(0)};
    return r;
}

// Unified addition (Hisil-Wong-Carter-Dawson, add-2008-hwcd-3), complete on ed25519.
inline ge ge_add(const ge &p, const ge &q) {
    const Consts &k = consts();
    fe a = fe_mul(fe_sub(p.Y, p.X), fe_sub(q.Y, q.X));
    fe b = fe_mul(fe_add(p.Y, p.X), fe_add(q.Y, q.X));
    fe c = fe_mul(fe_mul(p.T, k.d2), q.T);
    fe dd = fe_mul(p.Z, q.Z);
    dd = fe_add(dd, dd);
    fe e = fe_sub(b, a), f = fe_sub(dd, c), g = fe_add(dd, c), h = fe_add(b, a);
    ge r;
    r.X = fe_mul(e, f);
    r.Y = fe_mul(g, h);
    r.T = fe_mul(e, h);
    r.Z = fe_mul(f, g);
    return r;
}

inline ge ge_neg(const ge &p) {
    ge r = p;
    r.X = fe_neg(p.X);
    r.T = fe_neg(p.T);
    return r;
}

// Affine coordinates of p.
inline void ge_affine(fe &x, fe &y, const ge &p) {
    fe zi = fe_inv(p.Z);
    x = fe_mul(p.X, zi);
    y = fe_mul(p.Y, zi);
}

inline ge ge_from_affine(const fe &x, const fe &y) {
    ge r = {x, y, fe_small(1), fe_mul(x, y)};
    return r;
}

inline void ge_tobytes(uint8_t out[32], const ge &p) {
    fe x, y;
    ge_affine(x, y, p);
    fe_tobytes(out, y);
    out[31] |= (uint8_t)(fe_isodd(x) << 7);
}

inline bool ge_eq(const ge &p, const ge &q) {
    return fe_eq(fe_mul(p.X, q.Z), fe_mul(q.X, p.Z)) && fe_eq(fe_mul(p.Y, q.Z), fe_mul(q.Y, p.Z));
}

// RFC 8032 section 5.1.3. False for any invalid encoding (y >= p, no root, -0).
inline bool ge_frombytes(ge &out, const uint8_t s[32]) {
    // y must be canonical: reject 2^255-19 <= y < 2^255.
    bool all_ff = (s[31] & 0x7f) == 0x7f;
    for (int i = 30; i >= 1 && all_ff; i--) all_ff = s[i] == 0xff;
    if (all_ff && s[0] >= 0xed) return false;
    const int sign = s[31] >> 7;
    const Consts &k = consts();
    fe y = fe_frombytes(s);
    fe one = fe_small(1);
    fe yy = fe_sq(y);
    fe u = fe_sub(yy, one);
    fe v = fe_add(fe_mul(yy, k.d), one);
    fe v3 = fe_mul(fe_sq(v), v);
    fe x = fe_mul(fe_mul(fe_sq(v3), v), u);  // u v^7
    x = fe_pow22523(x);
    x = fe_mul(fe_mul(x, v3), u);            // u v^3 (u v^7)^((p-5)/8)
    fe vxx = fe_mul(fe_sq(x), v);
    if (!fe_eq(vxx, u)) {
        if (!fe_eq(vxx, fe_neg(u))) return false;
        x = fe_mul(x, k.sqrtm1);
    }
    if (fe_iszero(x) && sign) return false;
    if (fe_isodd(x) != sign) x = fe_neg(x);
    out = ge_from_affine(x, y);
    return true;
}

// n * p for n >= 0, by double-and-add.
inline ge ge_scalarmult(const ge &p, unsigned __int128 n) {
    ge r = ge_identity(), a = p;
    while (n) {
        if (n & 1) r = ge_add(r, a);
        a = ge_add(a, a);
        n >>= 1;
    }
    return r;
}

inline bool hex_decode(const std::string &hex, uint8_t *out, size_t n) {
    if (hex.size() != 2 * n) return false;
    for (size_t i = 0; i < n; i++) {
        int v = 0;
        for (int j = 0; j < 2; j++) {
            char c = hex[2 * i + j];
            int d = (c >= '0' && c <= '9') ? c - '0'
                  : (c >= 'a' && c <= 'f') ? c - 'a' + 10
                  : (c >= 'A' && c <= 'F') ? c - 'A' + 10 : -1;
            if (d < 0) return false;
            v = v * 16 + d;
        }
        out[i] = (uint8_t)v;
    }
    return true;
}

inline const Consts &consts() {
    static const Consts k = [] {
        Consts c;
        // d = -121665 / 121666
        c.d = fe_mul(fe_neg(fe_small(121665)), fe_inv(fe_small(121666)));
        c.d2 = fe_add(c.d, c.d);
        // sqrt(-1) = 2^((p-1)/4)
        fe two = fe_small(2), acc = fe_small(1);
        // (p-1)/4 = 2^253 - 5: square-and-multiply over its bits, MSB first.
        // 2^253 - 5 = 0b1111...11011 (253 bits: all ones except bit 2).
        for (int bit = 252; bit >= 0; bit--) {
            acc = fe_sq(acc);
            if (bit != 2) acc = fe_mul(acc, two);
        }
        c.sqrtm1 = acc;
        return c;
    }();
    return k;
}

// The base point B (decoded once; kept separate so consts() has no cycle).
inline const ge &base_point() {
    static const ge B = [] {
        uint8_t s[32];
        hex_decode("5866666666666666666666666666666666666666666666666666666666666666", s, 32);
        ge p;
        ge_frombytes(p, s);
        return p;
    }();
    return B;
}

}  // namespace ref
