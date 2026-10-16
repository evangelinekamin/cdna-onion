// Group operations the contract needs beyond upstream onion-gpu: decoding the
// public start point A0, negation, and points at a signed offset A0 + k*(8B).
// Built only from upstream's field and group primitives (fe25519.cuh and
// ge25519.cuh are byte-for-byte upstream).
#pragma once
#include "fe25519.cuh"
#include "ge25519.cuh"

// Curve constants as canonical little-endian encodings, generated from (and
// checked against) the Python reference: d = -121665/121666 and sqrt(-1).
__device__ __constant__ uint8_t D_BYTES[32] = {
    0xa3, 0x78, 0x59, 0x13, 0xca, 0x4d, 0xeb, 0x75,
    0xab, 0xd8, 0x41, 0x41, 0x4d, 0x0a, 0x70, 0x00,
    0x98, 0xe8, 0x79, 0x77, 0x79, 0x40, 0xc7, 0x8c,
    0x73, 0xfe, 0x6f, 0x2b, 0xee, 0x6c, 0x03, 0x52
};
__device__ __constant__ uint8_t SQRTM1_BYTES[32] = {
    0xb0, 0xa0, 0x0e, 0x4a, 0x27, 0x1b, 0xee, 0xc4,
    0x78, 0xe4, 0x2f, 0xad, 0x06, 0x18, 0x43, 0x2f,
    0xa7, 0xd7, 0xfb, 0x3d, 0x99, 0x00, 0x4d, 0x2b,
    0x0b, 0xdf, 0xc1, 0x4f, 0x80, 0x24, 0x83, 0x2b
};

__device__ bool fe_is_zero(const fe f) {
    uint8_t s[32];
    fe_tobytes(s, f);
    uint8_t acc = 0;
    for (int i = 0; i < 32; i++) acc |= s[i];
    return acc == 0;
}

__device__ int fe_low_bit(const fe f) {
    uint8_t s[32];
    fe_tobytes(s, f);
    return s[0] & 1;
}

// y < p for the 255-bit value in s (bit 255 ignored): RFC 8032 rejects y >= p.
__device__ bool y_is_canonical(const uint8_t s[32]) {
    if ((s[31] & 0x7f) != 0x7f) return true;
    for (int i = 30; i >= 1; i--)
        if (s[i] != 0xff) return true;
    return s[0] < 0xed;  // p = 2^255 - 19 = 0x7fff...ffed
}

// RFC 8032 section 5.1.3 decoding. Returns false for any invalid encoding.
__device__ bool ge_frombytes(ge_p3 *h, const uint8_t s[32]) {
    if (!y_is_canonical(s)) return false;
    const int sign = s[31] >> 7;
    // Upstream's fe_frombytes does not simply ignore bit 255 (it applies a
    // correction meant for a different convention); upstream only ever passes
    // it constants with that bit clear. Bit 255 here is the x sign, so clear it.
    uint8_t ybuf[32];
    for (int i = 0; i < 32; i++) ybuf[i] = s[i];
    ybuf[31] &= 0x7f;
    fe y, u, v, v3, x, vxx, check, d, one;
    fe_frombytes(d, D_BYTES);
    fe_1(one);
    fe_frombytes(y, ybuf);
    fe_sq(u, y);
    fe_mul(v, u, d);
    fe_sub(u, u, one);         // u = y^2 - 1
    fe_add(v, v, one);         // v = d*y^2 + 1
    fe_sq(v3, v);
    fe_mul(v3, v3, v);         // v^3
    fe_sq(x, v3);
    fe_mul(x, x, v);
    fe_mul(x, x, u);           // u*v^7
    fe_pow22523(x, x);         // (u*v^7)^((p-5)/8)
    fe_mul(x, x, v3);
    fe_mul(x, x, u);           // candidate root u*v^3*(u*v^7)^((p-5)/8)
    fe_sq(vxx, x);
    fe_mul(vxx, vxx, v);
    fe_sub(check, vxx, u);
    if (!fe_is_zero(check)) {
        fe_add(check, vxx, u);
        if (!fe_is_zero(check)) return false;  // no square root: not on the curve
        fe sqrtm1;
        fe_frombytes(sqrtm1, SQRTM1_BYTES);
        fe_mul(x, x, sqrtm1);
    }
    if (fe_is_zero(x) && sign) return false;
    if (fe_low_bit(x) != sign) fe_neg(x, x);
    fe_copy(h->X, x);
    fe_copy(h->Y, y);
    fe_1(h->Z);
    fe_mul(h->T, x, y);
    return true;
}

__device__ void ge_neg(ge_p3 *h) {
    fe_neg(h->X, h->X);
    fe_neg(h->T, h->T);
}

__device__ void ge_add_p3(ge_p3 *r, const ge_p3 *p, const ge_p3 *q) {
    ge_p1p1 t;
    ge_add(&t, p, q);
    ge_p1p1_to_p3(r, &t);
}

// r = k * (8B) for a signed offset k with |k| < 2^63, via upstream's
// double-and-add on the scalar 8|k|.
__device__ void ge_step_multiple(ge_p3 *r, int64_t k) {
    const uint64_t mag = k < 0 ? (uint64_t)(-(k + 1)) + 1 : (uint64_t)k;
    uint8_t s[32] = {0};
    for (int i = 0; i < 8; i++) s[i] = (uint8_t)(mag >> (8 * i));
    // Multiply the 64-bit magnitude by 8: a 3-bit shift across the 32 bytes.
    uint8_t carry = 0;
    for (int i = 0; i < 32; i++) {
        const uint8_t next = s[i] >> 5;
        s[i] = (uint8_t)(s[i] << 3) | carry;
        carry = next;
    }
    ge_scalarmult_base(r, s);
    if (k < 0) ge_neg(r);
}

// r = A0 + k*(8B): the candidate the contract assigns to offset k.
__device__ void ge_at_offset(ge_p3 *r, const ge_p3 *a0, int64_t k) {
    ge_p3 step;
    ge_step_multiple(&step, k);
    ge_add_p3(r, a0, &step);
}
