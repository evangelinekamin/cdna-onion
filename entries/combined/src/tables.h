// Host-side construction of everything the GPU walk needs that does not depend
// on A0 (built before "ready"), plus the per-run start points (which do).
// All exact, via the reference arithmetic.
#pragma once
#include <cstdint>
#include <vector>

#include "fe29.h"
#include "ref25519.h"
#include "walk.h"

namespace tables {

inline f29::fe to29(const ref::fe &a) {
    uint8_t s[32];
    ref::fe_tobytes(s, a);
    return f29::from_bytes(s);
}

inline const ref::ge &B8() {
    static const ref::ge b8 = ref::ge_scalarmult(ref::base_point(), 8);
    return b8;
}

// p + n*B8 for a signed n.
inline ref::ge offset(const ref::ge &p, int64_t n) {
    const unsigned __int128 mag = n < 0 ? (unsigned __int128)(-(__int128)n) : (unsigned __int128)n;
    ref::ge step = ref::ge_scalarmult(B8(), mag);
    return ref::ge_add(p, n < 0 ? ref::ge_neg(step) : step);
}

// entries[j] for j = 1..H (entries[0] unused).
inline std::vector<walk::Entry> build_table(int H) {
    std::vector<walk::Entry> t(H + 1);
    ref::ge T = B8();
    for (int j = 1; j <= H; j++) {
        ref::fe x2, y2;
        ref::ge_affine(x2, y2, T);
        const ref::fe r = ref::fe_inv(x2);
        const ref::fe s = ref::fe_mul(y2, r);
        t[j].sigma = to29(ref::fe_sq(s));
        t[j].rs = to29(ref::fe_mul(r, s));
        t[j].ny2 = to29(ref::fe_neg(y2));
        t[j].r = to29(r);
        t[j].ny2s = to29(ref::fe_neg(ref::fe_mul(y2, s)));
        T = ref::ge_add(T, B8());
    }
    return t;
}

// The giant step n*B8 (n > 0).
inline walk::Giant giant(int64_t n) {
    ref::fe gx, gy;
    ref::ge_affine(gx, gy, ref::ge_scalarmult(B8(), (unsigned __int128)n));
    walk::Giant g;
    g.gx = to29(gx);
    g.gy = to29(gy);
    g.dg = to29(ref::fe_mul(ref::consts().d, ref::fe_mul(gx, gy)));
    return g;
}

// A lane centered on P (exact). False if x = 0 (P is the identity or the
// 2-torsion point), where the walk's 1/x is undefined.
inline bool lane_at(walk::Lane &lane, const ref::ge &P) {
    ref::fe x, y;
    ref::ge_affine(x, y, P);
    if (ref::fe_iszero(x)) return false;
    lane.x = to29(x);
    lane.y = to29(y);
    lane.mu = to29(ref::fe_inv(x));
    return true;
}

}  // namespace tables
