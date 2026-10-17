// Cross-checks the GPU field arithmetic (src/fe29.h, compiled for the host)
// against the independent reference (src/ref25519.h), including inputs at the
// documented limb bounds and the prefix filter's exactness (leaf32).
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>

#include "../src/fe29.h"
#include "../src/ref25519.h"
#include "../src/vectors.h"

static int failures = 0, passes = 0;
static void expect(bool ok, const char *what, int i) {
    if (ok) {
        passes++;
    } else if (failures++ < 20) {
        std::fprintf(stderr, "FAIL: %s (case %d)\n", what, i);
    }
}

static std::mt19937_64 rng(12345);

// Value of a limb vector as a ref field element (any limbs < 2^32).
static ref::fe to_ref(const f29::fe &a) {
    ref::fe acc = ref::fe_small(0), scale = ref::fe_small(1);
    const ref::fe radix = ref::fe_small(1ULL << 29);
    for (int i = 0; i < 9; i++) {
        acc = ref::fe_add(acc, ref::fe_mul(ref::fe_small(a.l[i]), scale));
        scale = ref::fe_mul(scale, radix);
    }
    return acc;
}

static f29::fe from_ref(const ref::fe &a) {
    uint8_t s[32];
    ref::fe_tobytes(s, a);
    return f29::from_bytes(s);
}

// Random limbs below `bound` (exclusive), with a bias toward the bound.
static f29::fe random_limbs(uint64_t bound) {
    f29::fe r;
    for (int i = 0; i < 9; i++) {
        uint64_t v = rng() % bound;
        if (rng() % 4 == 0) v = bound - 1 - (rng() % 16);
        r.l[i] = (uint32_t)v;
    }
    return r;
}

static const uint64_t TIGHT = (1ULL << 29) + (1ULL << 17);
static const uint64_t LOOSE = 1627389952ULL;  // 2^30.6 rounded down: 1.515625 * 2^30

static bool tight(const f29::fe &a) {
    for (int i = 0; i < 9; i++)
        if (a.l[i] >= TIGHT) return false;
    return true;
}

static bool same(const f29::fe &a, const ref::fe &b) { return ref::fe_eq(to_ref(a), b); }

int main() {
    const int N = 20000;
    for (int i = 0; i < N; i++) {
        // Mix canonical, tight-at-bound, and loose-at-bound inputs.
        f29::fe a = (i % 3 == 0) ? from_ref(to_ref(random_limbs(1ULL << 32))) : random_limbs(TIGHT);
        f29::fe b = (i % 2 == 0) ? random_limbs(LOOSE) : random_limbs(TIGHT);
        f29::fe c = random_limbs(TIGHT), d = random_limbs(TIGHT);
        ref::fe ra = to_ref(a), rb = to_ref(b), rc = to_ref(c), rd = to_ref(d);

        f29::fe r;
        f29::mul(r, a, b);
        expect(same(r, ref::fe_mul(ra, rb)) && tight(r), "mul", i);
        f29::mul(r, b, a);  // operand order must not matter for the bounds we allow
        expect(same(r, ref::fe_mul(ra, rb)) && tight(r), "mul swapped", i);
        f29::sqr(r, a);
        expect(same(r, ref::fe_sq(ra)) && tight(r), "sqr", i);
        f29::ip(r, a, c, d, a);
        expect(same(r, ref::fe_add(ref::fe_mul(ra, rc), ref::fe_mul(rd, ra))) && tight(r), "ip", i);
        f29::add(r, a, c);
        expect(same(r, ref::fe_add(ra, rc)), "add", i);
        f29::sub(r, a, c);
        expect(same(r, ref::fe_sub(ra, rc)), "sub", i);
        for (int k = 0; k < 9; k++) expect(r.l[k] < LOOSE, "sub bound", i);
        f29::neg(r, c);
        expect(same(r, ref::fe_neg(rc)), "neg", i);

        // canon: any limbs below 2^32.
        f29::fe w = random_limbs(1ULL << 32), cw;
        f29::canon(cw, w);
        uint8_t s1[32], s2[32];
        ref::fe_tobytes(s1, to_ref(w));
        f29::to_bytes(s2, w);
        bool ok = true;
        for (int k = 0; k < 32; k++) ok = ok && s1[k] == s2[k];
        for (int k = 0; k < 9; k++) ok = ok && cw.l[k] < (1u << 29);
        expect(ok && cw.l[8] < (1u << 23), "canon", i);

        // The filter: exact unless it says it is unsure (about 2^-15 of calls).
        uint64_t words[4];
        uint32_t low32 = 0;
        bool unsure;
        // Signed limbs: c - d without the bias, and back.
        f29::fe D, U;
        f29::dif(D, c, d);
        unsure = f29::leaf32(a, D, low32);
        ref::fe_words(words, ref::fe_mul(ra, ref::fe_sub(rc, rd)));
        expect(unsure || (uint32_t)words[0] == low32, "leaf32 dif", i);
        f29::add(U, c, d);  // the A+ operand: unsigned, up to 2^30 per limb
        unsure = f29::leaf32(a, U, low32);
        ref::fe_words(words, ref::fe_mul(ra, ref::fe_add(rc, rd)));
        expect(unsure || (uint32_t)words[0] == low32, "leaf32 sum", i);
        f29::undif(U, D);
        expect(same(U, ref::fe_sub(rc, rd)), "undif", i);
        for (int k = 0; k < 9; k++) expect(U.l[k] < LOOSE, "undif bound", i);
    }
    // A large sample of the filter alone (never wrong when sure, rarely
    // unsure): sums and differences of tight values, half of them with every
    // limb at an extreme.
    {
        int unsure = 0, wrong = 0;
        const int M = 4000000;
        for (int i = 0; i < M; i++) {
            f29::fe a = random_limbs(TIGHT), c = random_limbs(TIGHT), d = random_limbs(TIGHT);
            if (i & 1) {
                const int mode = (i >> 1) % 4;  // c - d all positive, all negative, mixed; c + d largest
                for (int k = 0; k < 9; k++) {
                    a.l[k] = (uint32_t)(TIGHT - 1 - (rng() & 7));
                    const uint32_t hi = (uint32_t)(TIGHT - 1 - (rng() & 0xffff)), lo = (uint32_t)(rng() & 0xffff);
                    const bool pos = mode == 0 || (mode == 2 && (rng() & 1));
                    c.l[k] = pos || mode == 3 ? hi : lo;
                    d.l[k] = pos ? lo : hi;
                }
            }
            // Even i: the difference (A-); odd i: the sum (A+), largest limbs.
            f29::fe D;
            const bool sum = (i & 1) ? (i >> 1) % 4 == 3 : (i >> 1) & 1;
            if (sum) f29::add(D, c, d);
            else f29::dif(D, c, d);
            uint32_t low32;
            if (f29::leaf32(a, D, low32)) { unsure++; continue; }
            uint64_t words[4];
            const ref::fe rcd = sum ? ref::fe_add(to_ref(c), to_ref(d)) : ref::fe_sub(to_ref(c), to_ref(d));
            ref::fe_words(words, ref::fe_mul(to_ref(a), rcd));
            wrong += (uint32_t)words[0] != low32;
        }
        std::printf("leaf32: %d of %d unsure (expect ~%d), %d wrong\n", unsure, M,
                    (int)(M / 32768.0), wrong);
        expect(wrong == 0, "leaf32 sample", 0);
    }
    // Shared field vectors through the f29 path.
    int vi = 0;
    for (const char *op : {"mul", "sqr", "add", "sub"}) {
        for (auto &cs : read_cases("../../vectors/field.txt", op)) {
            uint8_t sa[32], sb[32];
            ref::hex_decode(cs[1], sa, 32);
            if (cs[2] != "-") ref::hex_decode(cs[2], sb, 32);
            // Vectors may hold non-canonical inputs: reduce through ref first.
            f29::fe a = from_ref(ref::fe_frombytes(sa));
            f29::fe b = cs[2] == "-" ? a : from_ref(ref::fe_frombytes(sb));
            f29::fe r;
            std::string o = op;
            if (o == "mul") f29::mul(r, a, b);
            else if (o == "sqr") f29::sqr(r, a);
            else if (o == "add") f29::add(r, a, b);
            else f29::sub(r, a, b);
            uint8_t out[32];
            f29::to_bytes(out, r);
            uint8_t want[32];
            ref::hex_decode(cs[3], want, 32);
            bool ok = true;
            for (int k = 0; k < 32; k++) ok = ok && out[k] == want[k];
            expect(ok, op, vi++);
        }
    }
    std::printf("fe29_test: %d passed, %d failed\n", passes, failures);
    return failures ? 1 : 0;
}
