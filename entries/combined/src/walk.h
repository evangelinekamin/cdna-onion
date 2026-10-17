// One lane's batch of the walk: the center C and the 2H points C + j*B8 and
// C - j*B8 (j = 1..H), all tested with one field inversion, plus the giant
// step to the lane's next center. Host and device compile the same code, so the
// CPU emulation test (test/walk_test.cpp) checks exactly what the GPU runs.
//
// The algebra (y-only, dual addition law, renormalized so every denominator
// is free):
//   With C = (x1, y1) affine, T_j = (x2, y2) = j*B8, the dual addition law gives
//     y(C + T_j) = (x1 y1 - x2 y2) / (x1 y2 - y1 x2)
//     y(C - T_j) = (x1 y1 + x2 y2) / (x1 y2 + y1 x2).
//   Divide through by x1 x2, with mu = 1/x1, lam = y1/x1, r = 1/x2, s = y2/x2:
//     y(C +- T_j) = (y1 r -+ mu y2) / (s -+ lam).
//   The two denominators multiply to q_j = s^2 - lam^2: a table constant minus
//   a per-batch constant, no multiplication. Bring both over q_j:
//     y(C +- T_j) = (alpha_j +- beta_j) / q_j,
//     alpha_j = y1 (r s) - (mu lam) y2,   beta_j = (lam y1) r - mu (y2 s).
//   Divide both by mu and let the inverses carry it instead:
//     y(C +- T_j) = (alpha'_j +- beta'_j) mu / q_j,
//     alpha'_j = (x1 y1)(r s) - lam y2,   beta'_j = y1^2 r - (y2 s),
//   where y2 s is a table constant, so beta' is one limb product plus an
//   addend: three products per pair instead of four. (Gemini's round-1 entry
//   normalized its numerators to leave a table constant the same way.) The
//   factor mu costs two multiplies per batch: the forward chain starts at
//   W mu and the inverse is multiplied by mu, so every backward inverse comes
//   out as mu / q_j and the pass ends at exactly 1/W, as the giant step needs.
//   Per pair of candidates: one forward multiply (prefix product of the q's),
//   two backward (Montgomery's trick), the two numerators, and two partial
//   "leaf" products that compute only what the prefix filter needs.
//   The forward pass stores one field element per pair (CKPT=1) or per two
//   pairs (CKPT=2, the default).
#pragma once
#include "fe29.h"

// Prefix products stored every CKPT steps (1 or 2). The scratch traffic costs
// clock (the card is power limited), so storing every other one pays for the
// extra multiply that rebuilds the inverses of a step pair from it.
#ifndef CKPT
#define CKPT 2
#endif

namespace walk {

using f29::fe;

// Per-j table constants (canonical), for T_j = j*B8 = (x2, y2).
struct Entry {
    fe sigma;  // (y2/x2)^2
    fe rs;     // y2/x2^2
    fe ny2;    // -y2
    fe r;      // 1/x2
    fe ny2s;   // -y2^2/x2
};

// The giant step G between a lane's consecutive centers, affine, with
// dg = d * gx * gy for the addition law.
struct Giant {
    fe gx, gy, dg;
};

// A lane between batches: the center's affine coordinates (tight) and 1/x.
// The center's offset k lives with the sink (Sink::test reports offsets
// relative to it), so it holds no registers through the hot loops.
struct Lane {
    fe x, y, mu;
};

// r = 1/a (0 for a = 0): a^(2^255 - 21), 254 squarings and 11 multiplies.
HD void sqr_n(fe &r, const fe &a, int n) {
    f29::sqr(r, a);
#if defined(__HIP_DEVICE_COMPILE__)
#pragma unroll 1
#endif
    for (int i = 1; i < n; i++) f29::sqr(r, r);
}

HD void inv(fe &r, const fe &z) {
    fe z2, z9, z11, t, z5_0, z10_0, z20_0, z50_0, z100_0;
    f29::sqr(z2, z);
    sqr_n(t, z2, 2);
    f29::mul(z9, t, z);
    f29::mul(z11, z9, z2);
    f29::sqr(t, z11);
    f29::mul(z5_0, t, z9);
    sqr_n(t, z5_0, 5);
    f29::mul(z10_0, t, z5_0);
    sqr_n(t, z10_0, 10);
    f29::mul(z20_0, t, z10_0);
    sqr_n(t, z20_0, 20);
    f29::mul(t, t, z20_0);
    sqr_n(t, t, 10);
    f29::mul(z50_0, t, z10_0);
    sqr_n(t, z50_0, 50);
    f29::mul(z100_0, t, z50_0);
    sqr_n(t, z100_0, 100);
    f29::mul(t, t, z100_0);
    sqr_n(t, t, 50);
    f29::mul(t, t, z50_0);
    sqr_n(t, t, 5);
    f29::mul(r, t, z11);
}

// Both candidates C +- T_j of pair j, given invq = mu/q_j: the two
// numerators, the two partial "leaf" products, the target pre-test, and
// (rarely) the full multiply that settles a candidate exactly. w1 = x1 y1,
// lam = y1/x1 and y1sq = y1^2 are the batch's numerator factors.
template <int H, class Sink>
HD void pair(const fe &w1, const fe &lam, const fe &y1sq, const Entry &t, const fe &invq, int j,
             Sink &sink) {
    fe alpha, beta, Ap, Am;
    f29::ip_k<true>(alpha, w1, t.rs, lam, t.ny2);   // x1 y1 (r s) - lam y2
    f29::mul_add_k<true>(beta, y1sq, t.r, t.ny2s);  // y1^2 r - y2 s
    f29::add(Ap, alpha, beta);
    f29::dif(Am, alpha, beta);  // signed limbs, as leaf32 takes them: no 128p bias
    // Hot path: y's low 32 bits. Only candidates that pass the 32-bit test
    // (or that leaf32 cannot decide, about one in 2^15) take the full
    // multiply, which settles longer targets exactly.
    uint32_t yp32, ym32;
    const bool up = f29::leaf32(invq, Ap, yp32), um = f29::leaf32(invq, Am, ym32);
    // Both tests first, then one (rarely taken) branch: a second branch
    // would make the compiler carry the second test's results across the
    // first as 0/1 values in VGPRs.
    const bool rp = up | sink.maybe32(yp32), rm = um | sink.maybe32(ym32);
    if (rp | rm) {
        if (rp) {
            fe y;
            f29::mul(y, invq, Ap);
            f29::canon(y, y);
            sink.test(f29::low64(y), j);
        }
        if (rm) {
            fe y;
            f29::undif(Am, Am);
            f29::mul(y, invq, Am);
            f29::canon(y, y);
            sink.test(f29::low64(y), -j);
        }
    }
}

// Scratch slot of the giant step's four factors (after the prefix products).
#define GSLOT(H) ((H) / CKPT)

// What a batch needs from outside: the table (Tab(j) -> const Entry&, j = 1..H),
// per-lane scratch (Mem: store(j, fe) and load(j) for j = 0..H+3, plus
// set_nlam/nlam to park the per-batch -lambda^2 where it holds no registers), and the
// filter (Sink: test(y64, d) with the exact low 64 bits of the y of the
// candidate at offset center + d; maybe32(y32), a pre-test on the low 32 bits
// that must never reject a match; advance(step) once per batch to move the
// center; and mark(phase), a profiling hook).
// Returns false if the batch was degenerate (a zero
// denominator: only possible when A0 is a small known multiple of B8, plus
// torsion); its results are then meaningless, and every flagged candidate is
// re-derived exactly on the host anyway.
template <int H, class Tab, class Mem, class Sink>
HD bool batch(Lane &L, const Giant &G, int64_t step, const Tab &tab, Mem &mem, Sink &sink) {
    sink.mark(0);  // phase 0: batch setup and giant-step numerators
    // The center itself, exactly. Tested first, so y1 holds no registers
    // through the loops.
    fe yc;
    f29::canon(yc, L.y);
    sink.test(f29::low64(yc), 0);

    fe lam, Lam, nLam, w1, y1sq;
    f29::mul(lam, L.y, L.mu);
    f29::sqr(Lam, lam);
    f29::neg(nLam, Lam);  // q_j = sigma_j + nLam
    mem.set_nlam(nLam);   // parked outside the register file (LDS on the GPU)
    f29::mul(w1, L.x, L.y);
    f29::sqr(y1sq, L.y);

    // Giant step: x' = nX/dX, y' = nY/dY, 1/x' = dX/nX, all from W = dX dY nX.
    fe e, nX, nY, dX, dY, W;
    f29::mul_k<true>(e, w1, G.dg);
    f29::ip_k<true>(nX, L.x, G.gy, L.y, G.gx);
    f29::ip_k<true>(nY, L.y, G.gy, L.x, G.gx);
    dX = e;
    dX.l[0] += 1;
    fe one = {{1, 0, 0, 0, 0, 0, 0, 0, 0}};
    f29::sub(dY, one, e);
    f29::mul(W, dX, dY);
    f29::mul(W, W, nX);

    // The giant step's factors wait in scratch (slots H..H+3) until the end,
    // so they hold no registers through the loops.
    mem.store(GSLOT(H), dX);
    mem.store(GSLOT(H) + 1, dY);
    mem.store(GSLOT(H) + 2, nX);
    mem.store(GSLOT(H) + 3, nY);

    sink.mark(1);
    fe P;
    f29::mul(P, W, L.mu);  // the chain starts at W mu (see the algebra above)
    // Forward: P_j = W mu q_1 ... q_j; store P_{j-CKPT} for j = CKPT, 2CKPT, ...
#if defined(__HIP_DEVICE_COMPILE__)
#pragma unroll 1
#endif
    for (int j = CKPT; j <= H; j += CKPT) {
        mem.store(j / CKPT - 1, P);
#pragma unroll
        for (int i = CKPT - 1; i >= 0; i--) {
            fe q;
            f29::add(q, tab(j - i).sigma, mem.nlam());
            f29::mul(P, P, q);
        }
    }
    sink.mark(2);
    fe I, Pc;
    inv(I, P);
    f29::canon(Pc, P);
    const bool ok = !f29::is_zero_canon(Pc);
    f29::mul(I, I, L.mu);  // I = mu/P_H, so the backward pass yields mu/q_j

    sink.mark(3);
#if CKPT == 1
    // Backward: I = mu/P_j on entry to step j. P_{j-1} is loaded one step
    // ahead so its memory latency overlaps a whole step of arithmetic.
    fe Pnext = mem.load(H - 1);
#if defined(__HIP_DEVICE_COMPILE__)
#pragma unroll 1
#endif
    for (int j = H; j >= 1; j--) {
        const Entry &t = tab(j);
        const fe Pj1 = Pnext;
        Pnext = mem.load(j >= 2 ? j - 2 : 0);
        fe invq, q;
        f29::mul(invq, Pj1, I);  // mu/q_j
        f29::add(q, t.sigma, mem.nlam());
        f29::mul(I, I, q);
        pair<H>(w1, lam, y1sq, t, invq, j, sink);
    }
#else
    static_assert(CKPT == 2, "CKPT must be 1 or 2");
    // Backward, two steps at a time from P_{j-2} (j even), I = mu/P_j:
    //   A = P_{j-2} I = mu/(q_{j-1} q_j), mu/q_j = A q_{j-1}, mu/q_{j-1} = A q_j,
    //   mu/P_{j-2} = I q_j q_{j-1}: five multiplies per two steps (four in the
    //   one-step loop) for half the scratch traffic.
    fe Pnext = mem.load(H / 2 - 1);
#if defined(__HIP_DEVICE_COMPILE__)
#pragma unroll 1
#endif
    for (int j = H; j >= 2; j -= 2) {
        const fe Pj2 = Pnext;
        Pnext = mem.load(j >= 4 ? j / 2 - 2 : 0);
        fe A, q0, q1, inv0, inv1;
        f29::mul(A, Pj2, I);
        f29::add(q0, tab(j).sigma, mem.nlam());
        f29::add(q1, tab(j - 1).sigma, mem.nlam());
        f29::mul(inv0, A, q1);  // mu/q_j
        f29::mul(inv1, A, q0);  // mu/q_{j-1}
        f29::mul(I, I, q0);
        f29::mul(I, I, q1);
        pair<H>(w1, lam, y1sq, tab(j), inv0, j, sink);
        pair<H>(w1, lam, y1sq, tab(j - 1), inv1, j - 1, sink);
    }
#endif

    sink.mark(4);
    // Now I = mu/P_0 = 1/W. Recover the next center.
    dX = mem.load(GSLOT(H));
    dY = mem.load(GSLOT(H) + 1);
    nX = mem.load(GSLOT(H) + 2);
    nY = mem.load(GSLOT(H) + 3);
    fe u, v, t1;
    f29::mul(u, dY, I);        // 1/(dX nX)
    f29::mul(t1, nX, u);       // 1/dX
    f29::mul(L.x, nX, t1);     // x' = nX/dX
    f29::mul(v, nX, I);        // 1/(dX dY)
    f29::mul(t1, dX, v);       // 1/dY
    f29::mul(L.y, nY, t1);     // y' = nY/dY
    f29::mul(t1, dX, u);       // 1/nX
    f29::mul(L.mu, dX, t1);    // 1/x' = dX/nX
    sink.advance(step);
    sink.mark(5);
    return ok;
}

}  // namespace walk
