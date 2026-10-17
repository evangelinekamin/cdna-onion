// Target prefixes: validation, the GPU's 64-bit fast-path masks, and the exact
// host-side check that every flagged offset passes before it is reported.
#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include "ref25519.h"

namespace targets {

static const int MAX_TARGETS = 256;
static const int MAX_CHARS = 49;
static const int FAST_CHARS = 12;  // 60 bits: what the GPU filter compares

inline int alphabet_index(char c) {
    if (c >= 'a' && c <= 'z') return c - 'a';
    if (c >= '2' && c <= '7') return 26 + (c - '2');
    return -1;
}

struct Target {
    std::string text;
    std::vector<uint8_t> bits;  // stream bits, MSB-first within each character
    uint64_t ymask = 0, yval = 0;  // fast path, over y's low 64 bits
};

// Stream bit s (MSB-first over pubkey bytes) is bit 8*(s/8) + 7 - s%8 of y's
// little-endian low 64 bits.
inline int ybit(int s) { return 8 * (s / 8) + 7 - (s % 8); }

// "" on success, else what is wrong.
inline std::string parse(const std::vector<std::string> &texts, std::vector<Target> &out) {
    if (texts.empty() || (int)texts.size() > MAX_TARGETS) return "need 1 to 256 targets";
    out.clear();
    for (const std::string &t : texts) {
        if (t.empty() || (int)t.size() > MAX_CHARS) return "target length must be 1 to 49";
        Target g;
        g.text = t;
        for (char c : t) {
            int v = alphabet_index(c);
            if (v < 0) return "target has a character outside a-z2-7";
            for (int b = 4; b >= 0; b--) g.bits.push_back((uint8_t)((v >> b) & 1));
        }
        const int fast = (int)std::min<size_t>(g.bits.size(), 5 * FAST_CHARS);
        for (int s = 0; s < fast; s++) {
            g.ymask |= 1ULL << ybit(s);
            if (g.bits[s]) g.yval |= 1ULL << ybit(s);
        }
        out.push_back(g);
    }
    return "";
}

inline bool matches(const uint8_t pub[32], const Target &t) {
    for (size_t s = 0; s < t.bits.size(); s++)
        if (((pub[s / 8] >> (7 - s % 8)) & 1) != t.bits[s]) return false;
    return true;
}

// Exact re-derivation of candidate k: A0 + k*B8, via a table of 2^i * B8.
class Checker {
  public:
    Checker() {
        ref::ge p = ref::ge_scalarmult(ref::base_point(), 8);
        for (int i = 0; i < 64; i++) {
            pow2_[i] = p;
            p = ref::ge_add(p, p);
        }
    }

    void set_start(const ref::ge &a0) { a0_ = a0; }

    void point(ref::ge &out, int64_t k) const {
        const uint64_t mag = k < 0 ? (uint64_t)0 - (uint64_t)k : (uint64_t)k;
        ref::ge s = ref::ge_identity();
        for (int i = 0; i < 64; i++)
            if ((mag >> i) & 1) s = ref::ge_add(s, pow2_[i]);
        out = ref::ge_add(a0_, k < 0 ? ref::ge_neg(s) : s);
    }

    void pubkey(uint8_t pub[32], int64_t k) const {
        ref::ge p;
        point(p, k);
        ref::ge_tobytes(pub, p);
    }

    bool hit(int64_t k, const std::vector<Target> &ts) const {
        uint8_t pub[32];
        pubkey(pub, k);
        for (const Target &t : ts)
            if (matches(pub, t)) return true;
        return false;
    }

  private:
    ref::ge pow2_[64];
    ref::ge a0_ = ref::ge_identity();
};

}  // namespace targets
