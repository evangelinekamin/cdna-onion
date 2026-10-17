// Checks the host reference arithmetic (src/ref25519.h) against the shared vectors.
// Build and run: make -C entries/mine test-ref
#include <cstdio>
#include <cstdlib>
#include <string>

#include "../src/ref25519.h"
#include "../src/vectors.h"

using namespace ref;

static int failures = 0, passes = 0;
static void expect(bool ok, const std::string &what) {
    if (ok) {
        passes++;
    } else {
        failures++;
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    }
}

static std::string hex(const uint8_t *b, int n) {
    static const char *d = "0123456789abcdef";
    std::string s;
    for (int i = 0; i < n; i++) {
        s += d[b[i] >> 4];
        s += d[b[i] & 15];
    }
    return s;
}

static fe fe_hex(const std::string &h) {
    uint8_t b[32];
    if (!hex_decode(h, b, 32)) std::abort();
    return fe_frombytes(b);
}

static std::string fe_str(const fe &a) {
    uint8_t b[32];
    fe_tobytes(b, a);
    return hex(b, 32);
}

static ge ge_hex(const std::string &h) {
    uint8_t b[32];
    ge p;
    if (!hex_decode(h, b, 32) || !ge_frombytes(p, b)) std::abort();
    return p;
}

static std::string ge_str(const ge &p) {
    uint8_t b[32];
    ge_tobytes(b, p);
    return hex(b, 32);
}

int main(int argc, char **argv) {
    const std::string dir = argc > 1 ? argv[1] : "../../vectors";
    for (const char *op : {"add", "sub", "mul", "sqr", "neg", "inv", "frombytes"}) {
        for (auto &c : read_cases(dir + "/field.txt", op)) {
            fe a = fe_hex(c[1]);
            fe b = c[2] == "-" ? fe_small(0) : fe_hex(c[2]);
            fe r;
            std::string o = op;
            if (o == "add") r = fe_add(a, b);
            else if (o == "sub") r = fe_sub(a, b);
            else if (o == "mul") r = fe_mul(a, b);
            else if (o == "sqr") r = fe_sq(a);
            else if (o == "neg") r = fe_neg(a);
            else if (o == "inv") r = fe_inv(a);
            else r = a;
            expect(fe_str(r) == c[3], o + " " + c[1] + " " + c[2]);
        }
    }
    const ge &B = base_point();
    for (auto &c : read_cases(dir + "/point.txt", "scalarmult_base")) {
        // Decimal scalar up to 2^256: little-endian bytes, then double-and-add.
        uint8_t k[33] = {0};
        for (char ch : c[1]) {
            int carry = ch - '0';
            for (int i = 0; i < 33; i++) {
                int v = k[i] * 10 + carry;
                k[i] = (uint8_t)v;
                carry = v >> 8;
            }
        }
        ge r = ge_identity();
        for (int bit = 263; bit >= 0; bit--) {
            r = ge_add(r, r);
            if ((k[bit / 8] >> (bit % 8)) & 1) r = ge_add(r, B);
        }
        expect(ge_str(r) == c[2], "scalarmult_base " + c[1]);
    }
    for (auto &c : read_cases(dir + "/point.txt", "add"))
        expect(ge_str(ge_add(ge_hex(c[1]), ge_hex(c[2]))) == c[3], "add");
    for (auto &c : read_cases(dir + "/point.txt", "sub"))
        expect(ge_str(ge_add(ge_hex(c[1]), ge_neg(ge_hex(c[2])))) == c[3], "sub");
    for (auto &c : read_cases(dir + "/point.txt", "double"))
        expect(ge_str(ge_add(ge_hex(c[1]), ge_hex(c[1]))) == c[3], "double");
    for (auto &c : read_cases(dir + "/decode.txt", "decode")) {
        uint8_t b[32];
        hex_decode(c[1], b, 32);
        ge p;
        bool ok = ge_frombytes(p, b);
        if (c[2] == "invalid") expect(!ok, "decode rejects " + c[1]);
        else expect(ok && ge_str(p) == c[2], "decode " + c[1]);
    }
    const ge B8 = ge_scalarmult(B, 8);
    for (auto &c : read_cases(dir + "/offset.txt", "offset")) {
        ge a0 = ge_hex(c[1]);
        long long k = std::stoll(c[2]);
        ge step = ge_scalarmult(B8, (unsigned __int128)(k < 0 ? -(__int128)k : k));
        if (k < 0) step = ge_neg(step);
        expect(ge_str(ge_add(a0, step)) == c[3], "offset " + c[2]);
    }
    for (auto &c : read_cases(dir + "/chain.txt", "chain")) {
        ge a0 = ge_hex(c[1]);
        int i = std::stoi(c[2]);
        expect(ge_str(ge_add(a0, ge_scalarmult(B8, (unsigned)i))) == c[3], "chain " + c[2]);
    }
    std::printf("ref_test: %d passed, %d failed\n", passes, failures);
    return failures ? 1 : 0;
}
