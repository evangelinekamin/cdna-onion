// CPU emulation of the GPU walk (src/walk.h compiled for the host): runs a few
// lanes for a few batches and checks every candidate against the reference.
// Each tested k must appear exactly once, the k's must tile a contiguous range,
// and the hot path's 32-bit value must equal the low 32 bits of y.
#include <cstdio>
#include <cstdlib>
#include <map>
#include <random>
#include <vector>

#include "../src/tables.h"
#include "../src/walk.h"

static int failures = 0;

struct VecTab {
    const std::vector<walk::Entry> *t;
    const walk::Entry &operator()(int j) const { return (*t)[j]; }
};

struct VecMem {
    std::vector<f29::fe> v;
    f29::fe nl;
    void store(int j, const f29::fe &x) { v[j] = x; }
    f29::fe load(int j) const { return v[j]; }
    void set_nlam(const f29::fe &x) { nl = x; }
    f29::fe nlam() const { return nl; }
};

struct MapSink {
    std::map<int64_t, uint64_t> *seen;
    int64_t center = 0;
    std::vector<uint32_t> pending;  // hot-path 32-bit values, in call order
    std::map<int64_t, uint32_t> low32;
    int duplicates = 0;
    void test(uint64_t y, int64_t d) {
        const int64_t k = center + d;
        if (!seen->emplace(k, y).second) duplicates++;
        if (d != 0 && !pending.empty()) {  // the 32-bit value that led here
            low32[k] = pending.front();
            pending.erase(pending.begin());
        }
    }
    void advance(int64_t step) { center += step; }
    void mark(int) {}
    // Records the hot path's 32-bit value and always asks for the exact one, so
    // both paths are checked for every candidate.
    bool maybe32(uint32_t y) {
        pending.push_back(y);
        return true;
    }
};

template <int H>
static void run(uint64_t seed, int lanes, int batches, int64_t kbase) {
    std::mt19937_64 rng(seed);
    // A random start point A0 = a*B.
    unsigned __int128 a = ((unsigned __int128)rng() << 64) | rng();
    const ref::ge A0 = ref::ge_scalarmult(ref::base_point(), a);
    const std::vector<walk::Entry> table = tables::build_table(H);
    const int64_t S = 2 * H + 1;
    const walk::Giant G = tables::giant((int64_t)lanes * S);
    std::map<int64_t, uint64_t> seen;
    MapSink sink{&seen, 0, {}, {}, 0};
    for (int l = 0; l < lanes; l++) {
        walk::Lane lane;
        const int64_t k0 = kbase + l * S;
        if (!tables::lane_at(lane, tables::offset(A0, k0))) std::abort();
        sink.center = k0;
        VecMem mem{std::vector<f29::fe>(H + 4), {}};
        for (int b = 0; b < batches; b++) {
            if (!walk::batch<H>(lane, G, (int64_t)lanes * S, VecTab{&table}, mem, sink)) {
                std::fprintf(stderr, "FAIL: degenerate batch\n");
                failures++;
            }
        }
    }
    // Every k in [kbase - H, kbase - H + lanes*batches*S) exactly once.
    const int64_t lo = kbase - H, n = (int64_t)lanes * batches * S;
    int bad = 0;
    if ((int64_t)seen.size() != n || seen.begin()->first != lo || seen.rbegin()->first != lo + n - 1) {
        std::fprintf(stderr, "FAIL: tested %zu offsets, want %lld from %lld\n", seen.size(),
                     (long long)n, (long long)lo);
        failures++;
    }
    ref::ge P = tables::offset(A0, lo);
    for (int64_t k = lo; k < lo + n; k++, P = ref::ge_add(P, tables::B8())) {
        auto it = seen.find(k);
        if (it == seen.end()) continue;
        ref::fe x, y;
        ref::ge_affine(x, y, P);
        uint64_t w[4];
        ref::fe_words(w, y);
        bad += w[0] != it->second;
        auto l32 = sink.low32.find(k);
        if (l32 != sink.low32.end()) bad += (uint32_t)w[0] != l32->second;
    }
    if (bad || sink.duplicates) {
        std::fprintf(stderr, "FAIL: H=%d seed=%llu: %d wrong, %d duplicates\n", H,
                     (unsigned long long)seed, bad, sink.duplicates);
        failures++;
    }
    std::printf("H=%d lanes=%d batches=%d kbase=%lld: %zu candidates checked\n", H, lanes, batches,
                (long long)kbase, seen.size());
}

int main() {
    run<8>(1, 3, 3, 0);
    run<8>(2, 2, 2, -1000);
    run<64>(3, 4, 3, 123456789);
    run<256>(4, 2, 2, -77);
    run<1024>(5, 1, 2, 5);
    run<2048>(9, 1, 3, -3);
    std::printf(failures ? "walk_test: FAILED\n" : "walk_test: all passed\n");
    return failures ? 1 : 0;
}
