// onion-walk: a Tor v3 vanity-address grinder for AMD CDNA (onionrig-grind/1).
//
// Each GPU lane walks its own sequence of centers C, testing C and C +- j*B8
// (j = 1..H) per batch with a single field inversion; see walk.h for the
// algebra and fe29.h for the field arithmetic. The GPU's prefix filter may
// flag a candidate that does not match (rarely); it never misses one. Every
// flagged offset is re-derived on the host with independent arithmetic
// (ref25519.h) before it is reported, so a reported offset is always right.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "fe29.h"
#include "json.h"
#include "ref25519.h"
#include "tables.h"
#include "targets.h"
#include "vectors.h"
#include "walk.h"

#ifndef WALK_H
#define WALK_H 4096
#endif
#ifndef BLOCK
#define BLOCK 64  // one wave per block: thread t's LDS and scratch offsets coincide
#endif

static const char *CONTRACT = "onionrig-grind/1";
static const char *NAME = "onion-walk";
static const char *VERSION = "0.6";
static const char *MODEL = "Claude Opus 5.5 (claude-opus-5-5)";

static constexpr int H = WALK_H;
static constexpr int64_t SPAN = 2 * H + 1;  // candidates per lane per batch
static constexpr uint32_t HIT_CAP = 1u << 20;
// Target duration of one kernel launch. Longer launches shrink the share of
// each launch's tail (the last waves to finish run with fewer neighbours);
// shorter ones bound stop latency and hit-flush delay (contract: about 1 s).
#ifndef LAUNCH_SECONDS
#define LAUNCH_SECONDS 0.3
#endif

using f29::fe;

// ---------------------------------------------------------------------------
// Device side
// ---------------------------------------------------------------------------
__constant__ int c_nt;
__constant__ uint32_t c_keys[targets::MAX_TARGETS];  // the general form's sorted masked values
__constant__ uint64_t c_ymask[targets::MAX_TARGETS];
__constant__ uint64_t c_yval[targets::MAX_TARGETS];

struct DevTab {
    const walk::Entry *__restrict__ t;
    __device__ const walk::Entry &operator()(int j) const { return t[j]; }
};

// Per-lane scratch through buffer resources: element j limb i of lane l
// (wave w = l / 64, t = l % 64) at dword [((j * waves + w) * 9 + i) * 64 + t].
// Each access is one coalesced 256-byte row per wave; the wave's part of the
// address is in the (scalar) resource base, the limb offsets (i * 256 bytes)
// are instruction immediates, and the only VGPR operand is 4t, which with
// one wave per block is also the LDS address of the thread's nlam limb 0: one
// VGPR for all addressing. (The layout measured against a flat limb-major
// one, a [j][lane][9] one with 16-byte accesses, and two 16-byte groups plus
// one dword group: this one or equal, see NOTES.md.)
static constexpr int SCRATCH_ELEMENTS = GSLOT(H) + 4;  // per lane: checkpoints, giant-step factors

struct DevMem {
    uint32_t *__restrict__ base;
    uint32_t lanes, wave;  // wave = lane / 64, wave-uniform
    uint32_t *lds;         // this block's LDS: nlam limb i of thread t at [i * BLOCK + t]
    __device__ void set_nlam(const fe &x) const {
#pragma unroll
        for (int i = 0; i < 9; i++) lds[i * BLOCK + threadIdx.x] = x.l[i];
    }
    __device__ fe nlam() const {
        fe x;
#pragma unroll
        for (int i = 0; i < 9; i++) x.l[i] = lds[i * BLOCK + threadIdx.x];
        return x;
    }
    __device__ __amdgpu_buffer_rsrc_t rsrc(int j) const {
        return __builtin_amdgcn_make_buffer_rsrc(base + (size_t)9 * j * lanes + 576 * wave, 0, 2304,
                                                 0x00020000);
    }
    __device__ uint32_t off() const { return 4 * (threadIdx.x & 63); }
    __device__ void store(int j, const fe &x) const {
        const __amdgpu_buffer_rsrc_t r = rsrc(j);
        // The empty asm stops LLVM from hoisting off + 256*i into nine VGPRs,
        // so the constants land in the instructions' immediate offset field.
        uint32_t o = off();
        asm volatile("" : "+v"(o));
#pragma unroll
        for (int i = 0; i < 9; i++) __builtin_amdgcn_raw_buffer_store_b32(x.l[i], r, o + 256 * i, 0, 0);
    }
    __device__ fe load(int j) const {
        const __amdgpu_buffer_rsrc_t r = rsrc(j);
        uint32_t o = off();
        asm volatile("" : "+v"(o));
        fe x;
#pragma unroll
        for (int i = 0; i < 9; i++) x.l[i] = __builtin_amdgcn_raw_buffer_load_b32(r, o + 256 * i, 0, 0);
        return x;
    }
};

// Targets for the GPU filter. The hot path tests y's low 32 bits under one
// mask (the shortest target's: prefix masks nest, so this never rejects a
// match). The fast form holds up to four target values, padded by repetition,
// in SGPRs (every benchmark). The general form binary-searches the sorted
// masked values (size a power of two, padded with ~0) in LDS: log2(size)
// dependent reads instead of a compare per target. The exact 64-bit test runs
// only on the rare path, from constant memory.
struct FastTargets {
    uint32_t mask, val[4], size;
};

// PROFILE builds read the shader clock at each phase boundary of a batch and
// add the cycles per phase into prof[] (lane 0 of each wave), for development.
#ifndef PROFILE
#define PROFILE 0
#endif
static constexpr int NPHASE = 6;
// prof[NPHASE] and prof[NPHASE+1]: summed s_memtime and s_memrealtime deltas
// over each wave's whole launch, for the effective shader clock.
static constexpr int NPROF = NPHASE + 2;

template <bool FAST>
struct DevSink {
    FastTargets ft;
    const uint32_t *keys;  // general form: this block's LDS copy of c_keys
    unsigned long long *hits;
    uint32_t *count;
    long long *kbase;   // the lanes' center offsets k, kept in memory: a lane's
                        // is needed only on the rare hit path and once per batch
    uint32_t wave;      // lane / 64, wave-uniform
    __device__ long long *center() const { return kbase + 64 * wave + (threadIdx.x & 63); }
#if PROFILE
    // Per-phase cycle sums live in LDS (one row per wave), so profiling holds
    // no registers and needs no scratch; each mark() closes the previous phase.
    unsigned long long *prof;
    unsigned long long *lds_prof;  // this wave's NPHASE sums, start clocks, last mark
    // mark(next) closes phase next - 1 (phase NPHASE - 1 for next = 0): batches
    // call mark(0) .. mark(NPHASE - 1) in order, so no phase state is needed.
    __device__ void mark(int next) const {
        if ((threadIdx.x & 63) == 0) {
            const unsigned long long t = clock64(), last = lds_prof[NPHASE + 2];
            if (last) lds_prof[next == 0 ? NPHASE - 1 : next - 1] += t - last;
            lds_prof[NPHASE + 2] = t;
        }
    }
    __device__ void flush() const {
        if ((threadIdx.x & 63) == 0)
            for (int i = 0; i < NPHASE; i++) atomicAdd(&prof[i], lds_prof[i]);
    }
    __device__ void clocks(unsigned long long dt, unsigned long long dwall) const {
        if ((threadIdx.x & 63) == 0) {
            atomicAdd(&prof[NPHASE], dt);
            atomicAdd(&prof[NPHASE + 1], dwall);
        }
    }
#else
    __device__ void mark(int) const {}
    __device__ void flush() const {}
    __device__ void clocks(unsigned long long, unsigned long long) const {}
#endif
    __device__ void record(int64_t d) const {
        const int64_t k = *center() + d;
        const uint32_t slot = atomicAdd(count, 1u);
        if (slot < HIT_CAP) hits[slot] = (unsigned long long)k;
    }
    // Exact test of y's low 64 bits (the first 12 characters) for the
    // candidate at offset center + d; longer targets are completed on the host.
    __device__ void test(uint64_t y, int64_t d) const {
        bool hit = false;
        for (int i = 0; i < c_nt; i++) hit |= (y & c_ymask[i]) == c_yval[i];
        if (hit) record(d);
    }
    // The low-32-bit pre-test: exact for targets of up to six characters, a
    // pre-filter for longer ones. Never rejects a match.
    __device__ bool maybe32(uint32_t y) const {
        if (FAST) {
            const uint32_t m = y & ft.mask;
            return (m == ft.val[0]) | (m == ft.val[1]) | (m == ft.val[2]) | (m == ft.val[3]);
        }
        // Branchless lower bound over a power-of-two size.
        const uint32_t key = y & ft.mask;
        uint32_t i = 0;
        for (uint32_t s = ft.size >> 1; s; s >>= 1) i += keys[i + s - 1] < key ? s : 0;
        return keys[i] == key;
    }
    __device__ void advance(long long step) const { *center() += step; }
};

// Lane state between launches, limb-major: x, y, mu as fe index 0..2; the
// center offsets k in their own array (the sink reads and advances them).
struct LaneBuf {
    uint32_t *fe_;
    long long *k;
    __device__ void load(walk::Lane &L, uint32_t lanes, uint32_t lane) const {
#pragma unroll
        for (int i = 0; i < 9; i++) {
            L.x.l[i] = fe_[(size_t)(0 * 9 + i) * lanes + lane];
            L.y.l[i] = fe_[(size_t)(1 * 9 + i) * lanes + lane];
            L.mu.l[i] = fe_[(size_t)(2 * 9 + i) * lanes + lane];
        }
    }
    __device__ void store(const walk::Lane &L, uint32_t lanes, uint32_t lane) const {
        // Opaque lane index: otherwise LLVM keeps the 27 load addresses from
        // the kernel's start alive until this store, and spills them to
        // scratch (the kernel's only scratch use).
        asm volatile("" : "+v"(lane));
#pragma unroll
        for (int i = 0; i < 9; i++) {
            fe_[(size_t)(0 * 9 + i) * lanes + lane] = L.x.l[i];
            fe_[(size_t)(1 * 9 + i) * lanes + lane] = L.y.l[i];
            fe_[(size_t)(2 * 9 + i) * lanes + lane] = L.mu.l[i];
        }
    }
};

#ifndef WAVES_PER_EU
#define WAVES_PER_EU 4
#endif
// Grinds run two independent lane sets ("halves"), each sized to fill the GPU,
// on two streams, launched alternately. A launch's waves retire unevenly (the
// SIMDs favor older waves), and the other half's queued launch fills the CUs as
// they do, instead of the GPU draining at every launch boundary.
#ifndef STREAMS
#define STREAMS 2
#endif

template <bool FAST>
__global__ void __launch_bounds__(BLOCK, WAVES_PER_EU)
grind_kernel(LaneBuf st, uint32_t lanes, int nb, const walk::Entry *__restrict__ table,
             const walk::Giant *__restrict__ giant, long long step, uint32_t *__restrict__ pbuf,
             FastTargets ft, unsigned long long *__restrict__ hits, uint32_t *__restrict__ count,
             uint32_t *__restrict__ degenerate, unsigned long long *__restrict__ prof) {
    const uint32_t lane = blockIdx.x * blockDim.x + threadIdx.x;
    if (lane >= lanes) return;
    walk::Lane L;
    st.load(L, lanes, lane);
    DevTab tab{table};
    __shared__ uint32_t lds[9 * BLOCK];
    __shared__ uint32_t keys[FAST ? 1 : targets::MAX_TARGETS];
    if (!FAST) {
        for (uint32_t i = threadIdx.x; i < ft.size; i += BLOCK) keys[i] = c_keys[i];
        __syncthreads();
    }
    const uint32_t wave = __builtin_amdgcn_readfirstlane(lane / 64);
    DevMem mem{pbuf, lanes, wave, lds};
#if PROFILE
    // Per wave: NPHASE phase sums, then the launch's start clocks (in LDS, so
    // they hold no registers through the batches).
    __shared__ unsigned long long lds_prof[BLOCK / 64][NPHASE + 3];
    if ((threadIdx.x & 63) == 0) {
        for (int i = 0; i < NPHASE; i++) lds_prof[threadIdx.x / 64][i] = 0;
        lds_prof[threadIdx.x / 64][NPHASE + 2] = 0;
        lds_prof[threadIdx.x / 64][NPHASE] = clock64();
        lds_prof[threadIdx.x / 64][NPHASE + 1] = wall_clock64();
    }
    DevSink<FAST> sink{ft, keys, hits, count, st.k, wave, prof, lds_prof[threadIdx.x / 64]};
#else
    DevSink<FAST> sink{ft, keys, hits, count, st.k, wave};
#endif
    for (int b = 0; b < nb; b++)
        if (!walk::batch<H>(L, *giant, step, tab, mem, sink)) atomicAdd(degenerate, 1u);
    sink.flush();
#if PROFILE
    sink.clocks(clock64() - lds_prof[threadIdx.x / 64][NPHASE],
                wall_clock64() - lds_prof[threadIdx.x / 64][NPHASE + 1]);
#endif
    st.store(L, lanes, lane);
}

// A point prepared for mixed addition: (y + x, y - x, 2d x y), canonical.
struct Prep {
    fe ypx, ymx, t2d;
};

struct Ext {
    fe X, Y, Z, T;
};

// P += Q (unified addition with Z_Q = 1). Canonicalizes every sum so the
// multiply bounds hold trivially; this runs once per lane per start.
__device__ void madd(Ext &P, const Prep &q) {
    fe a, b, c, d, e, f, g, h, t;
    f29::sub(t, P.Y, P.X);
    f29::canon(t, t);
    f29::mul(a, t, q.ymx);
    f29::add(t, P.Y, P.X);
    f29::canon(t, t);
    f29::mul(b, t, q.ypx);
    f29::mul(c, P.T, q.t2d);
    f29::add(d, P.Z, P.Z);
    f29::canon(d, d);
    f29::sub(e, b, a);
    f29::canon(e, e);
    f29::sub(f, d, c);
    f29::canon(f, f);
    f29::add(g, d, c);
    f29::canon(g, g);
    f29::add(h, b, a);
    f29::canon(h, h);
    f29::mul(P.X, e, f);
    f29::mul(P.Y, g, h);
    f29::mul(P.T, e, h);
    f29::mul(P.Z, f, g);
}

// Lane l starts at center A + l*D (A = A0 + kbase*B8, D = SPAN*B8), using
// steps[i] = 2^i * D: k = kbase + l*SPAN.
__global__ void __launch_bounds__(BLOCK)
init_kernel(LaneBuf st, uint32_t lanes, const Prep *__restrict__ steps, int nsteps, Ext A,
            long long kbase, long long span, uint32_t *__restrict__ dead) {
    const uint32_t lane = blockIdx.x * blockDim.x + threadIdx.x;
    if (lane >= lanes) return;
    Ext P = A;
    for (int i = 0; i < nsteps; i++)
        if ((lane >> i) & 1) madd(P, steps[i]);
    fe xz, inv, iz, ix, Xc;
    f29::mul(xz, P.X, P.Z);
    walk::inv(inv, xz);
    f29::mul(iz, P.X, inv);
    f29::mul(ix, P.Z, inv);
    walk::Lane L;
    f29::mul(L.x, P.X, iz);
    f29::mul(L.y, P.Y, iz);
    f29::mul(L.mu, P.Z, ix);
    f29::canon(L.x, L.x);
    f29::canon(L.y, L.y);
    f29::canon(L.mu, L.mu);
    f29::canon(Xc, P.X);
    if (f29::is_zero_canon(Xc)) atomicAdd(dead, 1u);
    st.store(L, lanes, lane);
    st.k[lane] = kbase + (long long)lane * span;
}

// Field self-test: op 0 mul, 1 sqr, 2 add, 3 sub, 4 inv, 5 ip (a*b + b*a).
__global__ void field_kernel(const fe *a, const fe *b, const int *op, fe *out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    fe r;
    switch (op[i]) {
        case 0: f29::mul(r, a[i], b[i]); break;
        case 1: f29::sqr(r, a[i]); break;
        case 2: f29::add(r, a[i], b[i]); break;
        case 3: f29::sub(r, a[i], b[i]); break;
        case 4: walk::inv(r, a[i]); break;
        default: f29::ip(r, a[i], b[i], b[i], a[i]); break;
    }
    f29::canon(out[i], r);
}

// ---------------------------------------------------------------------------
// Host side
// ---------------------------------------------------------------------------
static bool g_protocol = false;

static void emit(const std::string &line) {
    std::fwrite(line.data(), 1, line.size(), stdout);
    std::fputc('\n', stdout);
    std::fflush(stdout);
}

static std::string json_escape(const std::string &s) {
    std::string o;
    for (char c : s) {
        if (c == '"' || c == '\\') { o += '\\'; o += c; }
        else if ((unsigned char)c < 0x20) o += ' ';
        else o += c;
    }
    return o;
}

static void emit_error(const std::string &message) {
    emit("{\"event\": \"error\", \"message\": \"" + json_escape(message) + "\"}");
}

[[noreturn]] static void fatal(const std::string &what) {
    if (g_protocol) emit_error(what);
    std::fprintf(stderr, "fatal: %s\n", what.c_str());
    std::exit(3);
}

#define CHECK(call)                                                                       \
    do {                                                                                  \
        hipError_t e_ = (call);                                                           \
        if (e_ != hipSuccess)                                                             \
            fatal(std::string(#call) + " at line " + std::to_string(__LINE__) + ": " +    \
                  hipGetErrorString(e_));                                                 \
    } while (0)

using Clock = std::chrono::steady_clock;
static double seconds_since(Clock::time_point t) {
    return std::chrono::duration<double>(Clock::now() - t).count();
}

static std::string device_name() {
    hipDeviceProp_t prop;
    if (hipGetDeviceProperties(&prop, 0) != hipSuccess) return "unknown";
    return std::string(prop.name) + " (" + prop.gcnArchName + ", " +
           std::to_string(prop.multiProcessorCount) + " CUs)";
}

static Prep prep(const ref::ge &P) {
    ref::fe x, y;
    ref::ge_affine(x, y, P);
    Prep q;
    q.ypx = tables::to29(ref::fe_add(y, x));
    q.ymx = tables::to29(ref::fe_sub(y, x));
    q.t2d = tables::to29(ref::fe_mul(ref::consts().d2, ref::fe_mul(x, y)));
    return q;
}

static Ext ext29(const ref::ge &P) {
    Ext e;
    e.X = tables::to29(P.X);
    e.Y = tables::to29(P.Y);
    e.Z = tables::to29(P.Z);
    e.T = tables::to29(P.T);
    return e;
}

class Engine {
  public:
    uint32_t max_lanes = 0;
    double kernel_seconds = 0;  // PROFILE builds: summed launch durations from events

    void init() {
        CHECK(hipSetDevice(0));
        hipDeviceProp_t prop;
        CHECK(hipGetDeviceProperties(&prop, 0));
        int blocks_per_cu = 0;
        CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_cu, grind_kernel<true>, BLOCK, 0));
        if (blocks_per_cu < 1) blocks_per_cu = 1;
        max_lanes = (uint32_t)prop.multiProcessorCount * blocks_per_cu * BLOCK;
        std::fprintf(stderr, "device %s: %d blocks of %d per CU -> %u lanes, H=%d\n",
                     device_name().c_str(), blocks_per_cu, BLOCK, max_lanes, H);

        const std::vector<walk::Entry> table = tables::build_table(H);
        CHECK(hipMalloc(&d_table_, sizeof(walk::Entry) * table.size()));
        CHECK(hipMemcpy(d_table_, table.data(), sizeof(walk::Entry) * table.size(),
                        hipMemcpyHostToDevice));
        // Each lane set's scratch is ~7.9 GB at H=4096; a GPU without room
        // for the second runs one set (no launch overlap) rather than fail.
        const size_t elements = SCRATCH_ELEMENTS;
        for (int h = 0; h < STREAMS; h++) CHECK(hipStreamCreateWithFlags(&stream_[h], hipStreamNonBlocking));
        for (int h = 0; h < STREAMS; h++) {
            if (hipMalloc(&d_pbuf_[h], elements * 9 * max_lanes * sizeof(uint32_t)) != hipSuccess) {
                (void)hipGetLastError();
                if (h == 0) fatal("cannot allocate the scratch buffer");
                std::fprintf(stderr, "warning: memory for one lane set only\n");
                break;
            }
            CHECK(hipMalloc(&lanes_[h].fe_, (size_t)27 * max_lanes * sizeof(uint32_t)));
            CHECK(hipMalloc(&lanes_[h].k, (size_t)max_lanes * sizeof(long long)));
            sets_ = h + 1;
        }
        for (int i = 0; i < 2; i++) {
            CHECK(hipMalloc(&d_hits_[i], sizeof(unsigned long long) * HIT_CAP));
            CHECK(hipMalloc(&d_count_[i], sizeof(uint32_t)));
            CHECK(hipEventCreateWithFlags(&done_[i], PROFILE ? 0 : hipEventDisableTiming));
            CHECK(hipEventCreate(&begun_[i]));
        }
        CHECK(hipMalloc(&d_flags_, 2 * sizeof(uint32_t)));
        CHECK(hipMalloc(&d_steps_, sizeof(Prep) * 32));
        CHECK(hipMalloc(&d_giant_, sizeof(walk::Giant)));
        CHECK(hipMalloc(&d_prof_, sizeof(unsigned long long) * NPROF));
        CHECK(hipMemset(d_prof_, 0, sizeof(unsigned long long) * NPROF));
        CHECK(hipStreamCreateWithFlags(&copy_, hipStreamNonBlocking));
    }

    // Targets for the GPU filter: the fast form for up to four targets, else
    // the general form. Prefix masks nest, so the fast form tests every target
    // under the shortest one's mask: a pre-test that never rejects a match,
    // and for targets of different lengths passes the longer ones' non-matches
    // only at the shorter length's rate (the exact test settles them).
    void set_targets(const std::vector<targets::Target> &ts) {
        fast_ = ts.size() <= 4;
        uint32_t mask = ~0u;
        for (const auto &t : ts) mask &= (uint32_t)t.ymask;
        ft_.mask = mask;
        for (int i = 0; i < 4; i++) ft_.val[i] = (uint32_t)ts[i < (int)ts.size() ? i : 0].yval & mask;
        std::vector<uint32_t> keys;
        for (const auto &t : ts) keys.push_back((uint32_t)t.yval & mask);
        std::sort(keys.begin(), keys.end());
        keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
        ft_.size = 8;
        while (ft_.size < keys.size()) ft_.size *= 2;
        keys.resize(ft_.size, ~0u);
        CHECK(hipMemcpyToSymbol(HIP_SYMBOL(c_keys), keys.data(), sizeof(uint32_t) * keys.size()));
        std::vector<uint64_t> m(ts.size()), v(ts.size());
        for (size_t i = 0; i < ts.size(); i++) {
            m[i] = ts[i].ymask;
            v[i] = ts[i].yval;
        }
        int n = (int)ts.size();
        CHECK(hipMemcpyToSymbol(HIP_SYMBOL(c_nt), &n, sizeof(n)));
        CHECK(hipMemcpyToSymbol(HIP_SYMBOL(c_ymask), m.data(), sizeof(uint64_t) * n));
        CHECK(hipMemcpyToSymbol(HIP_SYMBOL(c_yval), v.data(), sizeof(uint64_t) * n));
    }

    // Places `lanes` lanes in each of `halves` lane sets: lane l of half h
    // (global index g = h*lanes + l) starts at center A0 + (kbase + g*SPAN)*B8,
    // and each batch advances every lane by halves*lanes*SPAN. Launches into
    // slot h run half h (one half: both slots run half 0). Returns the number
    // of lanes whose center has x = 0 (never, unless A0 is a known small
    // multiple of B8 plus torsion).
    uint32_t start(const ref::ge &a0, long long kbase, uint32_t lanes, int halves = 1) {
        if (lanes % 64) fatal("lane count must be a multiple of 64");
        if (halves < 1 || halves > sets_) fatal("bad lane set count");
        lanes_n_ = lanes;
        halves_ = halves;
        step_ = (long long)halves * lanes * SPAN;
        const walk::Giant giant = tables::giant(step_);
        CHECK(hipMemcpy(d_giant_, &giant, sizeof(giant), hipMemcpyHostToDevice));
        std::vector<Prep> steps;
        ref::ge D = ref::ge_scalarmult(tables::B8(), (unsigned __int128)SPAN);
        int nsteps = 0;
        while ((1u << nsteps) < lanes) nsteps++;
        for (int i = 0; i < nsteps; i++) {
            steps.push_back(prep(D));
            D = ref::ge_add(D, D);
        }
        if (nsteps) CHECK(hipMemcpy(d_steps_, steps.data(), sizeof(Prep) * nsteps, hipMemcpyHostToDevice));
        CHECK(hipMemsetAsync(d_flags_, 0, 2 * sizeof(uint32_t), stream_[0]));
        for (int h = 0; h < halves; h++) {
            const long long k0 = kbase + (long long)h * lanes * SPAN;
            const Ext A = ext29(tables::offset(a0, k0));
            init_kernel<<<(lanes + BLOCK - 1) / BLOCK, BLOCK, 0, stream_[0]>>>(
                lanes_[h], lanes, d_steps_, nsteps, A, k0, SPAN, d_flags_ + 1);
            CHECK(hipGetLastError());
        }
        CHECK(hipStreamSynchronize(stream_[0]));
        uint32_t dead = 0;
        CHECK(hipMemcpy(&dead, d_flags_ + 1, sizeof(dead), hipMemcpyDeviceToHost));
        return dead;
    }

    // Enqueue a launch of nb batches per lane into hit buffer `slot`. With
    // two halves, slot h's launches run half h on its own stream; the caller
    // collects a slot before launching into it again.
    void launch(int nb, int slot) {
        const int h = halves_ == 2 ? slot : 0;
        hipStream_t s = stream_[h];
        CHECK(hipMemsetAsync(d_count_[slot], 0, sizeof(uint32_t), s));
        if (PROFILE) CHECK(hipEventRecord(begun_[slot], s));
        const dim3 grid((lanes_n_ + BLOCK - 1) / BLOCK);
        if (fast_)
            grind_kernel<true><<<grid, BLOCK, 0, s>>>(lanes_[h], lanes_n_, nb, d_table_, d_giant_,
                step_, d_pbuf_[h], ft_, d_hits_[slot], d_count_[slot], d_flags_, d_prof_);
        else
            grind_kernel<false><<<grid, BLOCK, 0, s>>>(lanes_[h], lanes_n_, nb, d_table_, d_giant_,
                step_, d_pbuf_[h], ft_, d_hits_[slot], d_count_[slot], d_flags_, d_prof_);
        CHECK(hipGetLastError());
        CHECK(hipEventRecord(done_[slot], s));
    }

    // Wait for the launch in `slot`; return its flagged offsets.
    std::vector<long long> collect(int slot, bool &overflow) {
        CHECK(hipEventSynchronize(done_[slot]));
        if (PROFILE) {
            float ms = 0;
            CHECK(hipEventElapsedTime(&ms, begun_[slot], done_[slot]));
            kernel_seconds += ms / 1000.0;
        }
        uint32_t n = 0;
        CHECK(hipMemcpyAsync(&n, d_count_[slot], sizeof(n), hipMemcpyDeviceToHost, copy_));
        CHECK(hipStreamSynchronize(copy_));
        overflow = n > HIT_CAP;
        n = std::min(n, HIT_CAP);
        std::vector<long long> ks(n);
        if (n) {
            CHECK(hipMemcpyAsync(ks.data(), d_hits_[slot], sizeof(long long) * n,
                                 hipMemcpyDeviceToHost, copy_));
            CHECK(hipStreamSynchronize(copy_));
        }
        return ks;
    }

    uint32_t degenerate() {
        uint32_t d = 0;
        CHECK(hipMemcpy(&d, d_flags_, sizeof(d), hipMemcpyDeviceToHost));
        return d;
    }

    long long candidates_per_launch(int nb) const { return (long long)lanes_n_ * nb * SPAN; }

    int lane_sets() const { return sets_; }  // lane sets allocated (STREAMS, or 1 if memory is short)

    std::vector<unsigned long long> profile() {
        std::vector<unsigned long long> p(NPROF);
        CHECK(hipMemcpy(p.data(), d_prof_, sizeof(unsigned long long) * NPROF, hipMemcpyDeviceToHost));
        return p;
    }

  private:
    walk::Entry *d_table_ = nullptr;
    uint32_t *d_pbuf_[STREAMS] = {};
    LaneBuf lanes_[STREAMS] = {};
    unsigned long long *d_hits_[2] = {nullptr, nullptr};
    uint32_t *d_count_[2] = {nullptr, nullptr};
    uint32_t *d_flags_ = nullptr;  // [0] degenerate batches, [1] dead lanes
    Prep *d_steps_ = nullptr;
    hipEvent_t done_[2], begun_[2];
    hipStream_t stream_[STREAMS], copy_;
    uint32_t lanes_n_ = 0;
    int halves_ = 1, sets_ = 0;
    long long step_ = 0;
    walk::Giant *d_giant_ = nullptr;
    unsigned long long *d_prof_ = nullptr;
    bool fast_ = false;
    FastTargets ft_{};
};

// ---------------------------------------------------------------------------
// --info
// ---------------------------------------------------------------------------
static int cmd_info() {
    emit(std::string("{\"contract\": \"") + CONTRACT + "\", \"name\": \"" + NAME +
         "\", \"version\": \"" + VERSION + "\", \"model\": \"" + MODEL + "\", \"device\": \"" +
         json_escape(device_name()) + "\"}");
    return 0;
}

// ---------------------------------------------------------------------------
// --grind
// ---------------------------------------------------------------------------
static std::atomic<bool> g_stop{false};

static void watch_stdin() {
    std::string line;
    while (std::getline(std::cin, line)) {
        JValue msg;
        if (JParser(line).parse(msg) && msg.kind == JValue::Object) {
            const JValue *cmd = msg.get("cmd");
            if (cmd && cmd->kind == JValue::String && cmd->str == "stop") break;
        }
    }
    g_stop = true;  // stop, or EOF: the rig went away
}

static void emit_hits(const std::vector<long long> &ks) {
    for (size_t first = 0; first < ks.size(); first += 50000) {
        std::string line = "{\"event\": \"hits\", \"k\": [";
        const size_t last = std::min(ks.size(), first + 50000);
        for (size_t i = first; i < last; i++) {
            if (i > first) line += ", ";
            line += "\"" + std::to_string(ks[i]) + "\"";
        }
        emit(line + "]}");
    }
}

static int cmd_grind() {
    g_protocol = true;
    Engine eng;
    eng.init();
    targets::Checker checker;
    {
        // Warm up both kernels (code object load) on a throwaway start.
        std::vector<targets::Target> ts;
        targets::parse({"a"}, ts);
        eng.set_targets(ts);
        eng.start(tables::B8(), 0, std::min<uint32_t>(eng.max_lanes, BLOCK));
        eng.launch(1, 0);
        bool overflow;
        eng.collect(0, overflow);
    }
    emit("{\"event\": \"ready\"}");

    std::string line;
    if (!std::getline(std::cin, line)) return 1;
    JValue start;
    if (!JParser(line).parse(start) || start.kind != JValue::Object || !start.get("cmd") ||
        start.get("cmd")->str != "start") {
        emit_error("expected a start command");
        return 1;
    }
    const JValue *a0hex = start.get("a0");
    uint8_t a0b[32];
    ref::ge a0;
    if (!a0hex || a0hex->kind != JValue::String || !ref::hex_decode(a0hex->str, a0b, 32) ||
        !ref::ge_frombytes(a0, a0b)) {
        emit_error("a0 is not a valid ed25519 point encoding");
        return 1;
    }
    std::vector<std::string> texts;
    if (const JValue *t = start.get("targets"))
        for (const JValue &v : t->arr) texts.push_back(v.str);
    std::vector<targets::Target> ts;
    if (std::string problem = targets::parse(texts, ts); !problem.empty()) {
        emit_error(problem);
        return 1;
    }
    const JValue *secs = start.get("seconds");
    const double seconds = (secs && secs->kind == JValue::Number) ? secs->num : -1.0;
    const auto t0 = Clock::now();
    std::thread(watch_stdin).detach();

    checker.set_start(a0);
    eng.set_targets(ts);
    const uint32_t dead = eng.start(a0, H, eng.max_lanes, eng.lane_sets());
    if (dead) std::fprintf(stderr, "warning: %u lanes start at x = 0 and are skipped\n", dead);

    long long candidates = 0, flagged = 0, verified = 0;
    int nb = 2;  // first launches short; then sized to about 0.3 s each
    auto last_progress = t0;
    int slot = 0;
    eng.launch(nb, slot);
    int inflight_nb = nb;
    auto launched_at = Clock::now();
    while (true) {
        const bool more = !g_stop && (seconds < 0 || seconds_since(t0) < seconds);
        int next_nb = nb;
        if (more) eng.launch(next_nb, slot ^ 1);
        bool overflow = false;
        std::vector<long long> ks = eng.collect(slot, overflow);
        const double took = seconds_since(launched_at);
        launched_at = Clock::now();
        if (overflow) std::fprintf(stderr, "warning: hit buffer overflow; some hits lost\n");
        candidates += eng.candidates_per_launch(inflight_nb);
        flagged += (long long)ks.size();
        std::vector<long long> good;
        for (long long k : ks)
            if (checker.hit(k, ts)) good.push_back(k);
        verified += (long long)good.size();
        if (!good.empty()) emit_hits(good);
        if (more && took > 0.01) nb = std::max(1, std::min(256, (int)std::lround(nb * LAUNCH_SECONDS / took)));
        if (seconds_since(last_progress) > 5) {
            emit("{\"event\": \"progress\", \"candidates\": \"" + std::to_string(candidates) + "\"}");
            last_progress = Clock::now();
        }
        if (!more) break;
        slot ^= 1;
        inflight_nb = next_nb;
    }
    const uint32_t deg = eng.degenerate();
    std::fprintf(stderr, "done: %lld candidates in %.2f s (%.3f G/s), %lld flagged, %lld verified, "
                 "%u degenerate batches\n", candidates, seconds_since(t0),
                 candidates / seconds_since(t0) / 1e9, flagged, verified, deg);
    emit("{\"event\": \"done\", \"candidates\": \"" + std::to_string(candidates) + "\"}");
    std::fflush(stdout);
    std::_Exit(0);  // the stdin watcher may still be blocked in getline
}

// ---------------------------------------------------------------------------
// --verify
// ---------------------------------------------------------------------------
struct Check {
    std::string name;
    bool ok;
    std::string detail;
};

static std::string hex(const uint8_t *b, int n) {
    static const char *d = "0123456789abcdef";
    std::string s;
    for (int i = 0; i < n; i++) {
        s += d[b[i] >> 4];
        s += d[b[i] & 15];
    }
    return s;
}

// The GPU field arithmetic against the shared field vectors.
static Check verify_field(const std::string &dir) {
    std::vector<fe> a, b;
    std::vector<int> op;
    std::vector<std::string> want;
    const char *names[] = {"mul", "sqr", "add", "sub", "inv"};
    for (int o = 0; o < 5; o++) {
        for (auto &c : read_cases(dir + "/field.txt", names[o])) {
            uint8_t sa[32], sb[32];
            ref::hex_decode(c[1], sa, 32);
            a.push_back(tables::to29(ref::fe_frombytes(sa)));
            if (c[2] != "-") {
                ref::hex_decode(c[2], sb, 32);
                b.push_back(tables::to29(ref::fe_frombytes(sb)));
            } else {
                b.push_back(a.back());
            }
            op.push_back(o);
            want.push_back(c[3]);
        }
    }
    // Inner products, against the reference.
    std::mt19937_64 rng(7);
    for (int i = 0; i < 256; i++) {
        uint8_t sa[32], sb[32];
        for (int k = 0; k < 32; k++) { sa[k] = (uint8_t)rng(); sb[k] = (uint8_t)rng(); }
        ref::fe ra = ref::fe_frombytes(sa), rb = ref::fe_frombytes(sb);
        a.push_back(tables::to29(ra));
        b.push_back(tables::to29(rb));
        op.push_back(5);
        uint8_t out[32];
        ref::fe ab = ref::fe_mul(ra, rb);
        ref::fe_tobytes(out, ref::fe_add(ab, ab));
        want.push_back(hex(out, 32));
    }
    const int n = (int)a.size();
    fe *da, *db, *dout;
    int *dop;
    CHECK(hipMalloc(&da, sizeof(fe) * n));
    CHECK(hipMalloc(&db, sizeof(fe) * n));
    CHECK(hipMalloc(&dout, sizeof(fe) * n));
    CHECK(hipMalloc(&dop, sizeof(int) * n));
    CHECK(hipMemcpy(da, a.data(), sizeof(fe) * n, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(db, b.data(), sizeof(fe) * n, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(dop, op.data(), sizeof(int) * n, hipMemcpyHostToDevice));
    field_kernel<<<(n + 63) / 64, 64>>>(da, db, dop, dout, n);
    CHECK(hipGetLastError());
    CHECK(hipDeviceSynchronize());
    std::vector<fe> out(n);
    CHECK(hipMemcpy(out.data(), dout, sizeof(fe) * n, hipMemcpyDeviceToHost));
    CHECK(hipFree(da)); CHECK(hipFree(db)); CHECK(hipFree(dout)); CHECK(hipFree(dop));
    int bad = 0;
    for (int i = 0; i < n; i++) {
        uint8_t s[32];
        f29::to_bytes(s, out[i]);
        if (hex(s, 32) != want[i]) bad++;
    }
    return {"field", bad == 0, std::to_string(n - bad) + "/" + std::to_string(n) + " ops"};
}

// A0 decoding (the host does it, as in --grind) against the shared vectors.
static Check verify_decode(const std::string &dir) {
    auto cases = read_cases(dir + "/decode.txt", "decode");
    int bad = 0;
    for (auto &c : cases) {
        uint8_t s[32];
        ref::hex_decode(c[1], s, 32);
        ref::ge p;
        const bool ok = ref::ge_frombytes(p, s);
        if (c[2] == "invalid") {
            bad += ok;
        } else {
            uint8_t out[32];
            if (ok) ref::ge_tobytes(out, p);
            bad += !ok || hex(out, 32) != c[2];
        }
    }
    return {"decode", !cases.empty() && bad == 0, std::to_string(cases.size()) + " cases"};
}

// Offsets A0 + k*B8 as the host checker computes them.
static Check verify_offset(const std::string &dir) {
    auto cases = read_cases(dir + "/offset.txt", "offset");
    targets::Checker chk;
    int bad = 0;
    for (auto &c : cases) {
        uint8_t s[32], out[32];
        ref::hex_decode(c[1], s, 32);
        ref::ge a0;
        if (!ref::ge_frombytes(a0, s)) { bad++; continue; }
        chk.set_start(a0);
        chk.pubkey(out, std::stoll(c[2]));
        bad += hex(out, 32) != c[3];
    }
    return {"offset", !cases.empty() && bad == 0, std::to_string(cases.size()) + " cases"};
}

// Runs the real grind kernel over [lo, hi) and returns the verified hits there.
static std::vector<long long> grind_window(Engine &eng, const ref::ge &a0,
                                           const std::vector<targets::Target> &ts, long long lo,
                                           long long hi, long long &flagged) {
    targets::Checker chk;
    chk.set_start(a0);
    eng.set_targets(ts);
    const uint32_t lanes = 64;
    const long long per_batch = (long long)lanes * SPAN;
    const int nb = (int)((hi - lo + per_batch - 1) / per_batch);
    eng.start(a0, lo + H, lanes);
    eng.launch(nb, 0);
    bool overflow = false;
    std::vector<long long> ks = eng.collect(0, overflow);
    flagged = (long long)ks.size();
    std::vector<long long> got;
    for (long long k : ks)
        if (k >= lo && k < hi && chk.hit(k, ts)) got.push_back(k);
    std::sort(got.begin(), got.end());
    if (overflow) got.push_back(LLONG_MIN);  // forces a mismatch
    return got;
}

static Check verify_kat(Engine &eng, const std::vector<std::string> &c) {
    // kat <name> <a0> <nt> <targets...> <lo> <hi> <nh> <hits...>
    const std::string name = "kat:" + c[1];
    uint8_t s[32];
    ref::ge a0;
    if (!ref::hex_decode(c[2], s, 32) || !ref::ge_frombytes(a0, s)) return {name, false, "bad a0"};
    const size_t nt = std::stoul(c[3]);
    std::vector<std::string> texts(c.begin() + 4, c.begin() + 4 + nt);
    std::vector<targets::Target> ts;
    if (!targets::parse(texts, ts).empty()) return {name, false, "bad targets"};
    const long long lo = std::stoll(c[4 + nt]), hi = std::stoll(c[5 + nt]);
    const size_t nh = std::stoul(c[6 + nt]);
    std::vector<long long> want;
    for (size_t i = 0; i < nh; i++) want.push_back(std::stoll(c[7 + nt + i]));
    long long flagged = 0;
    std::vector<long long> got = grind_window(eng, a0, ts, lo, hi, flagged);
    return {name, got == want,
            std::to_string(got.size()) + " hits (want " + std::to_string(want.size()) + "), " +
                std::to_string(flagged) + " flagged"};
}

// Fresh random start points: the grind kernel's hits in a window must equal
// an exhaustive host computation of the same window.
// Target sets for the randomized windows: four of 1 character (about one
// candidate in eight matches), two of 1 and two of 2 characters (the fast
// filter with mixed lengths), or 24 of 2 and 3 characters (the general form).
enum class TargetSet { Short, Mixed, Many };

static Check verify_random(Engine &eng, uint64_t seed, TargetSet set) {
    std::mt19937_64 rng(seed);
    const unsigned __int128 a = ((unsigned __int128)rng() << 64) | rng();
    const ref::ge a0 = ref::ge_scalarmult(ref::base_point(), a);
    std::vector<std::string> texts;
    const char *alpha = "abcdefghijklmnopqrstuvwxyz234567";
    const size_t count = set == TargetSet::Many ? 24 : 4;
    while (texts.size() < count) {
        std::string t(1, alpha[rng() % 32]);
        if (set == TargetSet::Mixed && texts.size() >= 2) t += alpha[rng() % 32];
        if (set == TargetSet::Many) {
            t += alpha[rng() % 32];
            if (texts.size() % 3 == 0) t += alpha[rng() % 32];
        }
        bool clash = false;  // targets must be prefix-free
        for (const auto &u : texts) clash = clash || t.compare(0, u.size(), u) == 0 || u.compare(0, t.size(), t) == 0;
        if (!clash) texts.push_back(t);
    }
    std::vector<targets::Target> ts;
    targets::parse(texts, ts);
    const long long lo = -(long long)(rng() % 100000),
                    hi = lo + (set == TargetSet::Short ? 2 : 1) * 64 * SPAN;
    long long flagged = 0;
    std::vector<long long> got = grind_window(eng, a0, ts, lo, hi, flagged);
    std::vector<long long> want;
    ref::ge P;
    targets::Checker chk;
    chk.set_start(a0);
    chk.point(P, lo);
    for (long long k = lo; k < hi; k++) {
        uint8_t pub[32];
        ref::ge_tobytes(pub, P);
        for (auto &t : ts)
            if (targets::matches(pub, t)) { want.push_back(k); break; }
        P = ref::ge_add(P, tables::B8());
    }
    const char *kind = set == TargetSet::Short ? "random:" : set == TargetSet::Mixed ? "random-mixed:" : "random-many:";
    return {kind + std::to_string(seed), got == want,
            std::to_string(got.size()) + " hits (want " + std::to_string(want.size()) + ") over " +
                std::to_string(hi - lo) + " offsets"};
}

static int cmd_verify(const std::string &dir) {
    Engine eng;
    eng.init();
    std::vector<Check> checks;
    checks.push_back(verify_field(dir));
    checks.push_back(verify_decode(dir));
    checks.push_back(verify_offset(dir));
    for (auto &c : read_cases(dir + "/grind.txt", "kat")) checks.push_back(verify_kat(eng, c));
    for (uint64_t seed : {11, 12}) checks.push_back(verify_random(eng, seed, TargetSet::Short));
    checks.push_back(verify_random(eng, 13, TargetSet::Mixed));
    checks.push_back(verify_random(eng, 14, TargetSet::Many));
    int failed = 0;
    std::string list;
    for (size_t i = 0; i < checks.size(); i++) {
        failed += !checks[i].ok;
        list += std::string(i ? ", " : "") + "{\"name\": \"" + json_escape(checks[i].name) +
                "\", \"ok\": " + (checks[i].ok ? "true" : "false") + ", \"detail\": \"" +
                json_escape(checks[i].detail) + "\"}";
        std::fprintf(stderr, "%s %s: %s\n", checks[i].ok ? "PASS" : "FAIL", checks[i].name.c_str(),
                     checks[i].detail.c_str());
    }
    emit("{\"passed\": " + std::to_string(checks.size() - failed) + ", \"failed\": " +
         std::to_string(failed) + ", \"checks\": [" + list + "]}");
    return failed == 0 ? 0 : 1;
}

// ---------------------------------------------------------------------------
// --bench SECONDS [TARGETS [CHARS]]: a development aid, not part of the
// contract. Grinds a random start point for SECONDS with TARGETS random
// targets of CHARS characters and prints the self-timed rate (the rig's
// proof-of-work rate is the only official one).
// ---------------------------------------------------------------------------
static int cmd_bench(double seconds, int ntargets, int chars) {
    Engine eng;
    eng.init();
    std::random_device rd;
    std::mt19937_64 rng(((uint64_t)rd() << 32) ^ rd());
    const unsigned __int128 a = ((unsigned __int128)rng() << 64) | rng();
    const ref::ge a0 = ref::ge_scalarmult(ref::base_point(), a);
    const char *alpha = "abcdefghijklmnopqrstuvwxyz234567";
    std::vector<std::string> texts;
    while ((int)texts.size() < ntargets) {
        std::string t;
        for (int i = 0; i < chars; i++) t += alpha[rng() % 32];
        if (std::find(texts.begin(), texts.end(), t) == texts.end()) texts.push_back(t);
    }
    std::vector<targets::Target> ts;
    targets::parse(texts, ts);
    targets::Checker checker;
    checker.set_start(a0);
    eng.set_targets(ts);
    const auto t0 = Clock::now();
    eng.start(a0, H, eng.max_lanes, eng.lane_sets());
    long long candidates = 0, flagged = 0, verified = 0;
    int nb = 2, slot = 0, inflight = nb;
    eng.launch(nb, slot);
    auto last = Clock::now();
    while (true) {
        const bool more = seconds_since(t0) < seconds;
        const int next = nb;
        if (more) eng.launch(next, slot ^ 1);
        bool overflow = false;
        std::vector<long long> ks = eng.collect(slot, overflow);
        const double took = seconds_since(last);
        last = Clock::now();
        candidates += eng.candidates_per_launch(inflight);
        flagged += (long long)ks.size();
        for (long long k : ks) verified += checker.hit(k, ts);
        if (more && took > 0.01) nb = std::max(1, std::min(256, (int)std::lround(nb * LAUNCH_SECONDS / took)));
        if (!more) break;
        slot ^= 1;
        inflight = next;
    }
    const double wall = seconds_since(t0);
    const double expect = candidates * (double)ts.size() / std::pow(32.0, chars);
    if (PROFILE) {
        std::vector<unsigned long long> p = eng.profile();
        int wall_khz = 0;
        CHECK(hipDeviceGetAttribute(&wall_khz, hipDeviceAttributeWallClockRate, 0));
        // Effective shader clock: s_memtime ticks per s_memrealtime second.
        const double mhz = p[NPHASE + 1] ? (double)p[NPHASE] / p[NPHASE + 1] * wall_khz / 1000.0 : 0;
        const double waves = eng.max_lanes / 64.0;
        const double wave_pairs = (double)candidates / SPAN * H / eng.max_lanes;  // per wave
        // Seconds per wave inside the kernel (s_memrealtime), GPU kernel time
        // (events), and host wall time: gaps between them are idle GPU time.
        const double inside = (double)p[NPHASE + 1] / waves / (wall_khz * 1000.0);
        std::printf("{\"prof\": [%.0f, %.2f, %.2f, %.2f", mhz, inside, eng.kernel_seconds, wall);
        for (int i = 0; i < NPHASE; i++) std::printf(", %.0f", (double)p[i] / waves / wave_pairs);
        std::printf("]}\n");
    }
    std::printf("{\"rate\": %.6e, \"seconds\": %.3f, \"candidates\": %lld, \"flagged\": %lld, "
                "\"verified\": %lld, \"expected\": %.1f, \"degenerate\": %u}\n",
                candidates / wall, wall, candidates, flagged, verified, expect, eng.degenerate());
    return 0;
}

int main(int argc, char **argv) {
    if (argc == 2 && std::string(argv[1]) == "--info") return cmd_info();
    if (argc == 2 && std::string(argv[1]) == "--grind") return cmd_grind();
    if (argc == 3 && std::string(argv[1]) == "--verify") return cmd_verify(argv[2]);
    if (argc >= 3 && std::string(argv[1]) == "--bench")
        return cmd_bench(std::atof(argv[2]), argc > 3 ? std::atoi(argv[3]) : 4,
                         argc > 4 ? std::atoi(argv[4]) : 5);
    std::fprintf(stderr, "usage: %s --info | --verify VECTORS_DIR | --grind\n", argv[0]);
    return 2;
}
