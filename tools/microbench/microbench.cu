// Primitive-throughput microbenchmarks: settle the integer-vs-FP64 field-multiply
// question with measurements instead of datasheets (docs/design/CDNA-PERF-REFERENCE.md s6).
//
// Each kernel runs CHAINS independent dependency chains per thread (enough
// instruction-level parallelism to saturate the pipe) of one primitive, in a
// recurrence the compiler cannot hoist or fold. Each test sizes its loop so one
// sample takes about target_ms on this GPU, then reports the mean of `samples`
// samples (mean, never min, as in the rig) and their relative spread.
//
// Every block also reads the shader-clock cycle counter and a fixed-rate wall
// clock at its start and end. Their ratio is the clock the kernel actually ran
// at, so each test reports ops per CU per clock both at the nominal (peak) clock
// and at the measured one: a GPU that clocks down under load is told apart from
// an instruction that issues below its peak rate.
//
// Usage: microbench [samples=5] [target_ms=100]. Output: one JSON line per test,
// with host epoch times (t_start, t_end) to line up with sampled telemetry.
// Build: make cuda | make hip HIP_ARCHS="gfx90a gfx950"; `make isa` confirms each
// kernel compiles to the instruction it claims to measure.

#if defined(__HIPCC__)
  #include <hip/hip_runtime.h>
  #define gpuMalloc hipMalloc
  #define gpuFree hipFree
  #define gpuMemcpy hipMemcpy
  #define gpuMemcpyDeviceToHost hipMemcpyDeviceToHost
  #define gpuDeviceSynchronize hipDeviceSynchronize
  #define gpuEvent_t hipEvent_t
  #define gpuEventCreate hipEventCreate
  #define gpuEventDestroy hipEventDestroy
  #define gpuEventRecord hipEventRecord
  #define gpuEventSynchronize hipEventSynchronize
  #define gpuEventElapsedTime hipEventElapsedTime
  #define gpuDeviceProp hipDeviceProp_t
  #define gpuGetDeviceProperties hipGetDeviceProperties
  #define gpuDeviceGetAttribute hipDeviceGetAttribute
  #define gpuDevAttrClockRate hipDeviceAttributeClockRate
  #define gpuError_t hipError_t
  #define gpuSuccess hipSuccess
  #define gpuGetErrorString hipGetErrorString
  #define gpuGetLastError hipGetLastError
#else
  #include <cuda_runtime.h>
  #define gpuMalloc cudaMalloc
  #define gpuFree cudaFree
  #define gpuMemcpy cudaMemcpy
  #define gpuMemcpyDeviceToHost cudaMemcpyDeviceToHost
  #define gpuDeviceSynchronize cudaDeviceSynchronize
  #define gpuEvent_t cudaEvent_t
  #define gpuEventCreate cudaEventCreate
  #define gpuEventDestroy cudaEventDestroy
  #define gpuEventRecord cudaEventRecord
  #define gpuEventSynchronize cudaEventSynchronize
  #define gpuEventElapsedTime cudaEventElapsedTime
  #define gpuDeviceProp cudaDeviceProp
  #define gpuGetDeviceProperties cudaGetDeviceProperties
  #define gpuDeviceGetAttribute cudaDeviceGetAttribute
  #define gpuDevAttrClockRate cudaDevAttrClockRate
  #define gpuError_t cudaError_t
  #define gpuSuccess cudaSuccess
  #define gpuGetErrorString cudaGetErrorString
  #define gpuGetLastError cudaGetLastError
#endif

#include <cstdint>
#include <cstdio>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "../../entries/baseline/src/fe25519.cuh"  // ref10 field arithmetic (upstream, MIT)

// Any failed call ends the run: carrying on would print a throughput for work
// that never ran.
#define CHECK(call)                                                                \
    do {                                                                           \
        const gpuError_t err_ = (call);                                            \
        if (err_ != gpuSuccess) {                                                  \
            std::fprintf(stderr, "microbench: %s failed: %s\n", #call,             \
                         gpuGetErrorString(err_));                                 \
            std::exit(1);                                                          \
        }                                                                          \
    } while (0)

#define CHAINS 8
#define BLOCKS 4096
#define THREADS 256

// Shader-clock cycles and wall-clock ticks at the start and end of one block.
struct Stamp {
    unsigned long long cycles0, cycles1, wall0, wall1;
};

__device__ inline unsigned long long wall_ticks() {
#if defined(__HIPCC__)
    return wall_clock64();  // s_memrealtime: fixed rate, hipDeviceAttributeWallClockRate
#else
    unsigned long long ns;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(ns));
    return ns;  // nanoseconds
#endif
}

__device__ inline Stamp stamp_begin() {
    return {(unsigned long long)clock64(), 0, wall_ticks(), 0};
}

// Called after the block's result is stored, so the reads cannot move above the loop.
__device__ inline void stamp_end(Stamp st, Stamp *stamps) {
    if (threadIdx.x == 0) {
        st.cycles1 = (unsigned long long)clock64();
        st.wall1 = wall_ticks();
        stamps[blockIdx.x] = st;
    }
}

// 64-bit result of a 32x32 multiply plus a 64-bit addend: v_mad_u64_u32 on CDNA.
__global__ void k_mad_u64_u32(uint64_t *out, Stamp *stamps, uint32_t m, int iters) {
    const Stamp st = stamp_begin();
    uint64_t acc[CHAINS];
    for (int c = 0; c < CHAINS; c++) acc[c] = threadIdx.x + 977u * c + 1;
    for (int i = 0; i < iters; i++) {
        #pragma unroll
        for (int c = 0; c < CHAINS; c++) acc[c] = (uint64_t)(uint32_t)acc[c] * (m + c) + acc[c];
    }
    uint64_t s = 0;
    for (int c = 0; c < CHAINS; c++) s ^= acc[c];
    out[blockIdx.x * blockDim.x + threadIdx.x] = s;
    stamp_end(st, stamps);
}

// Low 32 bits of a 32x32 product: v_mul_lo_u32.
__global__ void k_mul_lo_u32(uint64_t *out, Stamp *stamps, uint32_t m, int iters) {
    const Stamp st = stamp_begin();
    uint32_t acc[CHAINS];
    for (int c = 0; c < CHAINS; c++) acc[c] = threadIdx.x * 2 + 2 * c + 1;
    for (int i = 0; i < iters; i++) {
        #pragma unroll
        for (int c = 0; c < CHAINS; c++) acc[c] *= (m | 1) + 2 * c;
    }
    uint32_t s = 0;
    for (int c = 0; c < CHAINS; c++) s ^= acc[c];
    out[blockIdx.x * blockDim.x + threadIdx.x] = s;
    stamp_end(st, stamps);
}

// High 32 bits of a 32x32 product, plus a cheap add to keep the chain alive.
__global__ void k_mul_hi_u32(uint64_t *out, Stamp *stamps, uint32_t m, int iters) {
    const Stamp st = stamp_begin();
    uint32_t acc[CHAINS];
    for (int c = 0; c < CHAINS; c++) acc[c] = threadIdx.x + 0x9e3779b9u * (c + 1);
    for (int i = 0; i < iters; i++) {
        #pragma unroll
        for (int c = 0; c < CHAINS; c++) acc[c] = __umulhi(acc[c], m + c) + acc[c];
    }
    uint32_t s = 0;
    for (int c = 0; c < CHAINS; c++) s ^= acc[c];
    out[blockIdx.x * blockDim.x + threadIdx.x] = s;
    stamp_end(st, stamps);
}

// Fused multiply-add in double precision: v_fma_f64, reading three VGPR pairs'
// worth of operands (the accumulator, plus 1e-4 kept in registers).
__global__ void k_fma_f64(uint64_t *out, Stamp *stamps, double m, int iters) {
    const Stamp st = stamp_begin();
    double acc[CHAINS];
    for (int c = 0; c < CHAINS; c++) acc[c] = 1.0 + threadIdx.x * 1e-9 + c * 1e-3;
    for (int i = 0; i < iters; i++) {
        #pragma unroll
        for (int c = 0; c < CHAINS; c++) acc[c] = fma(acc[c], m, 1e-4);
    }
    double s = 0;
    for (int c = 0; c < CHAINS; c++) s += acc[c];
    uint64_t bits;
    memcpy(&bits, &s, 8);
    out[blockIdx.x * blockDim.x + threadIdx.x] = bits;
    stamp_end(st, stamps);
}

// The same FMA with one VGPR operand (two inline constants): if it runs faster,
// register reads, not the FMA pipe, limit v_fma_f64 above.
__global__ void k_fma_f64_1src(uint64_t *out, Stamp *stamps, int iters) {
    const Stamp st = stamp_begin();
    double acc[CHAINS];
    for (int c = 0; c < CHAINS; c++) acc[c] = 1.0 + threadIdx.x * 1e-9 + c * 1e-3;
    for (int i = 0; i < iters; i++) {
        #pragma unroll
        for (int c = 0; c < CHAINS; c++) acc[c] = fma(acc[c], 0.5, 0.5);
    }
    double s = 0;
    for (int c = 0; c < CHAINS; c++) s += acc[c];
    uint64_t bits;
    memcpy(&bits, &s, 8);
    out[blockIdx.x * blockDim.x + threadIdx.x] = bits;
    stamp_end(st, stamps);
}

// Fused multiply-add in single precision: the full-rate reference.
__global__ void k_fma_f32(uint64_t *out, Stamp *stamps, float m, int iters) {
    const Stamp st = stamp_begin();
    float acc[CHAINS];
    for (int c = 0; c < CHAINS; c++) acc[c] = 1.0f + threadIdx.x * 1e-6f + c * 1e-3f;
    for (int i = 0; i < iters; i++) {
        #pragma unroll
        for (int c = 0; c < CHAINS; c++) acc[c] = fmaf(acc[c], m, 1e-4f);
    }
    float s = 0;
    for (int c = 0; c < CHAINS; c++) s += acc[c];
    uint32_t bits;
    memcpy(&bits, &s, 4);
    out[blockIdx.x * blockDim.x + threadIdx.x] = bits;
    stamp_end(st, stamps);
}

// A whole ref10 field multiply (10 x 25.5-bit limbs, int64 accumulators): the
// unit the baseline's hot loop is built from. Two chains for ILP.
__global__ void k_fe_mul_ref10(uint64_t *out, Stamp *stamps, int iters) {
    const Stamp st = stamp_begin();
    fe a, b, c;
    for (int i = 0; i < 10; i++) {
        a[i] = (int32_t)((threadIdx.x * 2654435761u + i) & 0x1ffffff);
        b[i] = (int32_t)((threadIdx.x * 40503u + 7 * i) & 0x1ffffff);
        c[i] = (int32_t)((blockIdx.x * 69069u + 3 * i) & 0x1ffffff);
    }
    for (int i = 0; i < iters; i++) {
        fe_mul(a, a, b);
        fe_mul(c, c, b);
    }
    uint64_t s = 0;
    for (int i = 0; i < 10; i++) s += (uint32_t)a[i] ^ (uint32_t)c[i];
    out[blockIdx.x * blockDim.x + threadIdx.x] = s;
    stamp_end(st, stamps);
}

struct Test {
    const char *name;
    const char *instruction;  // what one counted op compiles to on CDNA
    double ops_per_iter;      // counted ops per thread per loop iteration
    int iters;                // minimum loop count; scaled up to reach target_ms
};

struct Setup {
    int cus;
    double nominal_mhz;  // the device's reported (peak) clock
    double wall_mhz;     // rate of wall_ticks()
    int samples;
    double target_ms;
    Stamp *stamps;       // BLOCKS entries, rewritten by every launch
};

static double epoch_seconds() {
    using namespace std::chrono;
    return duration<double>(system_clock::now().time_since_epoch()).count();
}

template <typename Launch>
static double timed_ms(const Test &t, Launch launch, int iters, gpuEvent_t start, gpuEvent_t stop) {
    CHECK(gpuEventRecord(start, 0));
    launch(iters);
    CHECK(gpuGetLastError());
    CHECK(gpuEventRecord(stop, 0));
    CHECK(gpuEventSynchronize(stop));
    float ms = 0;
    CHECK(gpuEventElapsedTime(&ms, start, stop));
    if (!(ms > 0)) {
        std::fprintf(stderr, "microbench: %s timed at %g ms\n", t.name, ms);
        std::exit(1);
    }
    return ms;
}

// Mean clock over every block of the last launch: total cycles over total time.
static double measured_mhz(const Setup &s) {
    std::vector<Stamp> host(BLOCKS);
    CHECK(gpuMemcpy(host.data(), s.stamps, sizeof(Stamp) * BLOCKS, gpuMemcpyDeviceToHost));
    double cycles = 0, ticks = 0;
    for (const Stamp &b : host) {
        cycles += (double)(b.cycles1 - b.cycles0);
        ticks += (double)(b.wall1 - b.wall0);
    }
    return ticks > 0 ? cycles / ticks * s.wall_mhz : 0;
}

template <typename Launch>
static void run(const Test &t, Launch launch, const Setup &s) {
    gpuEvent_t start, stop;
    CHECK(gpuEventCreate(&start));
    CHECK(gpuEventCreate(&stop));
    launch(t.iters / 16);  // warm-up: code resident
    CHECK(gpuGetLastError());
    CHECK(gpuDeviceSynchronize());
    // Rates differ by up to 64x between parts (FP64 above all), so a fixed loop
    // count gives some GPUs samples too short to time well.
    const double probe = timed_ms(t, launch, t.iters, start, stop);
    const int iters = (int)std::clamp(t.iters * s.target_ms / probe, (double)t.iters, 1e8);
    // Untimed full-length samples first: an idle GPU takes a few hundred ms to clock up.
    for (int i = 0; i < 3; i++) timed_ms(t, launch, iters, start, stop);
    const double t_start = epoch_seconds();
    double sum = 0, lo = 1e300, hi = 0;
    for (int i = 0; i < s.samples; i++) {
        const double ms = timed_ms(t, launch, iters, start, stop);
        sum += ms;
        lo = std::min(lo, ms);
        hi = std::max(hi, ms);
    }
    const double t_end = epoch_seconds();
    CHECK(gpuEventDestroy(start));
    CHECK(gpuEventDestroy(stop));
    const double ms = sum / s.samples;
    const double ops = (double)BLOCKS * THREADS * t.ops_per_iter * iters;
    const double per_s = ops / (ms / 1e3);
    const double clock = measured_mhz(s);
    const double per_measured = clock > 0 ? per_s / (s.cus * clock * 1e6) : 0;
    std::printf("{\"test\": \"%s\", \"instruction\": \"%s\", \"iters\": %d, "
                "\"samples\": %d, \"seconds\": %.6f, \"rel_spread\": %.4f, "
                "\"ops_per_s\": %.6e, \"ops_per_cu_per_clock\": %.3f, "
                "\"measured_clock_mhz\": %.0f, \"ops_per_cu_per_measured_clock\": %.3f, "
                "\"t_start\": %.3f, \"t_end\": %.3f}\n",
                t.name, t.instruction, iters, s.samples, ms / 1e3, (hi - lo) / ms, per_s,
                per_s / (s.cus * s.nominal_mhz * 1e6), clock, per_measured, t_start, t_end);
    std::fflush(stdout);
}

int main(int argc, char **argv) {
    const int samples = argc > 1 ? std::atoi(argv[1]) : 5;
    const double target_ms = argc > 2 ? std::atof(argv[2]) : 100.0;
    if (samples < 1 || !(target_ms > 0)) {
        std::fprintf(stderr, "usage: microbench [samples>=1] [target_ms>0]\n");
        return 2;
    }
    gpuDeviceProp prop;
    CHECK(gpuGetDeviceProperties(&prop, 0));
    int clock_khz = 0;
    CHECK(gpuDeviceGetAttribute(&clock_khz, gpuDevAttrClockRate, 0));
#if defined(__HIPCC__)
    const std::string arch = prop.gcnArchName;
    int wall_khz = 0;
    CHECK(gpuDeviceGetAttribute(&wall_khz, hipDeviceAttributeWallClockRate, 0));
    const double wall_mhz = wall_khz / 1e3;
#else
    const std::string arch = "sm_" + std::to_string(prop.major) + std::to_string(prop.minor);
    const double wall_mhz = 1e3;  // %globaltimer counts nanoseconds
#endif
    std::printf("{\"device\": \"%s\", \"arch\": \"%s\", \"compute_units\": %d, "
                "\"nominal_clock_mhz\": %.0f, \"wall_clock_mhz\": %.0f, \"chains\": %d, "
                "\"samples\": %d, \"target_ms\": %.0f}\n",
                prop.name, arch.c_str(), prop.multiProcessorCount, clock_khz / 1e3, wall_mhz,
                CHAINS, samples, target_ms);

    uint64_t *out;
    Stamp *stamps;
    CHECK(gpuMalloc(&out, sizeof(uint64_t) * BLOCKS * THREADS));
    CHECK(gpuMalloc(&stamps, sizeof(Stamp) * BLOCKS));
    const Setup s{prop.multiProcessorCount, clock_khz / 1e3, wall_mhz, samples, target_ms, stamps};
    const dim3 g(BLOCKS), b(THREADS);
    run({"mad_u64_u32", "v_mad_u64_u32", CHAINS, 4096},
        [&](int n) { k_mad_u64_u32<<<g, b>>>(out, stamps, 0x9e3779b1u, n); }, s);
    run({"mul_lo_u32", "v_mul_lo_u32", CHAINS, 4096},
        [&](int n) { k_mul_lo_u32<<<g, b>>>(out, stamps, 0x85ebca6bu, n); }, s);
    run({"mul_hi_u32_plus_add", "v_mul_hi_u32 + v_add_u32", CHAINS, 4096},
        [&](int n) { k_mul_hi_u32<<<g, b>>>(out, stamps, 0xc2b2ae35u, n); }, s);
    run({"fma_f64", "v_fma_f64", CHAINS, 4096},
        [&](int n) { k_fma_f64<<<g, b>>>(out, stamps, 0.9999, n); }, s);
    run({"fma_f64_1src", "v_fma_f64 (one VGPR operand)", CHAINS, 4096},
        [&](int n) { k_fma_f64_1src<<<g, b>>>(out, stamps, n); }, s);
    // On CDNA the compiler packs these chains into v_pk_fma_f32 (2 FMAs per
    // instruction), so expect about twice the lane count per CU per clock there.
    run({"fma_f32", "v_fma_f32 / v_pk_fma_f32 (FP32 reference)", CHAINS, 8192},
        [&](int n) { k_fma_f32<<<g, b>>>(out, stamps, 0.9999f, n); }, s);
    run({"fe_mul_ref10", "ref10 fe_mul (10x25.5-bit limbs)", 2, 256},
        [&](int n) { k_fe_mul_ref10<<<g, b>>>(out, stamps, n); }, s);
    CHECK(gpuFree(stamps));
    CHECK(gpuFree(out));
    return 0;
}
