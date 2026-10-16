// Primitive-throughput microbenchmarks: settle the integer-vs-FP64 field-multiply
// question with measurements instead of datasheets (docs/design/CDNA-PERF-REFERENCE.md s6).
//
// Each kernel runs CHAINS independent dependency chains per thread (enough
// instruction-level parallelism to saturate the pipe) of one primitive, in a
// recurrence the compiler cannot hoist or fold. Each test sizes its loop so one
// sample takes about TARGET_MS on this GPU, then reports the mean of SAMPLES
// samples (mean, never min, as in the rig) and their relative spread. Output is
// one JSON line per test: sustained ops/s and ops per compute unit per nominal clock.
//
// Build: make cuda | make hip HIP_ARCHS="gfx90a gfx950"; `make isa` confirms each
// kernel compiles to the instruction it claims to measure.

#if defined(__HIPCC__)
  #include <hip/hip_runtime.h>
  #define gpuMalloc hipMalloc
  #define gpuFree hipFree
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
#include <cstdlib>
#include <cstring>
#include <string>

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
#define SAMPLES 5
#define TARGET_MS 100.0

// 64-bit result of a 32x32 multiply plus a 64-bit addend: v_mad_u64_u32 on CDNA.
__global__ void k_mad_u64_u32(uint64_t *out, uint32_t m, int iters) {
    uint64_t acc[CHAINS];
    for (int c = 0; c < CHAINS; c++) acc[c] = threadIdx.x + 977u * c + 1;
    for (int i = 0; i < iters; i++) {
        #pragma unroll
        for (int c = 0; c < CHAINS; c++) acc[c] = (uint64_t)(uint32_t)acc[c] * (m + c) + acc[c];
    }
    uint64_t s = 0;
    for (int c = 0; c < CHAINS; c++) s ^= acc[c];
    out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}

// Low 32 bits of a 32x32 product: v_mul_lo_u32.
__global__ void k_mul_lo_u32(uint64_t *out, uint32_t m, int iters) {
    uint32_t acc[CHAINS];
    for (int c = 0; c < CHAINS; c++) acc[c] = threadIdx.x * 2 + 2 * c + 1;
    for (int i = 0; i < iters; i++) {
        #pragma unroll
        for (int c = 0; c < CHAINS; c++) acc[c] *= (m | 1) + 2 * c;
    }
    uint32_t s = 0;
    for (int c = 0; c < CHAINS; c++) s ^= acc[c];
    out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}

// High 32 bits of a 32x32 product, plus a cheap add to keep the chain alive.
__global__ void k_mul_hi_u32(uint64_t *out, uint32_t m, int iters) {
    uint32_t acc[CHAINS];
    for (int c = 0; c < CHAINS; c++) acc[c] = threadIdx.x + 0x9e3779b9u * (c + 1);
    for (int i = 0; i < iters; i++) {
        #pragma unroll
        for (int c = 0; c < CHAINS; c++) acc[c] = __umulhi(acc[c], m + c) + acc[c];
    }
    uint32_t s = 0;
    for (int c = 0; c < CHAINS; c++) s ^= acc[c];
    out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}

// Fused multiply-add in double precision: v_fma_f64.
__global__ void k_fma_f64(uint64_t *out, double m, int iters) {
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
}

// Fused multiply-add in single precision: the full-rate reference.
__global__ void k_fma_f32(uint64_t *out, float m, int iters) {
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
}

// A whole ref10 field multiply (10 x 25.5-bit limbs, int64 accumulators): the
// unit the baseline's hot loop is built from. Two chains for ILP.
__global__ void k_fe_mul_ref10(uint64_t *out, int iters) {
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
}

struct Test {
    const char *name;
    const char *instruction;  // what one counted op compiles to on CDNA
    double ops_per_iter;      // counted ops per thread per loop iteration
    int iters;                // minimum loop count; scaled up to reach TARGET_MS
};

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

template <typename Launch>
static void run(const Test &t, Launch launch, int cus, int clock_khz) {
    gpuEvent_t start, stop;
    CHECK(gpuEventCreate(&start));
    CHECK(gpuEventCreate(&stop));
    launch(t.iters / 16);  // warm-up: clocks up, code resident
    CHECK(gpuGetLastError());
    CHECK(gpuDeviceSynchronize());
    // Rates differ by up to 64x between parts (FP64 above all), so a fixed loop
    // count gives some GPUs samples too short to time well.
    const double probe = timed_ms(t, launch, t.iters, start, stop);
    const int iters = (int)std::clamp(t.iters * TARGET_MS / probe, (double)t.iters, 1e8);
    // Untimed full-length samples first: an idle GPU takes a few hundred ms to clock up.
    for (int i = 0; i < 3; i++) timed_ms(t, launch, iters, start, stop);
    double sum = 0, lo = 1e300, hi = 0;
    for (int i = 0; i < SAMPLES; i++) {
        const double ms = timed_ms(t, launch, iters, start, stop);
        sum += ms;
        lo = std::min(lo, ms);
        hi = std::max(hi, ms);
    }
    CHECK(gpuEventDestroy(start));
    CHECK(gpuEventDestroy(stop));
    const double ms = sum / SAMPLES;
    const double ops = (double)BLOCKS * THREADS * t.ops_per_iter * iters;
    const double per_s = ops / (ms / 1e3);
    const double per_cu_clock = per_s / ((double)cus * clock_khz * 1e3);
    std::printf("{\"test\": \"%s\", \"instruction\": \"%s\", \"iters\": %d, "
                "\"samples\": %d, \"seconds\": %.6f, \"rel_spread\": %.4f, "
                "\"ops_per_s\": %.6e, \"ops_per_cu_per_clock\": %.3f}\n",
                t.name, t.instruction, iters, SAMPLES, ms / 1e3, (hi - lo) / ms, per_s,
                per_cu_clock);
    std::fflush(stdout);
}

int main() {
    gpuDeviceProp prop;
    CHECK(gpuGetDeviceProperties(&prop, 0));
    int clock_khz = 0;
    CHECK(gpuDeviceGetAttribute(&clock_khz, gpuDevAttrClockRate, 0));
    const int cus = prop.multiProcessorCount;
#if defined(__HIPCC__)
    const std::string arch = prop.gcnArchName;
#else
    const std::string arch = "sm_" + std::to_string(prop.major) + std::to_string(prop.minor);
#endif
    std::printf("{\"device\": \"%s\", \"arch\": \"%s\", \"compute_units\": %d, "
                "\"nominal_clock_mhz\": %.0f, \"chains\": %d}\n",
                prop.name, arch.c_str(), cus, clock_khz / 1e3, CHAINS);

    uint64_t *out;
    CHECK(gpuMalloc(&out, sizeof(uint64_t) * BLOCKS * THREADS));
    const dim3 g(BLOCKS), b(THREADS);
    run({"mad_u64_u32", "v_mad_u64_u32", CHAINS, 4096},
        [&](int n) { k_mad_u64_u32<<<g, b>>>(out, 0x9e3779b1u, n); }, cus, clock_khz);
    run({"mul_lo_u32", "v_mul_lo_u32", CHAINS, 4096},
        [&](int n) { k_mul_lo_u32<<<g, b>>>(out, 0x85ebca6bu, n); }, cus, clock_khz);
    run({"mul_hi_u32_plus_add", "v_mul_hi_u32 + v_add_u32", CHAINS, 4096},
        [&](int n) { k_mul_hi_u32<<<g, b>>>(out, 0xc2b2ae35u, n); }, cus, clock_khz);
    run({"fma_f64", "v_fma_f64", CHAINS, 4096},
        [&](int n) { k_fma_f64<<<g, b>>>(out, 0.9999, n); }, cus, clock_khz);
    // On CDNA the compiler packs these chains into v_pk_fma_f32 (2 FMAs per
    // instruction), so expect about twice the lane count per CU per clock there.
    run({"fma_f32", "v_fma_f32 / v_pk_fma_f32 (FP32 reference)", CHAINS, 8192},
        [&](int n) { k_fma_f32<<<g, b>>>(out, 0.9999f, n); }, cus, clock_khz);
    run({"fe_mul_ref10", "ref10 fe_mul (10x25.5-bit limbs)", 2, 256},
        [&](int n) { k_fe_mul_ref10<<<g, b>>>(out, n); }, cus, clock_khz);
    CHECK(gpuFree(out));
    return 0;
}
