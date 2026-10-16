// baseline-onion-gpu: NIKIB0Y/onion-gpu (MIT, commit 24e5874) ported to the
// onionrig-grind/1 contract (docs/design/CONTRACT.md).
//
// The arithmetic is upstream's, untouched: fe25519.cuh and ge25519.cuh are
// byte-for-byte copies, and run_batch() below is upstream's fused Montgomery
// batch-inversion loop. What changed, and why:
//   - Start points: upstream derives a secret scalar per thread on the GPU. Here
//     every thread starts at A0 + k_base*(8B) from the rig's public A0, so no
//     secret ever touches the device (split-key search).
//   - Hits: report the signed offset k, not key material, for many targets.
//   - Host: the contract's --info / --verify / --grind instead of upstream's CLI.
// Upstream's launch geometry and batch sizes are kept as the tuning baseline.

#include "gpu.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "fe25519.cuh"
#include "ge25519.cuh"
#include "ge_extra.cuh"
#include "json_lite.h"

#ifndef THREADS_PER_BLOCK
#define THREADS_PER_BLOCK 128
#endif
#ifndef BLOCKS
#define BLOCKS 512
#endif
#ifndef INNER_BATCH
#define INNER_BATCH 16896  // max keys per thread per launch (44 * BINV_N); upstream's value
#endif
#ifndef BINV_N
#define BINV_N 384         // keys per batch Z-inversion
#endif
#ifndef MIN_BLOCKS_PER_SM
#define MIN_BLOCKS_PER_SM 3
#endif
#define TOTAL_THREADS ((unsigned long long)THREADS_PER_BLOCK * BLOCKS)
#define MAX_TARGETS 256
#define MAX_HITS (1u << 20)  // hits captured per launch
#define FAST_BITS 60         // 12 characters fit one masked 64-bit compare

static const char *CONTRACT = "onionrig-grind/1";
static const char *NAME = "baseline-onion-gpu";
static const char *VERSION = "1.0";

// ---------------------------------------------------------------------------
// Device state
// ---------------------------------------------------------------------------
__constant__ ge_cached c_step;                 // 8B in cached affine form (upstream)
__constant__ ge_p3 c_a0;                       // the public start point
__constant__ int c_ntargets;
__constant__ uint64_t c_mask[MAX_TARGETS];     // fast path: first <= 60 bits
__constant__ uint64_t c_value[MAX_TARGETS];
__constant__ uint8_t c_pattern[MAX_TARGETS][32];  // full target bits, stream order
__constant__ int c_bits[MAX_TARGETS];

__device__ ge_cached d_step_out;
__device__ ge_p3 d_point_out;
__device__ int d_ok;

struct Shared {
    unsigned int stored;  // hit slots claimed (may exceed MAX_HITS)
    unsigned int fault;   // the chain disagreed with scalar multiplication
};

__global__ void init_step_kernel() {  // upstream
    if (threadIdx.x || blockIdx.x) return;
    uint8_t sc[32] = {0};
    sc[0] = 8;
    ge_p3 S;
    ge_scalarmult_base(&S, sc);
    ge_cached_from_p3(&d_step_out, &S);
}

__global__ void decode_kernel(const uint8_t *bytes) {
    if (threadIdx.x || blockIdx.x) return;
    d_ok = ge_frombytes(&d_point_out, bytes) ? 1 : 0;
}

// Encodes A0 + k*(8B) for each k: exercises decoding, signed offsets, and the
// start-point path against the shared vectors.
__global__ void offset_kernel(const uint8_t *a0, const int64_t *ks, uint8_t *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    ge_p3 A, R;
    if (!ge_frombytes(&A, a0)) { out[i * 32] = 0xff; out[i * 32 + 31] = 0xff; return; }
    ge_at_offset(&R, &A, ks[i]);
    ge_p3_tobytes(out + i * 32, &R);
}

__global__ void encode_kernel(const uint8_t *in, uint8_t *out, int *ok, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    ge_p3 P;
    ok[i] = ge_frombytes(&P, in + i * 32) ? 1 : 0;
    if (ok[i]) ge_p3_tobytes(out + i * 32, &P);
}

// ---------------------------------------------------------------------------
// The batch: upstream's incremental chain with fused batch inversion, verbatim.
// ---------------------------------------------------------------------------
template <typename Sink>
__device__ __forceinline__ void run_batch(ge_p3 *P, fe *Wsave, fe *Zsave, int batch, Sink sink) {
    fe prefix;
    fe_1(prefix);
    #pragma unroll 1
    for (int j = 0; j < batch; j++) {
        fe_mul(Wsave[j], P->Y, prefix);
        fe_copy(Zsave[j], P->Z);
        fe_mul(prefix, prefix, P->Z);
        ge_p1p1 t;
        ge_madd(&t, P, &c_step);
        ge_p1p1_to_p3(P, &t);
    }
    fe inv;
    fe_invert(inv, prefix);
    #pragma unroll 1
    for (int k = batch - 1; k >= 0; k--) {
        fe y;
        fe_mul(y, Wsave[k], inv);
        sink(k, y);
        fe_mul(inv, inv, Zsave[k]);
    }
}

// Exact match of the y encoding against target t, over all of its bits.
__device__ bool matches_target(const uint8_t ybytes[32], int t) {
    const int bits = c_bits[t];
    for (int i = 0; i < bits / 8; i++)
        if (ybytes[i] != c_pattern[t][i]) return false;
    const int rest = bits % 8;
    if (rest == 0) return true;
    const uint8_t mask = (uint8_t)(0xff << (8 - rest));
    return (ybytes[bits / 8] & mask) == c_pattern[t][bits / 8];
}

// Cold path: confirm the full prefix, re-derive the point the slow way (a
// different code path from the chain), and only then report k.
__device__ __noinline__ void handle_candidate(int64_t k, const fe y, int64_t *hits, Shared *sh) {
    uint8_t ybytes[32];
    fe_tobytes(ybytes, y);
    bool any = false;
    for (int t = 0; t < c_ntargets && !any; t++) any = matches_target(ybytes, t);
    if (!any) return;

    ge_p3 R;
    ge_at_offset(&R, &c_a0, k);
    uint8_t pub[32];
    ge_p3_tobytes(pub, &R);
    bool same = (pub[31] & 0x7f) == ybytes[31];
    for (int i = 0; i < 31; i++) same = same && pub[i] == ybytes[i];
    if (!same) { atomicExch(&sh->fault, 1u); return; }

    unsigned int slot = atomicAdd(&sh->stored, 1u);
    if (slot < MAX_HITS) hits[slot] = k;
}

GRIND_LAUNCH_BOUNDS(THREADS_PER_BLOCK, MIN_BLOCKS_PER_SM)
__global__ void grind_kernel(int64_t k_origin, int keys_per_thread, int64_t k_end,
                             int64_t *__restrict__ hits, Shared *__restrict__ sh,
                             unsigned long long *__restrict__ counter) {
    const unsigned long long gtid = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    const int64_t k_base =
        k_origin + (int64_t)(gtid * (unsigned long long)keys_per_thread);
    if (k_base >= k_end) return;

    ge_p3 P;
    ge_at_offset(&P, &c_a0, k_base);
    fe Wsave[BINV_N], Zsave[BINV_N];
    unsigned long long local_count = 0;

    for (int base = 0; base < keys_per_thread; base += BINV_N) {
        const int batch = (base + BINV_N <= keys_per_thread) ? BINV_N : (keys_per_thread - base);
        run_batch(&P, Wsave, Zsave, batch, [&](int j, const fe y) {
            const uint64_t v = fe_low8_be(y);
            for (int t = 0; t < c_ntargets; t++) {
                if ((v & c_mask[t]) == c_value[t]) {
                    handle_candidate(k_base + base + j, y, hits, sh);
                    break;
                }
            }
        });
        local_count += batch;
    }
    atomicAdd(counter, local_count);
}

// ---------------------------------------------------------------------------
// Host: the grinder
// ---------------------------------------------------------------------------
static int alphabet_index(char c) {
    if (c >= 'a' && c <= 'z') return c - 'a';
    if (c >= '2' && c <= '7') return 26 + (c - '2');
    return -1;
}

static bool hex_decode(const std::string &hex, uint8_t *out, size_t n) {
    if (hex.size() != 2 * n) return false;
    for (size_t i = 0; i < n; i++) {
        unsigned v;
        if (std::sscanf(hex.c_str() + 2 * i, "%2x", &v) != 1) return false;
        out[i] = (uint8_t)v;
    }
    return true;
}

class Grinder {
  public:
    void init() {
        GPU_CHECK(gpuMalloc(&d_hits_, sizeof(int64_t) * MAX_HITS));
        GPU_CHECK(gpuMalloc(&d_shared_, sizeof(Shared)));
        GPU_CHECK(gpuMalloc(&d_counter_, sizeof(unsigned long long)));
        GPU_CHECK(gpuMalloc(&d_bytes_, 32));
        init_step_kernel<<<1, 1>>>();
        GPU_CHECK(gpuDeviceSynchronize());
        ge_cached step;
        GPU_CHECK(gpuMemcpyFromSymbol(&step, d_step_out, sizeof(step)));
        GPU_CHECK(gpuMemcpyToSymbol(c_step, &step, sizeof(step)));
    }

    bool set_start(const uint8_t a0[32]) {
        GPU_CHECK(gpuMemcpy(d_bytes_, a0, 32, gpuMemcpyHostToDevice));
        decode_kernel<<<1, 1>>>(d_bytes_);
        GPU_CHECK(gpuDeviceSynchronize());
        int ok = 0;
        GPU_CHECK(gpuMemcpyFromSymbol(&ok, d_ok, sizeof(ok)));
        if (!ok) return false;
        ge_p3 point;
        GPU_CHECK(gpuMemcpyFromSymbol(&point, d_point_out, sizeof(point)));
        GPU_CHECK(gpuMemcpyToSymbol(c_a0, &point, sizeof(point)));
        return true;
    }

    // Returns an error message, or "" on success.
    std::string set_targets(const std::vector<std::string> &targets) {
        if (targets.empty() || targets.size() > MAX_TARGETS) return "need 1..256 targets";
        std::vector<uint64_t> mask(targets.size()), value(targets.size());
        std::vector<int> bits(targets.size());
        std::vector<uint8_t> pattern(targets.size() * 32, 0);
        for (size_t t = 0; t < targets.size(); t++) {
            const std::string &s = targets[t];
            if (s.empty() || s.size() > 49) return "target length must be 1..49";
            bits[t] = 5 * (int)s.size();
            for (size_t c = 0; c < s.size(); c++) {
                int v = alphabet_index(s[c]);
                if (v < 0) return "target has a character outside a-z2-7";
                for (int b = 0; b < 5; b++) {   // stream bit 5c+b, MSB first
                    int bit = (int)(5 * c) + b;
                    if ((v >> (4 - b)) & 1) pattern[t * 32 + bit / 8] |= (uint8_t)(0x80 >> (bit % 8));
                }
            }
            const int fast = std::min(bits[t], FAST_BITS);
            uint64_t head = 0;
            for (int i = 0; i < 8; i++) head = (head << 8) | pattern[t * 32 + i];
            mask[t] = ~0ull << (64 - fast);
            value[t] = head & mask[t];
        }
        int n = (int)targets.size();
        GPU_CHECK(gpuMemcpyToSymbol(c_ntargets, &n, sizeof(n)));
        GPU_CHECK(gpuMemcpyToSymbol(c_mask, mask.data(), sizeof(uint64_t) * n));
        GPU_CHECK(gpuMemcpyToSymbol(c_value, value.data(), sizeof(uint64_t) * n));
        GPU_CHECK(gpuMemcpyToSymbol(c_bits, bits.data(), sizeof(int) * n));
        GPU_CHECK(gpuMemcpyToSymbol(c_pattern, pattern.data(), 32 * (size_t)n));
        return "";
    }

    struct Launch {
        std::vector<int64_t> hits;
        unsigned long long candidates = 0;
        bool fault = false;
        unsigned int dropped = 0;
    };

    // Every thread walks keys_per_thread consecutive offsets from k_origin + gtid*keys_per_thread;
    // threads that would start at or past k_end exit at once.
    Launch launch(int64_t k_origin, int keys_per_thread, int64_t k_end = INT64_MAX) {
        Shared sh = {0, 0};
        unsigned long long zero = 0;
        GPU_CHECK(gpuMemcpy(d_shared_, &sh, sizeof(sh), gpuMemcpyHostToDevice));
        GPU_CHECK(gpuMemcpy(d_counter_, &zero, sizeof(zero), gpuMemcpyHostToDevice));
        grind_kernel<<<BLOCKS, THREADS_PER_BLOCK>>>(k_origin, keys_per_thread, k_end, d_hits_,
                                                   d_shared_, d_counter_);
        GPU_CHECK(gpuGetLastError());
        GPU_CHECK(gpuDeviceSynchronize());
        Launch out;
        GPU_CHECK(gpuMemcpy(&sh, d_shared_, sizeof(sh), gpuMemcpyDeviceToHost));
        GPU_CHECK(gpuMemcpy(&out.candidates, d_counter_, sizeof(zero), gpuMemcpyDeviceToHost));
        out.fault = sh.fault != 0;
        const unsigned int n = std::min(sh.stored, MAX_HITS);
        out.dropped = sh.stored - n;
        out.hits.resize(n);
        if (n) GPU_CHECK(gpuMemcpy(out.hits.data(), d_hits_, sizeof(int64_t) * n,
                                   gpuMemcpyDeviceToHost));
        return out;
    }

  private:
    int64_t *d_hits_ = nullptr;
    Shared *d_shared_ = nullptr;
    unsigned long long *d_counter_ = nullptr;
    uint8_t *d_bytes_ = nullptr;
};

// ---------------------------------------------------------------------------
// Host: protocol I/O
// ---------------------------------------------------------------------------
static void emit(const std::string &line) {
    std::cout << line << "\n" << std::flush;
}

static std::string device_name() {
    gpuDeviceProp prop;
    GPU_CHECK(gpuGetDeviceProperties(&prop, 0));
#if defined(__HIPCC__)
    return std::string(prop.name) + " (" + prop.gcnArchName + ", hip)";
#else
    return std::string(prop.name) + " (sm_" + std::to_string(prop.major) +
           std::to_string(prop.minor) + ", cuda)";
#endif
}

static int cmd_info() {
    emit(std::string("{\"contract\": \"") + CONTRACT + "\", \"name\": \"" + NAME +
         "\", \"version\": \"" + VERSION + "\", \"model\": \"\", \"device\": \"" + device_name() +
         "\", \"upstream\": \"NIKIB0Y/onion-gpu@24e5874\"}");
    return 0;
}

static void emit_hits(const std::vector<int64_t> &hits) {
    const size_t per_line = 100000;  // keeps each line far below the rig's 16 MiB limit
    for (size_t first = 0; first < hits.size(); first += per_line) {
        std::string line = "{\"event\": \"hits\", \"k\": [";
        const size_t last = std::min(hits.size(), first + per_line);
        for (size_t i = first; i < last; i++) {
            if (i > first) line += ", ";
            line += "\"" + std::to_string(hits[i]) + "\"";
        }
        emit(line + "]}");
    }
}

static void emit_error(const std::string &message) {
    std::string escaped;
    for (char c : message) escaped += (c == '"' || c == '\\') ? std::string("\\") + c : std::string(1, c);
    emit("{\"event\": \"error\", \"message\": \"" + escaped + "\"}");
}

static bool g_protocol = false;  // true while speaking the grind protocol

[[noreturn]] void gpu_fatal(const char *file, int line, const char *what) {
    const std::string where = std::string(file) + ":" + std::to_string(line);
    if (g_protocol) emit_error("GPU error at " + where + ": " + what);
    std::fprintf(stderr, "GPU error at %s: %s\n", where.c_str(), what);
    std::exit(3);
}

static std::atomic<bool> g_stop{false};

static void watch_stdin() {
    std::string line;
    while (std::getline(std::cin, line)) {
        Json msg;
        std::string err;
        if (JsonReader(line).parse(msg, err) && msg.get("cmd") && msg.get("cmd")->text == "stop")
            break;
    }
    g_stop = true;  // a stop command, or EOF because the rig went away
}

// Parse and apply the start command; returns its duration (<0 for none), or a
// non-empty error.
static std::string apply_start(Grinder &grinder, const std::string &line, double &seconds) {
    Json start;
    std::string err;
    if (!JsonReader(line).parse(start, err) || !start.get("cmd") || start.get("cmd")->text != "start")
        return "expected a start command";
    uint8_t a0[32];
    const Json *a0_hex = start.get("a0");
    if (!a0_hex || !hex_decode(a0_hex->text, a0, 32) || !grinder.set_start(a0))
        return "a0 is not a valid ed25519 point encoding";
    std::vector<std::string> targets;
    if (const Json *ts = start.get("targets"))
        for (const Json &t : ts->items) targets.push_back(t.text);
    if (std::string problem = grinder.set_targets(targets); !problem.empty()) return problem;
    const Json *secs = start.get("seconds");
    seconds = (secs && secs->kind == Json::Number) ? secs->number : -1.0;
    return "";
}

static int cmd_grind() {
    g_protocol = true;
    Grinder grinder;
    grinder.init();
    emit("{\"event\": \"ready\"}");

    std::string line;
    if (!std::getline(std::cin, line)) return 1;
    double seconds = -1.0;
    if (std::string problem = apply_start(grinder, line, seconds); !problem.empty()) {
        emit_error(problem);
        return 1;
    }
    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();
    const auto deadline = t0 + std::chrono::duration<double>(std::max(seconds, 0.0));
    std::thread(watch_stdin).detach();

    // Upstream's INNER_BATCH per thread, shrunk only if a launch takes over 2 s, so
    // stop latency and hit flushing stay bounded on slow GPUs without changing
    // the baseline on any GPU fast enough not to need it.
    int keys_per_thread = INNER_BATCH;
    int64_t cursor = 0;
    unsigned long long total = 0;
    auto last_progress = t0;
    while (!g_stop && (seconds < 0 || clock::now() < deadline)) {
        const auto l0 = clock::now();
        Grinder::Launch r = grinder.launch(cursor, keys_per_thread);
        const double took = std::chrono::duration<double>(clock::now() - l0).count();
        cursor += (int64_t)(TOTAL_THREADS * (unsigned long long)keys_per_thread);
        if (r.fault) {
            emit_error("GPU fault: the incremental chain disagreed with scalar multiplication");
            return 2;
        }
        if (r.dropped)
            std::fprintf(stderr, "warning: hit buffer full, %u hits dropped this launch\n", r.dropped);
        total += r.candidates;
        emit_hits(r.hits);
        if (took > 2.0)
            keys_per_thread = std::max(BINV_N, (int)(keys_per_thread * 1.5 / took));
        const auto now = clock::now();
        if (now - last_progress > std::chrono::seconds(5)) {
            emit("{\"event\": \"progress\", \"candidates\": \"" + std::to_string(total) + "\"}");
            last_progress = now;
        }
    }
    emit("{\"event\": \"done\", \"candidates\": \"" + std::to_string(total) + "\"}");
    std::fflush(stdout);
    std::_Exit(0);  // the stdin watcher may still be blocked in getline
}

// ---------------------------------------------------------------------------
// Host: --verify against the shared vectors (flat .txt form)
// ---------------------------------------------------------------------------
struct Check {
    std::string name;
    bool ok;
};

static std::vector<std::vector<std::string>> read_cases(const std::string &path, const char *op) {
    std::vector<std::vector<std::string>> cases;
    std::ifstream in(path);
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream words(line);
        std::vector<std::string> fields;
        for (std::string w; words >> w;) fields.push_back(w);
        if (!fields.empty() && fields[0] == op) cases.push_back(fields);
    }
    return cases;
}

static Check verify_offsets(const std::string &dir) {
    // offset <a0> <k> <out>, all sharing one a0
    auto cases = read_cases(dir + "/offset.txt", "offset");
    if (cases.empty()) return {"offset", false};
    std::vector<int64_t> ks;
    for (auto &c : cases) ks.push_back(std::stoll(c[2]));
    uint8_t a0[32];
    if (!hex_decode(cases[0][1], a0, 32)) return {"offset", false};
    uint8_t *d_a0, *d_out;
    int64_t *d_ks;
    const int n = (int)ks.size();
    GPU_CHECK(gpuMalloc(&d_a0, 32));
    GPU_CHECK(gpuMalloc(&d_out, 32 * (size_t)n));
    GPU_CHECK(gpuMalloc(&d_ks, sizeof(int64_t) * n));
    GPU_CHECK(gpuMemcpy(d_a0, a0, 32, gpuMemcpyHostToDevice));
    GPU_CHECK(gpuMemcpy(d_ks, ks.data(), sizeof(int64_t) * n, gpuMemcpyHostToDevice));
    offset_kernel<<<(n + 63) / 64, 64>>>(d_a0, d_ks, d_out, n);
    GPU_CHECK(gpuDeviceSynchronize());
    std::vector<uint8_t> out(32 * (size_t)n);
    GPU_CHECK(gpuMemcpy(out.data(), d_out, out.size(), gpuMemcpyDeviceToHost));
    GPU_CHECK(gpuFree(d_a0)); GPU_CHECK(gpuFree(d_out)); GPU_CHECK(gpuFree(d_ks));
    bool ok = true;
    for (int i = 0; i < n; i++) {
        uint8_t want[32];
        ok = ok && hex_decode(cases[i][3], want, 32) && std::memcmp(want, &out[32 * i], 32) == 0;
    }
    return {"offset", ok};
}

static Check verify_decode(const std::string &dir) {
    // decode <in> <out|invalid>
    auto cases = read_cases(dir + "/decode.txt", "decode");
    const int n = (int)cases.size();
    if (n == 0) return {"decode", false};
    std::vector<uint8_t> in(32 * (size_t)n);
    for (int i = 0; i < n; i++)
        if (!hex_decode(cases[i][1], &in[32 * i], 32)) return {"decode", false};
    uint8_t *d_in, *d_out;
    int *d_ok;
    GPU_CHECK(gpuMalloc(&d_in, in.size()));
    GPU_CHECK(gpuMalloc(&d_out, in.size()));
    GPU_CHECK(gpuMalloc(&d_ok, sizeof(int) * n));
    GPU_CHECK(gpuMemcpy(d_in, in.data(), in.size(), gpuMemcpyHostToDevice));
    encode_kernel<<<(n + 63) / 64, 64>>>(d_in, d_out, d_ok, n);
    GPU_CHECK(gpuDeviceSynchronize());
    std::vector<uint8_t> out(in.size());
    std::vector<int> oks(n);
    GPU_CHECK(gpuMemcpy(out.data(), d_out, out.size(), gpuMemcpyDeviceToHost));
    GPU_CHECK(gpuMemcpy(oks.data(), d_ok, sizeof(int) * n, gpuMemcpyDeviceToHost));
    GPU_CHECK(gpuFree(d_in)); GPU_CHECK(gpuFree(d_out)); GPU_CHECK(gpuFree(d_ok));
    bool ok = true;
    for (int i = 0; i < n; i++) {
        if (cases[i][2] == "invalid") { ok = ok && !oks[i]; continue; }
        uint8_t want[32];
        ok = ok && oks[i] && hex_decode(cases[i][2], want, 32) &&
             std::memcmp(want, &out[32 * i], 32) == 0;
    }
    return {"decode", ok};
}

static Check verify_kat(Grinder &grinder, const std::vector<std::string> &c) {
    // kat <name> <a0> <nt> <targets...> <lo> <hi> <nh> <hits...>
    const std::string name = "kat:" + c[1];
    uint8_t a0[32];
    if (!hex_decode(c[2], a0, 32) || !grinder.set_start(a0)) return {name, false};
    const size_t nt = std::stoul(c[3]);
    std::vector<std::string> targets(c.begin() + 4, c.begin() + 4 + nt);
    if (!grinder.set_targets(targets).empty()) return {name, false};
    const int64_t lo = std::stoll(c[4 + nt]), hi = std::stoll(c[5 + nt]);
    const size_t nh = std::stoul(c[6 + nt]);
    std::vector<int64_t> want;
    for (size_t i = 0; i < nh; i++) want.push_back(std::stoll(c[7 + nt + i]));

    // One launch from k = lo; threads that would start past hi exit at once.
    Grinder::Launch r = grinder.launch(lo, INNER_BATCH, hi);
    std::vector<int64_t> got;
    for (int64_t k : r.hits)
        if (k >= lo && k < hi) got.push_back(k);
    std::sort(got.begin(), got.end());
    return {name, !r.fault && r.dropped == 0 && got == want};
}

static int cmd_verify(const std::string &dir) {
    Grinder grinder;
    grinder.init();
    std::vector<Check> checks = {verify_decode(dir), verify_offsets(dir)};
    for (auto &c : read_cases(dir + "/grind.txt", "kat")) checks.push_back(verify_kat(grinder, c));
    int failed = 0;
    std::string list;
    for (size_t i = 0; i < checks.size(); i++) {
        failed += !checks[i].ok;
        list += std::string(i ? ", " : "") + "{\"name\": \"" + checks[i].name + "\", \"ok\": " +
                (checks[i].ok ? "true" : "false") + "}";
    }
    emit("{\"passed\": " + std::to_string(checks.size() - failed) + ", \"failed\": " +
         std::to_string(failed) + ", \"checks\": [" + list + "]}");
    return (failed == 0 && checks.size() >= 4) ? 0 : 1;
}

int main(int argc, char **argv) {
    std::ios::sync_with_stdio(false);
    if (argc == 2 && std::string(argv[1]) == "--info") return cmd_info();
    if (argc == 2 && std::string(argv[1]) == "--grind") return cmd_grind();
    if (argc == 3 && std::string(argv[1]) == "--verify") return cmd_verify(argv[2]);
    std::fprintf(stderr, "usage: %s --info | --verify VECTORS_DIR | --grind\n", argv[0]);
    return 2;
}
