# Literature review: high-throughput ed25519 / twisted-Edwards batch ECC on AMD CDNA GPUs

Research date: 2026-10-15. Focus: raising throughput of a Tor v3 vanity .onion generator whose hot
loop is batched ed25519 / twisted-Edwards affine point addition + Montgomery batch inversion over
F_p, p = 2^255 - 19, on AMD CDNA (HIP/ROCm). Priority: 2026 work; seminal pre-2026 included.

Method note: the `aqua-research` MCP index was queried early and repeatedly. Its corpus (3,474 papers,
newest 2026-10-11) is almost entirely ML/LLM/agent research. It has **zero** ECC/bignum/finite-field
crypto papers (confirmed by multiple paper+claim searches), but it is the best index for Topic 4
(GPU kernel auto-optimization). All crypto content below came from arXiv/IACR/GitHub/web.

Legend for confidence: [VERIFIED] = read numbers from paper text/abstract; [SECONDARY] = from a
reliable secondary source, not the paper itself; [UNVERIFIED] = could not confirm this pass.

================================================================================
TOPIC 1 — GPU frameworks for high-throughput EC / ed25519 batch operations
================================================================================

### gECC — "A GPU-based high-throughput framework for Elliptic Curve Cryptography"
- arXiv:2501.03245 (8 Jan 2025); accepted ACM TACO 2025. Authors: Qian Xiong et al.
  (Huazhong Univ. of Science & Technology; Univ. of Copenhagen; NVIDIA). Code: github CGCL-codes/gECC.
- THE most directly relevant framework. [VERIFIED from HTML v1]:
  - **Field representation: INTEGER radix-2^32.** 256-bit integer = base D=32, m=8 32-bit limbs.
    Modular multiplication built on IMAD / IADD3 integer instructions. No FP64 in gECC itself.
  - **Explicit integer-vs-FP64 finding (decisive for our fork):** they contrast against "DPF"
    (the FP64 approach, D=52) and against CGBN (NVIDIA integer bignum lib). On SM2, total
    instruction counts for modmul: **DPF-1 = 325 vs CGBN-1 = 227**; CGBN-1 (integer) OUTPERFORMS
    DPF-1 (FP64). Core thesis: "modular-multiplication efficiency depends heavily on the number of
    IMAD instructions"; they minimize IMAD. gECC modmul is 1.63x-1.72x over CGBN-1.
  - **Batching scheme (= our scheme):** batched EC ops in AFFINE coordinates using Montgomery's
    trick; inversions batched with a **Gather-Apply-Scatter** scheme that reduces inversions from
    32n to sp (sp = #streaming processors = 432 on A100). Batch sizes 2^11..2^16 per SP; example of
    2^20 simultaneous EC point additions. **Batching beats Jacobian coordinates when N > 20.**
  - **Absolute throughput (NVIDIA A100, SM2 curve, 256-bit prime):**
    - ECDSA sign: gECC 14,141,978 sig/s vs RapidEC 3,386,544.
    - ECDSA verify: gECC 4,372,853 sig/s vs RapidEC 773,481.
    - Field mul (FPMUL) peak 14,389,168 ops/s @ batch 2^14/SP (RapidEC ~3.9M).
    - UPMUL peak 6,837,619 ops/s @ batch 2^15/SP (RapidEC ~1.48M).
  - Headline speedups: 5.56x ECDSA / 4.94x ECDH vs state-of-the-art GPU (RapidEC). (Minor internal
    inconsistency: verify speedup quoted 5.56x in abstract, 5.65x in S5.2; gen 4.18x in text.)
  - **Curve: SM2 ONLY** (prime q = 2^256 - 2^224 - 2^96 + 2^64 - 1). Does NOT cover Curve25519,
    Ed25519, twisted Edwards, Tor, or vanity. NVIDIA-only (A100/V100).
- RELEVANCE: top reference. Its batching + Gather-Apply-Scatter inversion + "minimize mul
  instruction count" port directly to our ed25519/HIP loop. BUT its "integer beats FP64" conclusion
  is NVIDIA-specific (fast IMAD, and on many NVIDIA parts FP64 is throttled). On CDNA the economics
  differ (see Topic 2/3) — do NOT assume integer wins on AMD.

### DPF-ECC — FP64 ("double-precision floating-point computing power") ECC framework
- Lili Gao, Fangyu Zheng, Niall Emmart, et al.
  - IPDPS 2020, pp. 494-504: "DPF-ECC: Accelerating ECC with floating-point computing power of GPUs".
  - IEEE TIFS 2021, vol. 16, pp. 3988-4002: "DPF-ECC: A framework for efficient ECC with double
    precision floating-point computing power."
- The seminal FP64-for-ECC line of work; the "other side" of our design fork. Uses FP64 DFMA with
  D=52 radix. [UNVERIFIED] absolute ECDSA ops/s this pass (gECC cites it as [31]/[32] but prints no
  DPF throughput; IEEE Xplore paywalled). Need to pull TIFS 2021 tables for direct FP64 ECC numbers.
- RELEVANCE: the canonical FP64 ECC design. Directly informs the FP64 radix-2^51/2^52 branch.

### Emmart, Zheng & Weems — "Faster Modular Exponentiation Using Double Precision Floating Point
Arithmetic on the GPU" (ARITH 2018, pp. 130-137) [+ companion pp. 138-144 on Barrett quotient sel.]
- [SECONDARY] FP64 mantissa tricks: set implicit leading mantissa bit so all 52 explicit bits are
  usable; reuse constant exponent fields in column sums with a single correction; pre-init column
  sums to avoid corrective subtraction. 20%-34% faster than best prior GPU for 2048-4096-bit RSA.
  GTX Titan Black: 132.5K-161.9K modexp/s @1024-bit.
- RELEVANCE: foundational FP64-bignum techniques underpinning DPF-ECC; the "2 DFMA per limb-product"
  pattern for capturing hi/lo of a >mantissa product. A GPU-FP64 implementer must know this.

### RapidEC (SC22, "Accelerating Elliptic Curve Digital Signature Algorithms on GPUs")
- Prior SOTA GPU ECDSA for SM2 that gECC dethrones. On A100: ~3.39M sign/s, ~773K verify/s
  (from gECC's comparison tables). NVIDIA, SM2.

### Mahé & Chauvet — GPGPU Curve25519 (IACR ePrint 2014/198; hgpu.org)
- [SECONDARY] Curve25519 scalar multiplication in **< 1 microsecond on AMD Radeon R9 290X** (GCN 1.0);
  "methods applicable to any parallel architecture." Old (2014), latency-oriented (one scalarmult,
  not batched throughput), but the ONLY GPU Curve25519-on-AMD result found. Historical anchor.

### ICICLE (Ingonyama) — GPU ZK crypto library
- [VERIFIED from GitHub] Backends: **CUDA + CPU only** (official). Curves: bn254, bls12_377,
  bls12_381, bw6_761, grumpkin; fields babybear/stark252/m31/koalabear. Primitives MSM/NTT/ECNTT.
  **No Ed25519/Curve25519/twisted Edwards. No ROCm/HIP/AMD backend shipped** (extensible-backend
  arch exists; earlier Metal/Vulkan were aspirational, not in README). ZK-oriented, not key-grinding.
- RELEVANCE: confirms the gap — no mature AMD-native or ed25519-native GPU crypto library exists.

### Cheng et al. — "High-Throughput ECC Using AVX2 Vector Instructions" (SAC 2020, Univ. Luxembourg)
- [SECONDARY] CPU throughput-oriented Curve25519: 4 scalarmults in parallel on 64-bit lanes,
  ~32,318 scalarmult/s @ Skylake 2 GHz, vs ~21,000 for best latency-optimized AVX2. Not GPU, but
  the clearest statement of the throughput-vs-latency batching philosophy we are adopting on GPU.

================================================================================
TOPIC 2 — FP64 / FMA modular multiplication on GPUs  (THE design-fork topic)
================================================================================

### Berthomieu, Graillat, Lesnoff, Mary — "Multiword matrix multiplication over large finite fields
in floating-point arithmetic" (arXiv:2601.07508v2, 4 Feb 2026; Sorbonne/LIP6 numerical-analysis group)
- [VERIFIED from PDF text] Computes C = AB mod p in FP. Existing FP approach limited to primes of
  bitsize <= half mantissa (26 bits in FP64), inefficient near the limit. New **multiword
  decomposition**: A = sum alpha^i A_i, B = sum beta^j B_j (u,v words); product = uv modular products
  A_i B_j mod p. **A (2,2) decomposition raises the 26-bit limit to ~52 bits (full mantissa).**
  (1,v) decompositions handle intermediate bitsizes. Rigorous rounding-error analysis fixes the
  scaling parameters. Outperforms single-word FP for bitsizes as low as 23; handles up to 52 bits
  with high performance. CPU AND GPU benchmarks (figure names incl. gpu_square).
- RELEVANCE: the newest (2026) rigorous FP64-modmul paper. Targets mod-p *matrix mult* (computer
  algebra) with primes <=52 bits — NOT a single 255-bit prime, NOT ECC point ops. But two ideas are
  directly useful: (a) the multiword/limb decomposition + proven error bounds is the correctness
  framework for a radix-2^51(ish) FP64 field mul; (b) it exploits FP64 BLAS/matrix units — a
  speculative but high-upside idea for us (cast a batch of field muls as small FP64 matrix products
  to hit the MFMA matrix-FP64 path; see Topic 3 for the 163 TFLOPS matrix-FP64 figure on MI300X).

### FP64 radix for 2^255-19 — technical nuance (synthesized)
- [SECONDARY] Canonical 64-bit CPU layout = 5 limbs of 51 bits (radix 2^51, Langley donna_c64).
  On FP64 the 51-bit limb *stores* exactly (< 2^53 mantissa), but a 51x51 limb PRODUCT is up to 102
  bits and does NOT fit one double. So an FP64 field mul needs either ~2 DFMA per limb-product to
  capture hi/lo (the Emmart/DPF technique, D=52), or a smaller radix (~2^43-2^48, 6 limbs) so column
  sums stay exactly representable. This is the crux of the "FP64 radix-2^51" branch and must be
  designed with the Berthomieu et al. error analysis.
- gECC's NVIDIA result (integer 227 < FP64 325 instrs) is the counter-evidence, but NVIDIA-specific.

================================================================================
TOPIC 3 — AMD CDNA / HIP / ROCm bignum / ECC / modular-arithmetic kernels
================================================================================

### State of the field: essentially EMPTY.
- No published ECC/bignum/modular-arithmetic GPU kernels on MI210/MI250X/MI300X/MI355X found. No
  keys/sec or modmul-throughput numbers on any MI-series part in the literature. This is a genuine
  gap — we will be generating the first such numbers and must self-benchmark.
- Mahé & Chauvet R9 290X (Topic 1) is the closest AMD GPU ECC datapoint, but 2014/GCN1.
- Visegrády (Topic 5) benchmarks on an AMD Ryzen AI PRO 350 with libsecp256k1 — AMD, but CPU/APU.

### AMD FP64 hardware facts (decisive context for the fork) [SECONDARY, ROCm docs + Wikipedia]
- MI300X (CDNA3): **Vector FP64 81.7 TFLOPS** (128 FLOP/clk/CU); **Matrix FP64 163.4 TFLOPS**
  (256 FLOP/clk/CU) via MFMA. FP64 is a first-class, full-featured path on CDNA (unlike many NVIDIA
  parts where FP64 is throttled).
- MI250X (CDNA2): ~47.9 vector / ~95.7 matrix FP64 TFLOPS [UNVERIFIED — Wikipedia table].
- MI355X (CDNA4): ~78.6 vector / ~157.3 matrix FP64 TFLOPS [UNVERIFIED — Wikipedia; note CDNA4 did
  NOT increase FP64 over CDNA3, and AMD has signalled de-emphasis of FP64 in the AI-focused CDNA4].
- Integer side: ROCm precision docs state 64-bit integer ops "may need several instructions on the
  32-bit integer ALUs" and non-native types run below full throughput; 32-bit integer MULTIPLY rate
  on CDNA is NOT clearly full-rate (V_MUL_LO/HI_U32 ~16-cycle latency on Vega/RX; throughput ratio
  not documented). [UNVERIFIED precise throughput — requires a microbenchmark.]
- IMPLICATION: the a-priori case for the FP64 radix branch is materially STRONGER on CDNA than on
  NVIDIA, precisely inverting gECC's conclusion — because CDNA gives full-rate FP64 FMA while its
  32-bit integer multiply is (at best) unclear. This must be settled by microbenchmarking both
  field-mul kernels on an actual MI-series GPU before committing.

================================================================================
TOPIC 4 — AI / LLM / RL-driven GPU kernel auto-optimization
================================================================================

KEY ASSESSMENT: every item below optimizes ML operators (GEMM/attention/conv) against PyTorch/Triton
baselines. **NONE targets cryptographic or multi-precision-integer kernels.** The one crypto-kernel
code-generation paper (Zhang & Franchetti) is a classical rewrite system, not LLM/RL. So applying
these optimizers to our ed25519 kernels is an OPEN opportunity, not prior art — the benchmarks,
reward signals and training data are all ML-operator-shaped and would need re-targeting.

### CUDA-L1 (arXiv:2507.14111; ICLR 2026). Li, Sun, Wang, Li, Shum.
- Contrastive RL for CUDA optimization. KernelBench 250 kernels: A100 avg 3.12x / median 1.42x;
  H100 avg 3.85x / median 1.32x. Pipeline: sample -> validate -> retrain on successes ->
  contrastive RL over scored prior implementations. ML operators. No crypto follow-up found.

### KernelBlaster — "Continual Cross-Task CUDA Optimization via Memory-Augmented In-Context RL"
- NVIDIA + UC Berkeley (Chris Dong et al.). Persistent CUDA Knowledge Base; profile-guided,
  textual-gradient agentic flow. KernelBench L1/L2/L3 geomean 1.43x / 2.50x / 1.50x over PyTorch.
  Open-source harness. ML operators.

### Kernel-Smith (HF 2603.28342, 30 Mar 2026). He Du et al.
- Evolutionary agent (population + archive + execution feedback) + evolution-oriented post-training
  (step-centric RL). Kernel-Smith-235B-RL = SOTA on KernelBench Triton, beating Gemini-3.0-pro and
  Claude-4.6-opus. IMPORTANTLY also validated on **MetaX MACA (non-NVIDIA) backend** -> cross-platform
  adaptation is feasible, mildly encouraging for a future AMD/HIP target.

### SOL-ExecBench (arXiv:2603.19173, 19 Mar 2026). Lin, Modi, Hari (NVIDIA) et al.
- 235 CUDA kernel problems from 124 models, NVIDIA Blackwell, BF16/FP8/NVFP4. Scores kernels against
  analytically derived **Speed-of-Light** hardware bounds (SOLAR pipeline), not software baselines.
  Sandboxed harness with clock locking, L2 clearing, anti-reward-hacking static checks. Benchmark
  philosophy ("close the gap to hardware SOL") worth borrowing for our own kernel evals.

### KernelBench-X (HF 2605.04956, May 2026). Han Wang et al.
- 176 tasks / 15 categories; evaluates correctness + hardware efficiency of LLM Triton kernels.
  Finding: "task structure determines correctness more than method." ML operators.

### KernelBench (arXiv:2502.10517, ICLR 2025). Ouyang et al. (Stanford).
- The canonical benchmark: 250 PyTorch ML workloads. Frontier reasoning models match the PyTorch
  baseline in <20% of cases out of the box. All downstream optimizers above are scored on it.

### AVO — "Agentic Variation Operators for Autonomous Evolutionary Search" (HF 2603.24517, Mar 2026).
- Terry Chen, ..., Andrew Kerr, Haicheng Wu, Tianqi Chen (CUTLASS/TVM lineage). Replaces fixed
  mutation/crossover with autonomous coding-agent variation operators over a lineage + domain KB +
  execution feedback. General evolutionary search incl. GPU code. ML/GPU, not crypto.

### Zhang & Franchetti — "Code Generation for Cryptographic Kernels using Multi-word Modular
Arithmetic on GPU" (arXiv:2501.07535; CGO 2025). (SPIRAL/CMU.)
- The ONLY crypto-specific GPU code-gen here. Formalizes **MoMA** (multi-word modular arithmetic):
  a rewrite system that recursively lowers large-bit-width integer arithmetic to machine-word ops,
  compatible with compilers/code generators. Evaluated on crypto kernels incl. BLAS and NTT across
  GPUs. INTEGER-based (machine words), not FP, not LLM-driven. Relevant as the principled way to
  auto-generate/verify multi-precision field kernels (correct-by-construction).

### ZKProphet — "Understanding Performance of Zero-Knowledge Proofs on GPUs"
(arXiv:2509.22684, 17 Sep 2025). Verma, Yuan, Talati, Austin (Univ. of Michigan).
- Not an optimizer — a performance DIAGNOSIS, and very useful to us:
  - **Field mul+sqr = 93.8% of NTT time and 80.0% of MSM time** -> field multiply is THE kernel to win.
  - A single field multiply ~**2,656 GPU cycles vs 402 on CPU**; part of the gap is **warp divergence
    from conditional (modular) reductions** serializing the warp -> argues strongly for BRANCHLESS /
    constant-time reduction (both Montgomery-integer and FP64 can be made branchless).
  - ZKP arithmetic "executes exclusively on the GPU's 32-bit integer pipeline" with limited ILP due
    to data dependencies, so throughput is bounded by integer compute units -> a direct argument for
    trying the FP64 path to escape the integer-pipeline bottleneck (esp. on CDNA, Topic 3).
  - Once MSM is optimized, NTT (hence field arithmetic) dominates up to 90% of latency.
- NVIDIA-focused; numbers may not transfer to CDNA, but the qualitative lessons do.

### InCoder-32B / InCoder-32B-Thinking (arXiv:2603.16790 / 2604.03144, Mar/Apr 2026)
- 32B code foundation models for "industrial" code incl. GPU kernel optimization + chip design,
  trained with error-driven chain-of-thought reasoning traces. Tangential; not crypto-specific.

================================================================================
TOPIC 5 — Batch modular inversion beyond the classic Montgomery trick (1987)
================================================================================

### Baseline — Montgomery's trick (1987)
- n independent inversions -> 1 shared inversion + ~3(n-1) muls. gECC's phrasing: "compress n modinv
  into one modinv by introducing 3n modmul." This is already near-optimal for THROUGHPUT (minimizes
  total multiplications). Our vanity loop is throughput-bound with huge independent parallelism, so
  the classic serial trick is the baseline to beat on total-modmul-count, not latency.

### Visegrády — "Low latency data-flow graphs for simultaneous modular inversion of many inputs"
(arXiv:2607.11337, 13-14 Jul 2026). Tamás Visegrády, Ripple Labs Research. CC BY-SA.
- [VERIFIED from PDF text] Rebuilds Montgomery's trick as a maximally parallel DFG: schedules the
  auxiliary complement-product multiplications into idle multiplier slots during the product/
  inversion accumulation, so the post-inversion phase adds **exactly one multiply layer regardless
  of N**. Critical path = ceil(log2 N) multiply layers + 1 inversion + 1 final parallel multiply.
  Measured on AMD Ryzen AI PRO 350 w/ libsecp256k1: **1.27x single-inversion latency for N=8,
  1.34x for N=16**; peak multiplier load (>N/2 simultaneous muls) lasts only ~80 ns of a ~1500-1600 ns
  total. Target: latency-sensitive batch (MPC presignatures, batch-ECDSA nonce inversion);
  constant-time capable. Refs [Tom26]=Tomescu "batch-normalize multimult" (May 2026),
  [BCH+26]=Bernstein et al. (below), [Arn24]=Arnold.
- **FLAG — likely NOT a win for us.** It is a LATENCY optimization that trades MORE simultaneous
  multipliers for shorter critical path; it does NOT reduce total multiplication count. On a GPU
  vanity grinder we already saturate throughput with thousands of independent batches, so we want
  minimum total modmuls per inversion (classic serial trick), not minimum latency of one batch.
  Adopt only if a specific latency-bound sub-step appears. Worth knowing; not our default.

### Kim, Cho & Park — "Accelerating NTRU+ Key Generation via Hierarchical Batch Inversion"
(IACR ePrint 2026/1191).
- [SECONDARY] Hierarchical/recursive batching: split inputs into k groups to expose independent
  product chains, then batch the group-product inversions. Polynomial-inversion latency -48.91% (C)
  / -59.57% (AVX2). Post-quantum (NTRU+), not ECC, but the hierarchical-grouping idea could reduce
  dependency-chain length in a very large batch inversion.

### Bernstein, Chen, Harrison, Huang, Maxwell, Wang, Wuille & Yang — "Accelerating and verifying
constant-time modular inversion" (EUROCRYPT 2026, LNCS 16547, pp. 598-627).
- [SECONDARY] Single-input constant-time inversion; builds on safegcd (Bernstein-Yang 2019 TCHES).
  **Uses f = 2^255 - 19 as its case study** — literally our modulus. Speedup + formal verification
  of the constant-time inversion. Relevant to (a) the ONE shared inversion at the bottom of each
  Montgomery batch (make it fast and correct) and (b) constant-time/correctness assurance.
- (Context: safegcd = the modern fast constant-time gcd/modular-inversion algorithm; O'Connor &
  Poelstra 2025 formally verified libsecp256k1's safegcd in Coq.)

================================================================================
TOPIC 6 — Vanity address generation / Tor onion key grinding / ed25519 keypair search
================================================================================

### mkp224o — the standard CPU v3 .onion vanity generator
- GitHub cathugger/mkp224o. C99 + libsodium, CC0, multithreaded (-t). Cost grows steeply with
  prefix length. **Has an experimental BATCH mode (`-B` flag, `--enable-batchnum=N` at configure):
  ~15x faster than default** [SECONDARY — from a third-party mirror commit, not upstream]. This 15x
  is precisely the Montgomery-batch-inversion speedup: it confirms that **batched ed25519 keygen
  (repeatedly add the generator to walk public keys, batch-invert to affine, test the base32 prefix)
  + Montgomery batch inversion IS the proven core architecture** — exactly our hot loop.

### Onionloom — newest GPU v3 .onion generator (Aug 2026)
- GitHub chrisch88dev/onionloom; Tor forum launch 1 Aug 2026. Rust. **GPU via Vulkan / DX12 / Metal
  (auto-detected)** — i.e. **runs on AMD GPUs via Vulkan** (no CUDA/ROCm dependency). "Custom
  paired-coordinate CPU engine with simultaneous field division"; CPU-verifies every GPU result;
  auto-tunes to hardware. Claims **">2x mkp224o on my hardware for the same prefix length"**; "4
  chars = seconds, 8 chars = overnight." No absolute keys/s or named GPU published.
- RELEVANCE: our closest competitor/baseline AND a proof that a portable GPU path (Vulkan) already
  grinds v3 onions on AMD. A HIP/ROCm CDNA-tuned kernel should beat a generic Vulkan shader; this is
  the bar to clear and the number to measure against.

### ed25519 GPU keygen lineage (throughput anchors) [all SECONDARY, self-reported README/forum]
- nano-vanity / lisk-vanity: foundational OpenCL ed25519 keypair generation. GTX 1080: **709,077 keys/s**.
- Algorand GPU vanity (ed25519): GTX 950 ~320K keys/s; Tesla T4 ~7.25M keys/s; L4 ~9.38M keys/s
  (author notes these exclude batch-size optimizations).
- Caution: a Solana "1.2-1.3 billion/s per RTX 4090" tool is NOT comparable — it bypasses ed25519 via
  CreateAccountWithSeed. Do not use as an ed25519 baseline.
- CPU baseline: Cheng et al. AVX2 ~32,318 Curve25519 scalarmult/s @ Skylake 2 GHz (Topic 1).

================================================================================
CROSS-CUTTING CONCLUSIONS
================================================================================

INTEGER-vs-FP64 FIELD MULTIPLY (the fork):
- gECC: on NVIDIA A100, INTEGER radix-2^32 beats FP64 (227 vs 325 modmul instrs); they chose integer.
- DPF-ECC / Emmart-Zheng-Weems: the FP64 radix-2^52 approach, strong on older NVIDIA where IMAD was
  the bottleneck.
- ZKProphet: current GPU field arithmetic is pinned to the 32-bit INTEGER pipeline with low ILP and
  warp divergence from conditional reductions -> a standing argument to try FP64 + go branchless.
- AMD CDNA hardware: full-rate vector FP64 (MI300X 81.7 TFLOPS) and 163 TFLOPS *matrix* FP64, while
  32-bit integer multiply rate is unclear/not-full-rate. This INVERTS gECC's NVIDIA conclusion:
  on CDNA the FP64 radix-2^51 branch is a-priori the stronger bet, and is under-explored.
- Berthomieu et al. (2026) gives the rigorous FP64 multiword error analysis + a matrix-unit angle.
- ACTION: this cannot be settled from the literature. Build BOTH field-mul kernels (integer
  radix-2^32 vs FP64 radix-2^51/2^52 with 2 DFMA/limb-product) and microbenchmark on a real MI-series
  GPU. Design FP64 with the Berthomieu error bounds; keep reductions branchless (ZKProphet lesson).
  Also prototype the "batch of field muls as small FP64 matrix products -> MFMA 163 TFLOPS" idea.

BATCHING SCHEME:
- Adopt gECC's affine + Montgomery's-trick + Gather-Apply-Scatter inversion (reduce inversions to
  ~#SMs), batch sizes O(2^14)/SP, N well above the N>20 affine-beats-Jacobian threshold.
- mkp224o's `-B` batch mode (~15x) validates the whole approach for exactly the onion use case.
- Do NOT adopt Visegrády's low-latency inversion DFG as default: it cuts latency, not total modmuls,
  and we are throughput-bound.
- Make the single shared inversion fast + constant-time via Bernstein et al. 2026 / safegcd.

BIGGEST GAPS / UNVERIFIED:
- No published ECC/bignum numbers on ANY AMD MI-series GPU — we are first; must self-benchmark.
- DPF-ECC absolute ECC throughput (TIFS 2021) not pulled this pass (paywalled).
- CDNA 32-bit-integer-multiply throughput ratio not documented — microbenchmark needed.
- MI250X/MI355X FP64 TFLOPS from Wikipedia, not AMD spec sheets — confirm before quoting.
- No LLM/RL kernel optimizer has been applied to crypto/MPI kernels — open opportunity, not prior art.
