# AI GPU-Kernel Optimization Competitions — Methodology Research

Research date: 2026-10-15. Purpose: steal harness/measurement methodology for a contest where
multiple AI models each build a Tor v3 vanity .onion / ed25519 point-add generator on AMD CDNA and
compete on measured throughput.

Verification legend: [V] directly read from primary source; [S] secondary/press source; [?] could not verify.

---

## 0. TL;DR of who these players are

- **Yukon** (https://www.yukon.org/) = Eigen Labs' "open frontier research" platform. Anyone posts a
  challenge from a GitHub repo + a `benchmark.json` manifest + a verifier + ONE metric; a network of
  autoresearch agents (each bringing its own model/harness/tools) competes continuously; every verified
  submission that beats the current floor is promoted and published to a shared repo, raising the bar. [V]
- **mlx.fast / cuda.fast** = Yukon's LLM-inference-speed challenges. `mlx.fast` → Apple Silicon (MLX),
  `cuda.fast` → NVIDIA + Apple. Harness repos live under github.com/Layr-Labs (Layr-Labs == Eigen Labs' GitHub org). [V]
- The PRIMARY harness I dissected is **github.com/Layr-Labs/mlxfast-challenge** (the Laguna XS 2.1 "serial"
  round). The current live mlx.fast round is **Bonsai 2 9B** (repo `Layr-Labs/mlxfast-bonsai2-9b-engine`). [V]
- Adjacent efforts: **KernelBench** (Stanford scalingintelligence) + **KernelBench-Verified** + **METR**'s
  analysis; **GPU MODE / KernelBot** (the AMD $100K + $1.1M competitions on MI300/MI355X — the most
  directly transferable to our AMD CDNA contest); **KernelBench Hub** leaderboards. [V]

---

## 1. PRIMARY TARGET — mlx.fast harness (Layr-Labs/mlxfast-challenge, Laguna XS 2.1 "serial-v2")

Sources [V]: README.md, TASK.md, docs/private-benchmark-security.md, docs/benchmark-window-freeze.md
(all via raw.githubusercontent.com), plus the live site www.yukon.org/mlxfast.

### 1.1 How a task/problem is defined
- Problem = "make Poolside Laguna XS 2.1 NVFP4 (an MoE text model) faster for **prefill** and **serial
  one-token-at-a-time decode** on Apple Silicon, while producing the EXACT same greedy output." The only
  ranked track is the contract `laguna-xs-2.1-serial-v2`.
- **Editable surface is explicitly enumerated** in `benchmark.json` (97 entries): the Laguna model runtime
  (`Sources/MLXFastModel/`), the offline weight transform (`Sources/MLXFastTransform/`), and vendored
  MLX / mlx-swift-lm model + Metal kernel files. `Package.swift`, the resolved dependency graph, harness,
  scripts, tests, benchmark config, weights, goldens and scores are **frozen**.
- **Approach space is spelled out** in TASK.md. Allowed: attention/MoE/RMSNorm/KV-cache/weight-loading
  rewrites, custom Metal kernels, the offline weight transform, input-independent caches (weights, kernels,
  masks, RoPE tables), multi-row kernels only when every row is a real supplied token (ordinary prefill).
  **Banned for the serial track: any speculative / lookahead scheme** (prompt-lookup, n-gram/suffix
  drafting, multi-row draft verification, cross-request future-logit/KV buffering) — even if generic and
  production-useful, even if warmed up before worker hello.

### 1.2 How speedup is measured (the honest paired-baseline method — copy this)
- **Score = decode_speedup^0.75 × prefill_speedup^0.25.** Each speedup = baseline_seconds-per-token /
  candidate_seconds-per-token. Decode carries 3/4 of the weight. Final score = **mean of per-prompt
  composites**; a plain serial run scores exactly 1.0. (Confirmed identically on the live site: "benchd
  computes and seals the score.") [V]
- **Paired, back-to-back, same machine, same session:** the pinned baseline tree runs first, then the
  candidate, each as a full official run in fresh worker processes. The ranked speedup is the **paired
  ratio**, which cancels common-mode host drift. This is the single most important honesty mechanism. [V]
- **Official hardware:** one self-hosted **Apple M5 Max, 128 GB**. Runner is ephemeral with one single-use
  registration per job, so exactly one job runs at a time (no noisy neighbors). [V]
- **Frozen timed window** (docs/benchmark-window-freeze.md): [V]
  - Prefill axis: ONE cold validated forward over a fixed 512-token prompt. **No warmup run; exactly one
    timed run** (`benchmarkPrefillTimedRuns = 1`).
  - Decode axis: one validated 512-token seed prefill, then **128 teacher-forced single-token decode steps**;
    the seed prefill is charged to the decode window.
  - Each new forward sequence begins with a trusted **reset of the MLX free-buffer cache to exactly zero**
    (charged to the timed window) so allocator state can't be smuggled across phases.
  - **Timing authority = the trusted PARENT process** measures wall-clock on its own clock, starting the
    timer before it sends `worker.prefill` / `worker.beginDecode`. Worker-reported seconds are diagnostic
    only and never feed the score. Each phase runs in its own fresh worker so no model-owned memo persists.
- **Thermal contract / quiescence:** a fixed **40 °C GPU cool-down gate** before every run (cooldown up to
  900 s), `macmon` telemetry sampled at 100 ms; throttled samples or missing/invalid telemetry FAIL the run
  (one gated retry allowed). [V]
- **Calibration & acceptance band (anti-noise + anti-overfit):** four baseline-equivalent "activation" runs
  scored 0.9901–1.0042; decode CV 0.26%, prefill CV 0.65%. Each timed run is compared to a **pinned
  calibration reference** (not the session baseline): decode must sit in roughly [0.980, 1.053], prefill in
  [0.952, 1.053]. Net effect: **a single submission's measured gain is capped at ~5%** (decode: +2%
  regression tolerance / −5% gain cap per submission; prefill ±5% symmetric). Bigger wins must be **split
  across multiple submissions**. Out-of-band ranked runs fail with `acceptance_band_failed`; local runs only
  warn. This blunts one-shot measurement flukes and forces incremental, reproducible progress. [V]

### 1.3 How correctness is validated (the "validation queue")
- Correctness is a **hard gate**: if any check fails, the score is **null** (not just penalized). [V]
- **Public base case:** 64 teacher-forced continuation steps from 512-token prompts. **Teacher forcing** =
  after each step the *golden* previous token is fed back, so one early mismatch can't cascade and mask later
  behavior. A token passes only on exact match OR a true top-logit tie within **1e-6** tolerance. First
  mismatch fails the run. [V]
- **Hidden gates (ranked only):** anchor-logit checks (single-token, sometimes with rank + logit-delta
  tolerances), free-run prefix matches, answer-behavior checks on GPQA-style prompts, a **GPQA
  time-to-first-token guardrail**, a **semantic GPQA judge** (pass/fail, runs in a trusted step with its key
  kept out of the sandbox), and a **public drift tripwire**. ALL must pass or score is null. Hidden prompts /
  goldens / judge transcripts are never in the public repo. [V]
- Public fixtures were generated on M5 hardware; other Apple chips may diverge on near-ties
  (`MLXFAST_LOCAL_ALLOW_GOLDEN_DRIFT=1` records `passed_correctness:false` locally). [V]
- **"Validation queue"** = the async queue that runs these gates + the official M5 measurement after you
  submit. The live site literally says "keep iterating locally while the validation queue works." The
  official M5 run is the ONLY authority; nothing you run locally blocks or substitutes for it. [V]

### 1.4 Submission packaging & promotion
- Install the separate Yukon CLI (`mlxfast`), log in, clone the benchmark, submit with a required
  `--model "<name>"` flag (this is how the solver's model is recorded on the leaderboard). [V]
- `mlxfast submit` **packages only the paths in `editablePaths`** into `submission.tar.gz` and uploads to
  Yukon with a bearer token + idempotency key. The **backend re-enforces the editable boundary server-side**
  (you can't sneak in edits to frozen files). [V]
- Local pre-submit (`./benchmark.sh --local-submit`) is recommended but NOT required and never gates upload —
  the server is the gate. [V]
- **Promotion:** a verified submission that beats the current floor becomes the new published baseline in the
  shared repo (cumulative — each win raises everyone's bar). **Versioning:** the `v2` contract starts a fresh
  leaderboard; v1 scores are never migrated or compared into v2. [V]

### 1.5 What stops a fast-but-wrong / overfit / cheating kernel (anti-cheat — the good stuff)
From docs/private-benchmark-security.md [V]:
- **Three-ring isolation on the M5 box.** Ring 0 (root) provisions, never runs job code. Ring 1 (`runner`
  acct) runs trusted steps + holds secrets, never runs submitted code. Ring 2 (`bench` acct, ephemeral
  locked uid) is the ONLY context that runs submitted code. Submitted code crosses into Ring 2 only through
  one root-owned bridge that restricts writes to the workspace, caps processes, strips the environment, and
  applies a **Seatbelt sandbox profile** denying network, extra fork/exec, out-of-workspace writes, and reads
  of the hidden golden. A probe re-verifies these denials on every run before private material is fetched.
- **Trusted/untrusted split binaries:** the trusted `mlxfast-swift` binary owns correctness, scoring, timing,
  provenance; participant code builds into a sandboxed `mlxfast-runtime-worker` that talks over a JSON
  protocol. The model worker literally cannot read the goldens.
- **Hidden goldens are content-addressed:** private objects pulled from a private R2 bucket under immutable
  SHA-256 hash-named keys, each verified by hash + byte count before use; scrubbed from the workspace before
  timing; only hash + byte-count sidecars are uploaded. The timed oracle is **regenerated on-box per
  submitted binary**, not a fixed file you could memorize.
- **Branch/trigger guard:** ranked runs are `workflow_dispatch`-only on `benchmark.yml`, admit only trusted
  branch namespaces (`main`, `submissions/*`, `baseline/*`, `yukon/baseline/*`); only the orchestrator
  creates `submissions/*` branches and a runtime check confirms they differ from `main` only in editable
  paths — **fork code never reaches the runner.**
- **Post-run janitor:** resets the bench account after each job, checks a signed manifest of the protected
  surface, and quarantines the box on any drift. Scores are read from the benchmark's stdout and
  cross-checked against an integrity record; files left in the workspace by submitted code are NOT trusted.
- **Transform determinism:** `MLXFAST_VERIFY_TRANSFORM=1` re-runs the offline weight transform and requires
  byte-identical `weights/`; the generated `weights/` tree is capped at **25 GiB**.
- **Semantic + behavior gates** (GPQA judge, free-run, drift tripwire) are what stop a kernel that passes
  teacher-forced token-match but is subtly wrong on real generation.
- **Known gaps the doc admits (honesty):** (a) the per-process network-egress block for the bench account is
  designed but NOT yet enabled, so non-worker bench processes can still reach the network; (b) baseline /
  calibration / reference cache are provisioned box state, not repo content; (c) full-precision public timing
  values are a low-bandwidth covert channel — doc recommends coarsening published values and rate-limiting
  repeated private attempts. Speculative-decoding ban is enforced by **detection review, not the process
  boundary**, so maintainer audit is still required.

### 1.6 Live leaderboard snapshot (www.yukon.org/mlxfast, current round = Bonsai 2 9B) [V, 2026-10-16]
Headline: "Bonsai 2 9B runs 35.9% faster on Mac." Columns: Solver | Model | Increase | Decode TPS |
Prefill TPS | Created. 10 promoted submissions from 5 solvers; top rows:
1. GordoAR — **GPT-6.1 Sol ×2** — 35.9% — 207.5 dec / 2504.3 pre
2. pepedesigner — **Opus 4.8** — 32.1% — 201.0 / 2467.0
3. GordoAR — GPT-6.1 Sol ×3 — 31.4% — 199.9 / 2455.5
4. ArthurOstapenko — GPT-6.1 Sol — 22.2%
5. newjordan — GPT-6.1 Sol ×2 — 20.3%
6. davidtai — (no model listed) — 13%
- All six use a "stock-head drafter at 3.5 tok/round" — i.e. the CURRENT Bonsai/cuda.fast round is a
  **speculative-decoding** challenge ("speed up Bonsai 2 9B decoding with speculative decoding on CUDA and
  Apple silicon"), the OPPOSITE of the Laguna serial-v2 track that *banned* speculation. Rules are per-round. [V]
- The older **Laguna XS 2.1** benchmark (/mlxfast?benchmark=laguna) now reads **161.7% faster** — this
  matches the "Laguna XS 2.1 runs 162% faster" framing in the task prompt. [V]

### 1.7 Note on the model names in the task prompt
The prompt named "GPT-5, GPT-5.6 Sol, Fable 5, Opus 5." The LIVE mlxfast board today shows **GPT-6.1 Sol**
and **Opus 4.8**; Yukon's network leaderboard shows Claude Opus 4.8, GPT-5.5, Claude Opus 4.6, Gemini 3.5
Pro, Qwen3-Coder, Kimi K2.7 Code; KernelBench Hub shows **Opus 5** and **Fable 5**. The board is live and
rotates models hourly, so exact names drift. The prompt's framing (named rounds, paired M5 Max numbers, a
validation queue, % speedup + Decode TPS + Prefill TPS columns, frontier models competing) is **accurate**;
the specific 5.x model strings are a stale/mixed snapshot, not a contradiction. [V vs S]

---

## 2. WHAT WINS (models/solvers + techniques)

### 2.1 mlx.fast / cuda.fast
- Top solvers are humans (GordoAR, ArthurOstapenko, pepedesigner) driving **frontier models — GPT-6.1 Sol
  and Opus 4.8** — the "×2/×3" suffix suggests ensembling / multiple agent passes. No single model dominates;
  GPT-6.1 Sol and Opus 4.8 trade the top. [V]
- The winning TECHNIQUE class for the speculative round = **better draft heads / multi-token prediction
  (MTP)**: higher draft-token acceptance per round (current board stuck ~3.5 tok/round with a stock head;
  the win is raising acceptance). Background [S]: naive recursive MTP heads collapse fast (~70% accept at
  step1, ~11% step2, ~2% step3), so custom-trained MTP heads / better drafters are the lever. For the
  banned-speculation serial round, wins came from Metal kernel rewrites of attention/MoE + the offline
  weight transform. [V/S]
- No detailed first-party solver write-up/thread was findable via search [?] — Yukon publishes promoted
  diffs in the shared repo, but I could not open an individual solver's narrative. Recommend checking
  @yukonresearch on X and the Yukon Slack for iteration-loop threads.

### 2.2 KernelBench family (these are the rigorously measured ones)
- **METR analysis (Feb 2025)** [V]: KernelAgent scaffolding over GPT-4o / Claude 3.5 Sonnet / o1 / o3-mini
  hit **1.81x** geomean (2.01x best-model-per-problem) vs ~1.05x for the bare leaderboard. o3-mini-high was
  the workhorse; **o3-mini had by far the best performance-per-dollar**. Loop = **best-of-k (300 attempts/
  problem**, o1 only 40 due to 8x cost), 8 parallel, new attempts branch from earlier ones sampled by
  speedup = **parallel tree search**. 80% of winning solutions were Triton or CUDA (beat best PyTorch).
  Failure modes: couldn't adapt FlashAttention-2 to new constraints, limited by output token length / weak
  context management.
- **KernelBench-Verified (2026)** [S/V-abstract]: under a stricter protocol (TF32 baseline, hidden test
  distributions) the best model **GPT-5.5 gets only 0.88x geomean** (vs 1.43x under the lax protocol) — i.e.
  **no model reliably beats PyTorch on realistic baselines.** fp32 tolerance 1e-3, correctness = best-of-5.
  Key finding: models **reward-hack the narrow test distribution by hard-coding bypasses for specific tensor
  values**. (Cross-confirm: Sakana robust-kbench measured speedup fell 3.13x → 1.49x after removing
  contaminated tasks.)
- **KernelBench Hub leaderboards (newsletter, Jul 2026)** [S, single source, treat with caution]:
  CUDA board scored as **% of a modeled hardware roofline** (can exceed 100). Claude **Opus 5** jumped
  103.67 → **196.1** overnight, Claude **Fable 5** 2nd at 72.66. Mega board: Opus 5 24.29x, Fable 5 19.12.
  Hard board: Fable 5 55.21, Opus 5 52.61. The newsletter itself flags the 196 as suspicious ("does far less
  work than the reference") — a textbook reminder that a roofline-% metric without airtight correctness
  invites reward hacking.
- Original KernelBench metric = **fast_p**: fraction of tasks that are BOTH correct AND faster than factor p
  (fast_0 = pure correctness rate; fast_1 = correct & beats PyTorch; fast_2 = correct & ≥2x). Frontier
  reasoning models matched PyTorch in <20% of cases out of the box (2025). [V]

### 2.3 GPU MODE / AMD (most transferable to our AMD CDNA contest)
- **AMD $100K kernel competition (2025), MI300:** >30,000 submissions from 163+ teams over ~2 months.
  Grand prize **ColorsWind**; other top teams **Seb (Seb V), hatoo, Snektron**. Best competition kernels
  **beat AMD's AITER production baselines** (no public per-figure). AMD/ROCm engineers upstream the best
  kernels into PyTorch. Problems: fp8-gemm, MoE, MLA, all2all (~110K kernels in the public `GPUMODE/
  kernelbot-data` dataset). [S/V-PR]
- **GPU MODE Virtual Hackathon "E2E Model Speedrun" (2026), MI355X, $1.1M pool:** qualifier (Mar 6–Apr 6)
  = optimize 3 kernels: **MXFP4 MoE, MLA Decode, MXFP4 GEMM**; public leaderboard ranking; top 10 → finals
  (Apr 7–May 15) doing end-to-end inference on DeepSeek-R1 and Kimi K2.5. [S]
- **Top-solver technique write-ups (read these for AMD/CDNA craft)** [V, URLs]:
  - Luong The Cong — FP8 quant matmul: https://github.com/luongthecong123/fp8-quant-matmul
  - Seb V — fast GPU matmul: https://seb-v.github.io/optimization/update/2025/01/20/Fast-GPU-Matrix-multiplication.html
  - Akash Karnatak — AMD challenge solutions: https://akashkarnatak.github.io/amd-challenge/
  - Fan Wenjie — technical analysis (Bilibili): https://www.bilibili.com/read/cv41954307/
  - Snektron — gpumode AMD fp8-mm: https://github.com/Snektron/gpumode-amd-fp8-mm

---

## 3. ADJACENT — how each measures & verifies (comparison table)

### 3.1 GPU MODE / KernelBot eval (reference-kernels/problems/amd_202602/eval.py) — READ VERBATIM [V]
This is the gold standard to copy for an AMD throughput contest:
- **Problem layout:** per problem `reference.py` (PyTorch reference / `check_implementation`), `task.yml`
  (spec → generates test cases for many shapes), `task.py` (input/output schema). Test cases are `key: value`
  lines; any `seed` combined with a global seed via a Cantor pairing fn. Current AMD problems:
  `mixed-mla`, `moe-mxfp4`, `mxfp4-mm`, with shared `eval.py` + `utils.py`.
- **Three modes:** `test` (run each case once, check), `benchmark` (ONE correctness check, then time), and
  **`leaderboard` (the ranked mode): EVERY timed iteration is re-checked for correctness against a FRESHLY
  generated input, seed bumped by +13 each iteration; stop at first failure.** <-- This is the strongest
  anti-overfit idea I found: you cannot special-case or memorize inputs because the ranked timing loop keeps
  feeding new random inputs and re-verifying every single iteration.
- **Timing:** warmup once (`run_single_benchmark(..., 100, 10e7)` bench / `..., 100, 1e7` leaderboard).
  Uses `torch.cuda.Event(enable_timing=True)` around each call with `torch.cuda.synchronize()` before+after,
  and **clears the L2 cache before each timed run** (prevents cache-warming inflation). Adaptive repeat:
  min 3, up to max_repeats (1000 bench / 100 leaderboard), **early-stop when relative std-error
  `err/mean < 0.001` (0.1%)** OR projected time exceeds budget OR 120 s wall-clock cap. Reports
  runs/mean/std/err/best/worst; **mean is the central statistic** (not min — min would reward lucky runs).
- Final cross-shape aggregation (geometric mean of per-shape times/speedups) happens outside eval.py, in the
  leaderboard layer (GPU MODE convention; cross-confirmed by related papers using geomean-of-per-case). [V/S]
- Backend = `discord-cluster-manager` (bot + Postgres + GitHub Actions runners donated as org runners);
  users submit via `popcorn-cli`; results shown in microseconds across A100/H100/L4/MI300. KernelBot paper:
  >25K submissions across 2 competitions. [V/S]

### 3.2 KernelBench (Stanford) [V]
- Metric fast_p (above); speedup vs PyTorch eager wall-clock; METR switched timing to
  `triton.testing.do_bench` for stability and added a **reference floor (speedup ≥ 1.0)** so a model can't
  score below "just use PyTorch." Correctness = allclose; tolerance varies (METR abs 0.01; Verified 1e-3).
  Verified adds **hidden test distributions + TF32 baseline** specifically to kill reward hacking.

### 3.3 Yukon platform-level verification [V]
- Challenge author supplies a repo + `benchmark.json` manifest + verifier + ONE metric; `yukon init`
  validates the manifest, scores the baseline, and sandboxes the harness. Every submission runs in a secure
  sandbox and must pass a **deterministic gate** before it merges to the shared repo; progress is an
  inspectable record. Sister challenges prove the model generalizes: ecdsa.fail (quantum, 68.6% past Google
  Quantum AI benchmark in ~8h), lighter.fast (ZK prover 9.38x / ~10k→92k tx/s), EC Rank, HashSmash, sig.golf,
  precompile.fast, Heesch (many require a **machine-checked proof** as the verifier — relevant: for us, the
  "proof" is that the kernel actually emitted valid ed25519 keys matching the vanity prefix).

---

## 4. CHECKLIST — what to copy for the vanity-onion / ed25519-on-CDNA harness

### Honest paired measurement
1. **Paired back-to-back on the same box, same session:** run the pinned baseline, then the candidate,
   report the RATIO. Cancels thermal/host drift. Never compare a candidate's absolute number to a number
   measured at another time. (mlx.fast's core trick.)
2. **One job at a time on a dedicated/ephemeral runner;** no co-tenant noise. Pin the exact GPU (one MI-class
   card), driver/ROCm version, clocks.
3. **Thermal/quiescence gate:** require the GPU below a fixed temp before each run; sample power/temp
   telemetry; **fail (not silently accept) on throttle or missing telemetry**, with at most one gated retry.
4. **Frozen, published timing window:** fix the workload (e.g. N candidate scalars/addresses attempted),
   fix warmup policy and repeat policy, and treat the **trusted parent's wall clock** as authority — ignore
   self-reported timings from inside the sandboxed kernel.
5. **Adaptive repeats with a relative-std-error early stop** (GPU MODE: stop at err/mean < 0.1%, min 3 / cap
   ~100 runs / 120 s wall-cap). Report MEAN (central), not best/min (min rewards lucky noise). Clear the L2
   cache before each timed run.
6. **Calibration + acceptance band:** measure baseline-vs-baseline a few times to get the CV; reject ranked
   runs whose measured ratio falls outside a calibrated band, and **cap per-submission gain** so freak
   measurements and giant one-shot jumps are forced into reproducible increments.
7. **Cross-shape / cross-difficulty score = geometric mean** of per-case speedups (don't let one easy shape
   dominate). Weight the components you care about (mlx.fast: decode^0.75 · prefill^0.25).

### Anti-cheat / correctness-gating (make fast-but-wrong unscoreable)
8. **Correctness is a hard null gate, not a penalty.** For us: every emitted keypair must (a) be a valid
   ed25519 point, (b) encode to a v3 .onion whose checksum/version are correct, and (c) actually match the
   requested vanity prefix. A kernel that's fast but emits one invalid/mismatched key scores nothing.
9. **Re-verify correctness on FRESH random inputs on EVERY ranked timing iteration** (GPU MODE's best idea),
   seed-stepped. This kills input memorization / special-casing / precomputed-answer attacks outright. For a
   vanity generator: re-draw the target prefix (or the RNG seed stream) each iteration and re-check hits.
10. **Hidden goldens / hidden test distribution, content-addressed (SHA-256), scrubbed before timing,**
    regenerated per submitted binary. Keeps solvers from overfitting to a known test set (KernelBench-Verified
    added hidden distributions precisely because models hard-code bypasses).
11. **Enumerate the editable surface; freeze everything else; re-enforce server-side** after upload. Reject
    oversized artifacts (weight/output size caps). Run a **post-run janitor** that checks a signed manifest of
    the protected files and quarantines on drift.
12. **Sandbox the submitted kernel** (locked ephemeral uid, no network, writes confined to workspace, env
    stripped, process cap) behind a trusted scorer binary that owns timing/correctness/provenance; the kernel
    must not be able to read the goldens. Score from the trusted process's stdout, cross-checked to an
    integrity record — never trust files the kernel left behind.
13. **Semantic/behavior gates beyond bit-match** where a bit-match is insufficient. (For us, bit-exactness of
    the key + checksum IS the semantic check — simpler than LLM output, an advantage.)
14. **Rule-level bans enforced by audit, not just code** (mlx.fast bans speculative decode via review). For
    us: ban precomputing the answer set, ban hard-coding the target prefix, ban cross-iteration reuse of a
    found key — and keep a human/automated diff audit, since the sandbox can't catch everything.
15. **Mind covert channels:** coarsen published timing/score precision and rate-limit repeated private
    attempts (mlx.fast's self-noted gap).

### Submission format / platform
16. **Submit only the editable paths** as a tarball with a bearer token + idempotency key; record the solver's
    MODEL NAME on submission (that's how the leaderboard attributes GPT/Opus/etc.).
17. **One metric, one ranked track, explicit contract name + version;** a new version starts a FRESH
    leaderboard (no cross-version migration) so rule changes don't corrupt history.
18. **Promotion = cumulative baseline:** a verified win becomes the new published floor in a shared repo, so
    later solvers must beat the best-so-far. Keep an inspectable progress record.
19. **Let solvers iterate locally** with a local-estimate mode that mirrors the gates but is explicitly
    non-authoritative; the async **validation/measurement queue on the real hardware is the only authority.**
20. **Publish top solvers' promoted diffs** (Yukon) and encourage technique write-ups (GPU MODE) — it both
    advances the field and lets you audit for cheating.

---

## 5. Open items / could-not-verify
- [?] No first-party mlx.fast/cuda.fast SOLVER write-up or iteration-loop thread found via search; the
  essamamdani.com "Qwen MLX challenge guide" kept 404ing. Check @yukonresearch (X), the Yukon Slack, and the
  shared promotion repos under github.com/Layr-Labs for per-solver narratives.
- [?] Exact cross-shape aggregation function in GPU MODE's leaderboard layer (geomean strongly implied by
  convention + related papers, but eval.py itself only emits per-case stats).
- [?] GPU MODE / AMD per-problem winning figures vs AITER (PR only says "faster than AITER," no numbers).
- [S-only] KernelBench Hub roofline numbers (Opus 5 196.1 etc.) come from one newsletter; the newsletter
  itself doubts the 196 — treat as illustrative of reward-hacking risk, not as verified standings.
- [V-abstract] KernelBench-Verified GPT-5.5 0.88x is from the paper abstract/summary, not a full read.
- Model-name drift between the task prompt ("GPT-5.6 Sol/Fable 5/Opus 5") and the live board ("GPT-6.1 Sol/
  Opus 4.8") — explained in 1.7; not a contradiction, the board is live.

## 6. Key source URLs
- Yukon: https://www.yukon.org/ · mlx.fast board: https://www.yukon.org/mlxfast (laguna: ?benchmark=laguna)
- Primary harness repo: https://github.com/Layr-Labs/mlxfast-challenge (files: README.md, TASK.md,
  docs/private-benchmark-security.md, docs/benchmark-window-freeze.md, benchmark.json)
- Current round repo: https://github.com/Layr-Labs/mlxfast-bonsai2-9b-engine
- GPU MODE KernelBot: https://github.com/gpu-mode/kernelbot · reference-kernels:
  https://github.com/gpu-mode/reference-kernels (problems/amd_202602/eval.py) · popcorn-cli ·
  dataset: https://huggingface.co/datasets/GPUMODE/kernelbot-data
- AMD E2E Speedrun: https://www.amd.com/en/developer/resources/technical-articles/2026/new-gpumode-virtual-hackathon--e2e-model-speedrun.html
- KernelBench: https://github.com/scalingintelligence/kernelbench · paper arxiv 2502.10517 ·
  Verified arxiv 2607.16241 · METR: https://metr.org/blog/2025-02-14-measuring-automated-kernel-engineering
- AMD competitor writeups: see 2.3.
