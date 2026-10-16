# vanitydotonion

How fast can a correct Tor v3 vanity-address generator run on AMD Instinct
(CDNA) GPUs? This repo is the measuring stick. It contains:

- **`onionrig`**, a conformance and benchmark rig. It scores any generator by
  verified proof-of-work, so a result cannot be inflated by bugs or dishonesty.
- **A calibration baseline**: [onion-gpu](https://github.com/NIKIB0Y/onion-gpu)
  ported to CUDA and HIP (gfx90a, gfx950) under a common contract.
- **Split-key search tooling**: the GPU host never sees a secret, and found keys
  are recombined locally and checked against real Tor.
- **The design corpus** that competing models build against: `docs/`.

Vanity prefixes are pure brute force. Each extra character is 32 times more
work, and 13 characters is out of reach on any hardware. The research question
here is throughput, and in particular how the integer-vs-FP64 trade-off plays out
across CDNA generations, which nobody has published yet.

## Layout

    src/onionrig/        the rig (Python): reference math, vectors, protocol, scoring, CLI
    entries/python-ref/  reference entry: the contract in ~170 lines of Python (slow, exact)
    entries/baseline/    onion-gpu ported to the contract; CUDA + HIP from one source
    tools/microbench/    primitive throughput (v_mad_u64_u32 vs v_fma_f64, ...) per GPU
    tools/sol/           one-shot scripts for an MI210 node (Slurm) or any GPU box
    tools/fetch-amd-docs.sh  downloads the AMD ISA guides and whitepapers into docs/
    vectors/             shared correctness vectors (JSON + flat text), content-hashed
    docs/design/         ARCHITECTURE, CONTRACT, CDNA-PERF-REFERENCE, MEGAPROMPT
    docs/research/       literature, tool, competition, and CDNA-craft syntheses
    tests/               ~230 tests: real-Tor key round trip, real-GPU conformance, hostile entries

## Quick start

    uv sync                                    # Python 3.11+, installs pynacl
    uv run onionrig vectors --check            # confirm the shared vectors are intact
    make -C entries/baseline cuda              # or: make hip (needs ROCm)
    uv run onionrig check entries/baseline/build/grind-cuda
    uv run onionrig bench entries/baseline/build/grind-cuda
    uv run onionrig board                      # leaderboard of everything benchmarked

Compare two entries on the same card with paired, ABBA-ordered runs:

    uv run onionrig duel "ENTRY_A" "ENTRY_B" --pairs 4

On an MI210 (Sol), `sbatch tools/sol/mi210-bench.sbatch` runs microbenchmarks,
conformance, and a benchmark in one job (see `tools/sol/README.md`).

## How scoring works

The rig gives an entry a fresh random public point `A0` and a few short target
prefixes. The entry reports each offset `k` where `A0 + k*8B` matches a target,
and the rig verifies every one with libsodium. One bad offset voids the run, and
one invalid run fails the entry's whole benchmark.
Throughput is `verified hits / (match probability * wall time)`, timed by the rig
from `start` to `done`. Runs start from a cooled GPU, are sampled for thermal
state throughout, and are repeated until the mean is precise. Benchmarks use
real-search conditions (one to four targets, a low hit rate), so per-hit work
does not distort the number. See `docs/design/CONTRACT.md`.

## A real vanity address, safely

    # On a TRUSTED machine (never the GPU host, and never in this repo's tree):
    uv run onionrig keygen ~/onion-keys --count 64
    # Copy only ~/onion-keys/start-points.json to the GPU host. There:
    uv run onionrig search ENTRY --points start-points.json --prefix myname --devices 0-7
    # Back on the trusted machine, for a hit recorded in search/search.jsonl:
    uv run onionrig combine --secrets ~/onion-keys/start-scalars.secret \
        --point N --k K --address ADDRESS --out ~/my_service_hs

The search host only ever handles public points and offsets. `combine` rebuilds
the key from the secret scalar, refuses anything Tor would load as a different
key, and requires the hit's address so a typo or wrong point can never yield a
silently different key. The test suite round-trips a found key through a real
`tor` process. Tor needs the HiddenServiceDir to be owned by the user Tor runs
as, with mode 0700. To put it under `/var/lib/tor`, write it as your user
first and move it with `sudo mv` plus `sudo chown -R debian-tor:debian-tor`.
Do not loosen the permissions.

## Security model

- **Secrets never reach the GPU host.** The split-key search ships public points
  and returns public offsets. Start scalars and Tor keys exist only on the trusted
  machine, are written owner-only (0600) with no core dumps, and are never logged.
  `keygen` refuses to write inside this project tree, which is the tree you copy to
  GPU hosts.
- **Entries are untrusted code.** AI-written entries run confined by default:
  - unprivileged network, PID, and mount namespaces (no network, no surviving
    child processes),
  - a scrubbed environment with a private empty HOME,
  - one visible GPU and pinned CPUs,
  - bounded output.
  They still see the filesystem as you, so **never run entries on a machine or
  account that holds start scalars or Tor keys.**
- **On WSL**, Windows processes running as you can read the Linux filesystem
  regardless of Linux permissions. For a real key, the trusted machine should
  be a real Linux box, or at least one with nothing untrusted on the Windows side.

## Status

- Rig, vectors, contract, reference entry, split-key tools: complete and tested.
- Baseline: verified on an RTX 4060 Ti (about 954 M candidates/s, matching
  upstream's own figure). It compiles for gfx90a and gfx950, where the static
  analysis shows occupancy of 1 wave/SIMD, the first lever for AMD entries.
- Next: the first MI210 run (microbench + baseline), then the model competition
  (`docs/design/MEGAPROMPT.md`).
