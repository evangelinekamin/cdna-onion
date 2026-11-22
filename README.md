# cdna-onion

High-throughput Tor v3 `.onion` vanity address generator in HIP for AMD Instinct (CDNA2/CDNA3/CDNA4) GPUs, paired with a proof-of-work conformance and benchmarking harness (`onionrig`).

Vanity prefixes are brute force: every extra character multiplies expected work by 32 ($2^5$). This repository contains the combined `fe29` + conjugate-pair factoring HIP kernel (`entries/combined`), the split-key search & proof-of-work verification harness (`src/onionrig`), shared correctness vectors (`vectors/`), and offline Ed25519 proof-of-possession signatures (`proofs/`).

## Measured performance

Single-GPU headline rates are proof-of-work verified through `cdna-onion bench` (every reported hit re-derived on the host with `libsodium`; 95% Student-$t$ CIs). The 8x MI355X figure is sustained progress-logged throughput during an 83.6-minute live search for `evangeline` that recovered three offline-verified 10-character Tor v3 keys. Scrubbed raw JSONL logs and redacted Round-1 agent transcripts are archived in the `v0.7.1` GitHub release.

| Build | GPU (Architecture) | Power Cap | Throughput | 95% CI / Spread |
|---|---|---:|---:|---|
| Baseline (`onion-gpu` HIP port) | 1x MI210 (`gfx90a`) | 300 W | 1.45 G/s | 1.43-1.46 G/s |
| `entries/combined` (`base`) | 1x MI210 (`gfx90a`) | 300 W | **17.294 G/s** | 17.265-17.322 G/s |
| `entries/combined` (`chain`) | 1x MI300X VF (`gfx942`) | 750 W | **54.60 G/s** | 54.27-54.92 G/s |
| `entries/combined` (`base`) | 1x MI350X (`gfx950`) | 1,000 W | **54.78 G/s** | 54.12-55.45 G/s |
| `entries/combined` (`base`) | **8x MI355X** (`gfx950`) | 1,400 W/GPU | **575.0 G/s** | 69.2-74.7 G/s/GPU |

## Requirements

- **GPU kernel:** AMD Instinct GPU (`gfx90a`, `gfx942`, or `gfx950`) with ROCm 7.x (`hipcc`).
- **Harness & split-key tools:** Python 3.11+ and [`uv`](https://docs.astral.sh/uv/) (installs `pynacl`).
- **Proof-of-possession verifier:** Any Python 3.8+ (zero external dependencies).

## Quick start: build and verify

```bash
# 1. Install the Python harness and check shared test vectors
uv sync
uv run cdna-onion vectors --check

# 2. Run CPU arithmetic and walk emulation tests (no GPU needed)
make -C entries/combined test

# 3. Build the HIP kernel for your CDNA GPU
#    MI210 (CDNA2):  GPU_ARCH=gfx90a
#    MI300X (CDNA3): GPU_ARCH=gfx942 DEFS="-DF29_CHAIN_ASM=1"
#    MI350X/MI355X:  GPU_ARCH=gfx950
#    (Run `make -C entries/combined clean` first when switching GPU_ARCH or DEFS.)
make -C entries/combined clean entry GPU_ARCH=gfx950

# 4. Run GPU conformance check and proof-of-work benchmark
uv run cdna-onion check entries/combined/entry
uv run cdna-onion bench entries/combined/entry
```

*(The CLI is installed as both `cdna-onion` and `onionrig`; both invoke the same entry point.)*

## Generating a real vanity address (split-key)

Secret scalars never touch the GPU host. Generate start keys offline on a trusted machine, send only the public points to the GPU node, and recombine matches locally:

```bash
# On your TRUSTED machine (never inside this repo tree, never on the GPU host):
uv run cdna-onion keygen ~/onion-keys --count 64

# Copy ONLY ~/onion-keys/start-points.json to the GPU host, then run:
uv run cdna-onion search entries/combined/entry \
    --points start-points.json --prefix myprefix --devices 0-7

# Back on your TRUSTED machine, reconstruct a hit from search/search.jsonl:
uv run cdna-onion combine --secrets ~/onion-keys/start-scalars.secret \
    --point N --k K --address ADDRESS --out ~/my_service_hs
```

`combine` enforces Ed25519 scalar clamping so Tor loads the exact key, refuses any address mismatch, and writes `hs_ed25519_secret_key`, `hs_ed25519_public_key`, and `hostname` with mode `0700`/`0600`. To install under `/var/lib/tor`, move the directory and set ownership to the `tor` user (`sudo chown -R debian-tor:debian-tor /var/lib/tor/my_service_hs`) without loosening permissions.

## Verify our 10-character `evangeline` proofs

To verify offline that we possess the Ed25519 secret keys for all three 10-character `.onion` addresses found during the 8x MI355X search (no dependencies required):

```bash
python3 proofs/verify.py
```

## Security model

- **Split-key isolation:** Start scalars and Tor secret keys never leave the trusted machine, are written owner-only (`0600`) with core dumps disabled, and `keygen` refuses to write inside this repository tree.
- **Confined benchmarking:** By default, `isolation.py` runs entries inside unprivileged Linux user, network, PID, and `/proc`-only mount namespaces (offline, scrubbed environment, empty `HOME`, `HIP_VISIBLE_DEVICES`, 2-core `taskset`, bounded output). Because mounting only `/proc` does not isolate the host filesystem, standalone local runs still see files accessible to the invoking user; untrusted arena submissions should additionally run inside an offline container and Slurm/cgroup allocation (as on ASU Sol's Apptainer `--containall` judge). On cloud containers that block nested user namespaces, pass `--no-isolation` explicitly for your own audited builds.
- **Proof-of-work scoring:** Every benchmark draws a fresh random start point `A0` and short targets; throughput is `verified_unique_hits / (p_match * wall_seconds)`. A single invalid offset disqualifies the entire benchmark.

## Repository layout

```text
entries/combined/     HIP kernel (fe29 9x29 limbs, pair factoring, mu-fold, CKPT2, leaf32)
entries/baseline/     HIPIFY'd onion-gpu calibration baseline (CUDA + HIP)
entries/python-ref/   Reference implementation of the entry contract (~170 lines of Python)
src/onionrig/         Confinement, proof-of-work benchmark rig, split-key & Tor key tooling
proofs/               Ed25519 proof-of-possession signatures + zero-dependency verifier
vectors/              Content-hashed field, point, walk, and onion test vectors
tools/microbench/     CDNA primitive throughput microbenchmarks (v_mad_u64_u32 vs v_fma_f64)
tools/rental/         Reproducible build, sweep, and multi-GPU search scripts
docs/design/          Protocol contract, architecture spec, CDNA perf reference, arena brief
docs/research/        Public literature, tool, and CDNA-craft notes
tests/                Unit, security, proof-of-possession, and Tor round-trip test suite
```

## License

MIT (see [LICENSE](LICENSE)).
