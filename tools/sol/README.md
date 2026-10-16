# Running on Sol (MI210) or any GPU node

One job builds everything for the node's GPU and records, in order:
1. an environment snapshot,
2. AMD resource usage,
3. primitive microbenchmarks,
4. a telemetry probe,
5. conformance of the baseline,
6. a measured benchmark of the baseline.

Results stream into one directory as they are produced.

**Keep it a benchmarking study.** Short runs measured in candidates/s answer the
research question. Long vanity searches for a personal address should not run on
Sol (see the acceptable-use notes in `docs/design/ARCHITECTURE.md` sec 1). This
job takes about 5 minutes of GPU time (4 on the MI210 in October 2026).

## One-time setup on a Sol login node

    git clone https://github.com/evangelinekamin/onionrig.git ~/onionrig && cd ~/onionrig
    tools/sol/setup.sh

`setup.sh` puts a pinned, checksum-verified uv, a uv-managed Python, and the
locked environment inside the repo (`.uv/`, `.venv/`), then checks the vectors.
No shell startup file changes, so `rm -rf ~/onionrig` removes every trace. The
job picks up the repo-local uv by itself; for commands by hand, first run
`. tools/sol/env.sh`. Cluster home and scratch storage are not archives: copy
`results/` somewhere permanent after each run.

The repo is private, so `git clone` asks for a username and password: give your
GitHub username and a fine-grained personal access token limited to this one
repository with read-only Contents access (GitHub > Settings > Developer settings).

## Run

1. Run `sbatch tools/sol/mi210-bench.sbatch`. It asks for one of the two MI210s in
   node `sg250`: partition `fpga`, QoS `public`, GPU type `mi200`. That partition
   gives a node to one user at a time, so nothing else shares it during the run.
2. Follow progress with `tail -f results/mi210-<jobid>/progress.log`.

If the job ends at once, its Slurm log `onionrig-<jobid>.log` says why. sg250
has ROCm installed system-wide (no module needed). On a node without `hipcc`,
the job loads a `rocm` module; if that fails, the log lists the ROCm modules on
offer, and you resubmit with `sbatch --export=ALL,ROCM_MODULE=<name> ...`.

On a node you already hold, interactively or on the MI355X box, skip Slurm:

    module load rocm            # or however ROCm is provided there
    tools/sol/run-all.sh results/my-run 60

The script detects the architecture (`rocminfo`) and builds for exactly that
target. gfx950 (MI355X) needs ROCm 7 or newer.

Entries run in unprivileged namespaces by default, so they have no network and
leave no stray processes. If the node disables unprivileged user namespaces,
`onionrig` refuses to run entries there. This script runs only the baseline, our
own code, so it then carries on without namespaces, logs that in `progress.log`,
and the results record it. Set `ONIONRIG_NO_ISOLATION=1` to skip them anyway.
Untrusted entries on such a node need another sandbox: a VM, or a dedicated
account with nothing else on it.

## What to bring back

The whole `results/<run>/` directory. The most important files:

| File | What it answers |
|---|---|
| `microbench.jsonl` | `v_mad_u64_u32` vs `v_fma_f64` ops per CU per clock: **the integer-vs-FP64 fork** |
| `resource-usage.txt` | VGPRs, scratch, and occupancy as compiled for this exact GPU |
| `check.txt` | conformance of the baseline on real CDNA |
| `bench-*.jsonl` | every measured run, with thermal samples; `onionrig board` ranks them |
| `telemetry-probe.txt` | real `amd-smi` output, to validate the telemetry parser on CDNA |
| `environment.txt` | node, driver, compiler, and which GPU Slurm handed the job |
| `microbench-telemetry.jsonl` | power, temperature, and clock sampled through the microbenchmarks |
| `seff.txt` | CPU and memory efficiency (`seff`), to keep the request right-sized. Its RUNNING warning is expected: the work ran as a finished job step, and the figures match `seff` run after the job |

The telemetry parser was written from `amd-smi`'s own source, and it has not
yet seen output from a live MI210. If the probe looks wrong, that file is what
is needed to fix it.
