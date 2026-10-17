# Rented GPUs: tuning on one, searching on many

Two kinds of session, each one command once the host is set up:

- **Tuning, one GPU** (an MI300X or MI355X, 1-3 hours): `tune.sh` measures the
  GPU and picks the build to use.
- **The search, every GPU** (the 8x MI355X node): `grind.sh` runs the real
  search with that build. It does no tuning: it checks each GPU's
  arithmetic (under a minute), then searches until stopped.

An MI300X runs the same hot loops as an MI355X (gfx942 and gfx950 compile to
identical instructions here), so choices about instructions (`chain`) carry
over from one to the other. Choices that trade memory traffic against
instructions (`ckpt1`, `h8k`) depend on the clock and power regime, which
differ: an MI355X settles those best.

## Before renting (the trusted machine)

1. **Start points.** Run `uv run onionrig keygen ~/onion-keys/NAME --count 64`.
   It writes two files. `start-points.json` is public and goes to the GPU host.
   `start-scalars.secret` never leaves this machine. keygen refuses to write
   inside the repository.
2. **Targets.** Use the base32 alphabet, a-z and 2-7. Up to four targets cost
   nothing extra in the GPU's filter, and each one multiplies the hit rate.
3. **The kit.** Commit, then run `tools/rental/kit.sh` (needs Docker, about 2
   minutes). It writes `dist/rental-kit-<commit>.tar.gz`, with prebuilt
   binaries for gfx90a, gfx942 and gfx950.
4. **SSH.** Register `~/.ssh/gpucloud_ed25519.pub` with the provider.

## Setting up the GPU host (both sessions)

    scp -i ~/.ssh/gpucloud_ed25519 dist/rental-kit-*.tar.gz \
        ~/onion-keys/NAME/start-points.json USER@HOST:
    ssh -i ~/.ssh/gpucloud_ed25519 USER@HOST
    tar -xzf rental-kit-*.tar.gz && cd kit
    tools/sol/setup.sh      # uv, Python and the rig, about a minute (needs the network)
    tmux new -s run         # survives a dropped connection (no tmux: use nohup ... &)

`rocminfo | grep gfx` should list the GPUs.

## Session A: tuning on one GPU

    tools/rental/tune.sh results/tune-mi300x

It runs these phases in order, and every result lands in the directory as it
is produced:

| Phase | Output | What it answers |
|---|---|---|
| snapshot | `environment.txt`, `rocminfo.txt`, `amd-smi-static.txt` | ROCm, driver, partition mode, power limits |
| build | `bin/`, `build.jsonl` | prebuilt binaries if they load here, else local builds |
| verify | `verify.jsonl`, `verify-*.txt` | each build's arithmetic on this GPU; a failing build is dropped |
| micro | `microbench.jsonl` | whether `v_mad_u64_u32` issues at full rate (ops per CU per clock) |
| profile | `profile.txt`, `profile/` | effective shader clock under the real kernel, cycles per phase |
| sweep | `sweep.txt`, `sweep/` | every build against the base, interleaved; picks one by a fixed rule |
| check | `check.txt` | the rig's conformance suite on the pick |
| bench | `bench.txt`, `bench/` | the rig's official proof-of-work rate for the pick |

The defaults take about an hour: a 6-round sweep of seven builds, 30-second
cells, and a 2-minute warm-up. `progress.log` says where it is.

- **Run one phase:** `tune.sh OUT sweep`.
- **Change the settings:** set environment variables. `ROUNDS`, `CELL_SECONDS`
  and `WARMUP` shape the sweep; `GPU` picks the GPU.

**The pick rule.** Each round runs every build once, in rotated order. A build
is compared with the base in the same round, so drift cancels. The sweep picks
the build with the best ratio to the base, as long as:
- its 95% interval lies entirely above 1;
- it never failed a cell;
- its verified hits always matched the work it claimed (proof of work).

`base2` is the base build run again. Its ratio shows the noise floor, and it is
never picked. The pick is written to `PICK`.

**Leftover time** can go to the real search on this GPU:

    BUILD=$(cat results/tune-mi300x/PICK) tools/rental/grind.sh \
        results/search-mi300x ../start-points.json PREFIX [PREFIX ...]

## Session B: the search on every GPU

    BUILD=PICKED tools/rental/grind.sh results/search ../start-points.json PREFIX [PREFIX ...]

PICKED is the name in session A's `PICK` file (`base` if there was no session A).
1. It verifies the build on every GPU in parallel and leaves out any GPU that fails.
2. Then `onionrig search` runs one entry per GPU, each pinned to CPUs of its own.
3. Each GPU reports its rate every few seconds in `search.log`, with that GPU's
   median wait for a hit. Divide that wait by the number of GPUs for the node's.
4. Verified hits go to `search/search.jsonl`, durable before the next batch.

**Stopping:** press Ctrl-C once to stop after in-flight work; twice aborts.

**Billing:** then pull the results and destroy the instance. On DigitalOcean a
powered-off droplet still bills.

**Restarts:** a restart never reuses a start point (`search/used-points.json`
tracks them).

## Pulling results home while it runs

    while sleep 300; do
        rsync -az -e "ssh -i ~/.ssh/gpucloud_ed25519" USER@HOST:kit/results/ ~/onionrig-results/HOST/
    done

## After a hit (the trusted machine)

Take `point`, `k` and `address` from the hit's line in `search.jsonl`:

    uv run onionrig combine --secrets ~/onion-keys/NAME/start-scalars.secret \
        --point POINT --k K --address ADDRESS --out ~/hs-ADDRESS

That directory is a Tor HiddenServiceDir. The command refuses a key that does
not reproduce the address.

## If something is off

| Symptom | What to do |
|---|---|
| `rocminfo` lists no GPU, or `/dev/kfd` is denied | Add the user to the `video` and `render` groups. In a container, pass `--device=/dev/kfd --device=/dev/dri --group-add video --group-add render` |
| "the prebuilt binaries do not cover or load" | The scripts build locally with the host's `hipcc` instead. gfx950 needs ROCm 7.0 or newer. Without that, run inside `rocm/dev-ubuntu-24.04:7.2.4` |
| "namespace isolation unavailable" | Nothing to do. These are our own builds, so the scripts pass `--no-isolation`, and the results record it |
| `rocminfo` shows 8 agents per GPU | The GPUs are in CPX partition mode: each agent is one XCD. `grind.sh` searches on every agent, which is fine. `tune.sh` tunes on agent 0 alone, so its ratios hold but its absolute rates are per XCD |
| "taskset is not installed" | Nothing to do. Entries run without CPU pinning |
| The provider sets `HIP_VISIBLE_DEVICES` | The scripts move it to `ROCR_VISIBLE_DEVICES`, which is how the rig narrows each entry |
| `telemetry-raw-g*.txt` instead of `telemetry-g*.jsonl` | The rig cannot parse this host's `amd-smi` output. The raw output is kept instead |
