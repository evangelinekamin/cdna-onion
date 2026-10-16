#!/usr/bin/env bash
# Build for this node's GPU, then run, in order: environment snapshot, AMD
# resource usage, primitive microbenchmarks, telemetry probe, conformance, and a
# measured benchmark of the baseline. Every result lands in OUT as it is
# produced, so a job that dies partway still leaves everything before that point.
#
# Usage: tools/sol/run-all.sh [OUT_DIR] [BENCH_SECONDS]
# Works on AMD (hipcc + rocminfo) or NVIDIA (nvcc) nodes; needs `uv` for the rig.
set -euo pipefail
umask 077  # results tie an onion address and benchmark history to this account

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
OUT=$(realpath -m "${1:-$ROOT/results/$(date +%Y%m%dT%H%M%S)}")
SECONDS_PER_RUN=${2:-60}
mkdir -p "$OUT"
log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "$OUT/progress.log"; }
# The repo-local uv from tools/sol/setup.sh, if that is how this checkout was set up.
# shellcheck source=env.sh
if [ -x "$ROOT/.uv/bin/uv" ]; then . "$ROOT/tools/sol/env.sh"; fi
rig() { uv run --project "$ROOT" onionrig "$@"; }
command -v uv >/dev/null || { echo "install uv first: https://docs.astral.sh/uv/" >&2; exit 1; }

if command -v hipcc >/dev/null && command -v rocminfo >/dev/null; then
    # Captured whole first: `rocminfo | grep -m1` dies of SIGPIPE under pipefail.
    AGENTS=$(rocminfo 2>&1) || { log "rocminfo failed: no usable AMD GPU here"; echo "$AGENTS" >&2; exit 1; }
    ARCH=$(grep -m1 -oE 'gfx[0-9a-f]+' <<< "$AGENTS") || { log "rocminfo lists no GPU agent"; exit 1; }
    BACKEND=hip
    MAKEVARS=(HIP_ARCHS="$ARCH")
else
    ARCH=cuda
    BACKEND=cuda
    MAKEVARS=()
fi
log "backend=$BACKEND arch=$ARCH out=$OUT"

{
    date -Is; hostname; uname -a
    env | grep -E '^(ROCR_VISIBLE|HIP_VISIBLE|CUDA_VISIBLE|GPU_DEVICE_ORDINAL|SLURM_[A-Z_]*GPU)' | sort
    command -v amd-smi >/dev/null && amd-smi static -g 0 2>&1
    command -v nvidia-smi >/dev/null && nvidia-smi -q 2>&1 | head -60
    command -v hipcc >/dev/null && hipcc --version 2>&1
} > "$OUT/environment.txt" 2>&1 || true

# Schedulers often set CUDA_VISIBLE_DEVICES and GPU_DEVICE_ORDINAL next to
# ROCR_VISIBLE_DEVICES. HIP applies them on top of ROCR's filtering, so on any GPU
# but the node's first they hide the allocated one. On AMD, ROCR's list is enough.
if [ "$BACKEND" = hip ] && [ -n "${ROCR_VISIBLE_DEVICES:-}" ]; then
    unset CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL
    log "GPU visibility: ROCR_VISIBLE_DEVICES=$ROCR_VISIBLE_DEVICES"
fi

log "building baseline and microbenchmarks"
make -C "$ROOT/entries/baseline" "$BACKEND" "${MAKEVARS[@]}" > "$OUT/build.log" 2>&1
make -C "$ROOT/tools/microbench" "$BACKEND" "${MAKEVARS[@]}" >> "$OUT/build.log" 2>&1
if [ "$BACKEND" = hip ]; then
    make -C "$ROOT/entries/baseline" resource-usage "${MAKEVARS[@]}" > "$OUT/resource-usage.txt" 2>&1
fi
ENTRY="$ROOT/entries/baseline/build/grind-$BACKEND"

log "microbenchmarks, with telemetry sampled alongside"
# Power and clock samples, lined up with each test by time (t_start/t_end), show
# whether a primitive runs below peak because the card clocks down. setsid makes
# the sampler a process group, so stopping it stops uv's child too.
setsid uv run --project "$ROOT" onionrig telemetry --follow "$OUT/microbench-telemetry.jsonl" \
    > "$OUT/microbench-telemetry.err" 2>&1 &
SAMPLER=$!
# 30 samples of ~150 ms per test: several telemetry readings land inside each one.
# A failure here should not cost the conformance and benchmark data that follow.
"$ROOT/tools/microbench/build/microbench-$BACKEND" 30 150 2> "$OUT/microbench.err" \
    | tee "$OUT/microbench.jsonl" || log "microbenchmarks failed; see microbench.err"
kill -TERM -- "-$SAMPLER" 2>/dev/null || true
wait "$SAMPLER" 2>/dev/null || true

log "telemetry probe"
rig telemetry > "$OUT/telemetry-probe.txt" 2>&1 || log "no telemetry backend (see telemetry-probe.txt)"

# Entries run in unprivileged namespaces where the node allows it. This script runs
# only the baseline, our own code, so it carries on without them where they are
# unavailable (results record it); `onionrig` itself refuses untrusted entries then.
ISO=()
if [ "${ONIONRIG_NO_ISOLATION:-}" = 1 ] || ! uv run --project "$ROOT" python -c \
        'import sys; from onionrig.rig.isolation import isolation_available as ok; sys.exit(not ok())'; then
    ISO=(--no-isolation)
    log "namespace isolation unavailable or disabled; running the baseline without it"
fi

log "conformance"
rig check "$ENTRY" --vectors "$ROOT/vectors" --out "$OUT" "${ISO[@]}" | tee "$OUT/check.txt"

log "benchmark (${SECONDS_PER_RUN}s runs)"
rig bench "$ENTRY" --seconds "$SECONDS_PER_RUN" --out "$OUT" "${ISO[@]}" | tee "$OUT/bench.txt"

log "done; copy $OUT back and run: onionrig board $OUT/*.jsonl"
