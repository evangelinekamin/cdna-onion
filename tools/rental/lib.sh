# shellcheck shell=bash
# Shared by tune.sh and grind.sh (sourced, not run): logging, the environment
# snapshot, GPU visibility, picking or building the entry binaries, telemetry.
# The caller sets OUT, the results directory.

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
ENTRY_SRC=$ROOT/entries/combined
BUILDS_FILE=$ROOT/tools/rental/builds.txt
RIG=$ROOT/.venv/bin/onionrig

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "$OUT/progress.log"; }
die() { log "ERROR: $*"; exit 1; }

# The rig's Python environment, once per host (it downloads uv and Python).
ensure_rig() {
    [ -x "$RIG" ] && return
    log "setting up the rig's Python environment (tools/sol/setup.sh, about a minute)"
    "$ROOT/tools/sol/setup.sh" >> "$OUT/setup.log" 2>&1 || die "rig setup failed: see setup.log"
}

# builds.txt as "NAME DEFS" lines, without comments and blank lines.
builds() { grep -Ev '^[[:space:]]*(#|$)' "$BUILDS_FILE"; }
defs_of() { builds | awk -v n="$1" '$1 == n { $1 = ""; sub(/^ +/, ""); print }'; }

# Providers sometimes preset HIP_VISIBLE_DEVICES or CUDA_VISIBLE_DEVICES. HIP
# applies them on top of ROCR_VISIBLE_DEVICES, and the rig narrows each entry
# with ROCR alone, so fold a HIP-level list into ROCR and drop the rest.
normalize_visibility() {
    local hip=${HIP_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-}}
    if [ -n "$hip" ] && [ -z "${ROCR_VISIBLE_DEVICES:-}" ]; then
        export ROCR_VISIBLE_DEVICES=$hip
        log "GPU visibility: HIP-level list $hip moved to ROCR_VISIBLE_DEVICES"
    fi
    unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL
}

# Run a command on the index-th visible GPU, the way the rig narrows an entry:
# ROCR gets the physical id, HIP sees the one survivor.
on_gpu() {
    local index=$1 physical=$1 visible
    shift
    if [ -n "${ROCR_VISIBLE_DEVICES:-}" ]; then
        IFS=, read -ra visible <<< "$ROCR_VISIBLE_DEVICES"
        physical=${visible[$index]}
    fi
    ROCR_VISIBLE_DEVICES=$physical HIP_VISIBLE_DEVICES=0 "$@"
}

# GPU agents as rocminfo lists them, one gfx name per GPU (captured whole first:
# `rocminfo | grep -m1` dies of SIGPIPE under pipefail).
gpu_agents() {
    local info
    info=$(rocminfo 2>&1) || return 1
    grep -oE '^[[:space:]]*Name:[[:space:]]+gfx[0-9a-f]+' <<< "$info" | awk '{ print $2 }'
}

# Where things stand before anything runs: hardware, ROCm, partitions, limits.
snapshot() {
    {
        echo "== date"; date -Is
        echo "== host"; hostname; uname -a; head -3 /etc/os-release
        echo "== cpu"; lscpu | grep -E '^(Model name|Socket|Core|Thread|CPU\(s\)|NUMA)'
        echo "== memory"; free -g
        echo "== cpus allowed"; nproc
        echo "== gpus"; gpu_agents | sort | uniq -c
        echo "== visibility"; env | grep -E '^(ROCR|HIP|CUDA|GPU_DEVICE)_' | sort
        echo "== hipcc"; hipcc --version
        echo "== amdgpu driver"; cat /sys/module/amdgpu/version
        echo "== amd-smi"; amd-smi version
        echo "== partitions"; amd-smi partition
        echo "== kit"; head -12 "$ROOT/prebuilt/BUILDINFO.txt"
    } > "$OUT/environment.txt" 2>&1 || true
    rocminfo > "$OUT/rocminfo.txt" 2>&1 || true
    amd-smi static > "$OUT/amd-smi-static.txt" 2>&1 || true
    log "snapshot: environment.txt, rocminfo.txt, amd-smi-static.txt"
}

# CONFINE: the rig's flags for confining entries. The rig refuses to run
# entries without namespaces, or to pin them without taskset, unless told;
# these are our own builds, so carry on without either where the host lacks
# it (the results record --no-isolation).
# shellcheck disable=SC2034  # CONFINE is read by the scripts that source this file
set_confinement() {
    CONFINE=()
    if ! "$ROOT/.venv/bin/python" -c 'import sys
from onionrig.rig.isolation import isolation_available
sys.exit(not isolation_available())'; then
        CONFINE+=(--no-isolation)
        log "namespace isolation unavailable here; running our own builds without it"
    fi
    if ! command -v taskset > /dev/null; then
        CONFINE+=(--cpus all)
        log "taskset is not installed; entries run without CPU pinning"
    fi
}

# OUT/bin/NAME/entry for each named build, and OUT/bin/microbench-hip: the kit's
# prebuilt fat binaries when they cover this GPU and load here, else builds
# with this host's hipcc for its GPU. One line per build in build.jsonl says which.
select_builds() {
    local arch name
    arch=$(gpu_agents | head -1) || true
    [ -n "$arch" ] || die "rocminfo lists no GPU (is /dev/kfd visible to this user?)"
    mkdir -p "$OUT/bin"
    if grep -qE "^archs:.* $arch( |$)" "$ROOT/prebuilt/BUILDINFO.txt" 2>/dev/null &&
            on_gpu 0 "$ROOT/prebuilt/base/entry" --info > "$OUT/prebuilt-info.txt" 2>&1; then
        log "using the kit's prebuilt binaries on $arch ($(grep -m1 '^hip:' "$ROOT/prebuilt/BUILDINFO.txt"))"
        for name in "$@"; do
            mkdir -p "$OUT/bin/$name"
            cp "$ROOT/prebuilt/$name/entry" "$OUT/bin/$name/entry" || die "no prebuilt $name"
            record_build "$name" prebuilt
        done
        cp "$ROOT/prebuilt/microbench-hip" "$OUT/bin/microbench-hip"
    else
        log "the prebuilt binaries do not cover or load on this $arch (prebuilt-info.txt); building"
        local specs=()
        for name in "$@"; do specs+=("$name=$ENTRY_SRC:$(defs_of "$name")"); done
        "$ENTRY_SRC/probe/mkprobe.sh" "$OUT/build" "${specs[@]}" > /dev/null
        make -C "$OUT/build" -j "$(nproc)" entry GPU_ARCH="$arch" > "$OUT/build.log" 2>&1 ||
            die "build failed: see build.log"
        for name in "$@"; do
            mkdir -p "$OUT/bin/$name"
            cp "$OUT/build/$name/entry" "$OUT/bin/$name/entry"
            record_build "$name" "local $(hipcc --version 2>&1 | grep -m1 -oE 'HIP version: \S+')"
        done
        make -C "$ROOT/tools/microbench" hip HIP_ARCHS="$arch" >> "$OUT/build.log" 2>&1 ||
            die "microbenchmark build failed: see build.log"
        cp "$ROOT/tools/microbench/build/microbench-hip" "$OUT/bin/microbench-hip"
    fi
}

record_build() {
    printf '{"build": "%s", "defs": "%s", "source": "%s", "sha256": "%s"}\n' "$1" \
        "$(defs_of "$1")" "$2" "$(sha256sum < "$OUT/bin/$1/entry" | cut -c1-64)" \
        >> "$OUT/build.jsonl"
}

# One background sampler per GPU, until stop_telemetry: the rig's JSON lines
# where it can read this host's amd-smi, raw amd-smi output otherwise.
TELEMETRY=()
start_telemetry() {
    local g
    for g in "$@"; do
        if "$RIG" telemetry --device "$g" > "$OUT/telemetry-probe-g$g.txt" 2>&1; then
            setsid "$RIG" telemetry --device "$g" --follow "$OUT/telemetry-g$g.jsonl" \
                --interval 2 > /dev/null 2>&1 &
        else
            setsid bash -c "while :; do date +%s; amd-smi metric -g $g; sleep 5; done" \
                > "$OUT/telemetry-raw-g$g.txt" 2>&1 &
        fi
        TELEMETRY+=("$!")
    done
}
stop_telemetry() {
    local pid
    for pid in "${TELEMETRY[@]}"; do kill -TERM -- "-$pid" 2>/dev/null || true; done
}
