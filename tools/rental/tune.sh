#!/usr/bin/env bash
# Session A, one GPU: measure, and choose the build for the multi-GPU run.
# Phases, in order, each writing its results into OUT as they land
# (progress.log says where things stand):
#   snapshot  hardware, ROCm, partition mode, power limits
#   build     every build in builds.txt: the kit's prebuilt binaries, or local builds
#   verify    each build's --verify against the shared vectors; a failure drops it
#   micro     primitive throughput: does v_mad_u64_u32 issue at full rate here?
#   profile   the profiling builds: effective shader clock and cycles per phase
#   sweep     interleaved rounds of every build (onionrig sweep); a fixed rule picks one
#   check     the rig's conformance suite on the pick
#   bench     the rig's official proof-of-work benchmark of the pick
#
# Usage: tools/rental/tune.sh OUT_DIR [PHASE ...]    (default: every phase)
# Knobs (environment): GPU=0, ROUNDS=6, CELL_SECONDS=30, WARMUP=120 (seconds
# of the base build before the sweep's first round, so it starts hot),
# PROFILE_SECONDS=60.
set -euo pipefail
umask 077  # results tie an address to this account

OUT=$(realpath -m "${1:?usage: tune.sh OUT_DIR [PHASE ...]}")
shift
mkdir -p "$OUT"
# shellcheck source-path=SCRIPTDIR source=lib.sh
. "$(dirname "$0")/lib.sh"
GPU=${GPU:-0}
ROUNDS=${ROUNDS:-6}
CELL_SECONDS=${CELL_SECONDS:-30}
WARMUP=${WARMUP:-120}
PROFILE_SECONDS=${PROFILE_SECONDS:-60}

all_builds() { builds | awk '{ print $1 }'; }

# A build passed verify (every build counts as passed when verify has not run).
passed() { [ ! -f "$OUT/verify.jsonl" ] || grep -q "\"build\": \"$1\", \"ok\": true" "$OUT/verify.jsonl"; }

picked() {
    local pick=""
    [ -f "$OUT/PICK" ] && pick=$(cat "$OUT/PICK")
    echo "${pick:-base}"
}

phase_snapshot() {
    local agents
    snapshot
    agents=$(gpu_agents | wc -l)
    if [ "$agents" -gt 1 ]; then
        log "note: $agents GPU agents visible and tuning uses agent $GPU alone. In CPX" \
            "partition mode an agent is one XCD: build ratios hold, absolute rates do not."
    fi
}

phase_build() {
    # shellcheck disable=SC2046  # build names are single words
    select_builds $(all_builds)
}

phase_verify() {
    local name started ok
    : > "$OUT/verify.jsonl"  # a rerun replaces the earlier verdicts
    for name in $(all_builds); do
        started=$SECONDS
        ok=false
        on_gpu "$GPU" "$OUT/bin/$name/entry" --verify "$ROOT/vectors" \
            > "$OUT/verify-$name.txt" 2>&1 && ok=true
        printf '{"build": "%s", "ok": %s, "seconds": %d, "log": "verify-%s.txt"}\n' \
            "$name" "$ok" $((SECONDS - started)) "$name" >> "$OUT/verify.jsonl"
        log "verify $name: $([ "$ok" = true ] && echo passed || echo FAILED)" \
            "in $((SECONDS - started)) s"
    done
}

phase_micro() {
    log "microbenchmarks (a few minutes): one JSON line per test in microbench.jsonl"
    on_gpu "$GPU" "$OUT/bin/microbench-hip" 30 150 > "$OUT/microbench.jsonl" \
        2> "$OUT/microbench.err" || log "microbenchmarks failed: see microbench.err"
}

phase_profile() {
    local args=() name
    for name in $(all_builds | grep '^prof'); do
        if passed "$name"; then args+=("$name=$OUT/bin/$name/entry"); fi
    done
    if [ ${#args[@]} -eq 0 ]; then
        log "profile: no profiling build passed verify"
        return
    fi
    log "profile: ${#args[@]} builds, $PROFILE_SECONDS s each"
    "$RIG" sweep "${args[@]}" --rounds 1 --seconds "$PROFILE_SECONDS" --device "$GPU" \
        --out "$OUT/profile" \
        "${CONFINE[@]}" > "$OUT/profile.txt" 2>&1 || log "profile failed: see profile.txt"
}

phase_sweep() {
    local args=() name
    passed base || die "the base build failed verify; nothing to compare against"
    for name in $(all_builds | grep -v '^prof'); do
        if passed "$name"; then
            args+=("$name=$OUT/bin/$name/entry")
        else
            log "sweep: leaving out $name, which failed verify"
        fi
    done
    args+=("base2=$OUT/bin/base/entry")  # the base again, as the noise floor
    log "sweep: ${#args[@]} builds x $ROUNDS rounds of $CELL_SECONDS s after $WARMUP s warm-up"
    "$RIG" sweep "${args[@]}" --rounds "$ROUNDS" --seconds "$CELL_SECONDS" --warmup "$WARMUP" \
        --device "$GPU" --out "$OUT/sweep" "${CONFINE[@]}" 2>&1 | tee -i "$OUT/sweep.txt"
    sed -n 's/^pick: //p' "$OUT/sweep.txt" | tail -n 1 > "$OUT/PICK"
    log "sweep picked: $(picked)"
}

phase_check() {
    local pick
    pick=$(picked)
    log "check: the rig's conformance suite on $pick"
    if ! "$RIG" check "$OUT/bin/$pick/entry" --vectors "$ROOT/vectors" --device "$GPU" \
            --out "$OUT/check" "${CONFINE[@]}" 2>&1 | tee -i "$OUT/check.txt"; then
        die "$pick does not conform: see check.txt"
    fi
}

phase_bench() {
    local pick
    pick=$(picked)
    log "bench: the rig's official benchmark of $pick (cool-downs between runs, 5-15 min)"
    if ! "$RIG" bench "$OUT/bin/$pick/entry" --device "$GPU" --out "$OUT/bench" "${CONFINE[@]}" \
            2>&1 | tee -i "$OUT/bench.txt"; then
        die "bench failed: see bench.txt"
    fi
}

PHASES=("$@")
[ ${#PHASES[@]} -gt 0 ] || PHASES=(snapshot build verify micro profile sweep check bench)
for phase in "${PHASES[@]}"; do
    declare -F "phase_$phase" > /dev/null || die "unknown phase '$phase' (see the top of $0)"
done
log "tune.sh, GPU $GPU: ${PHASES[*]}"
normalize_visibility
ensure_rig
set_confinement
start_telemetry "$GPU"
trap stop_telemetry EXIT
for phase in "${PHASES[@]}"; do "phase_$phase"; done
log "done. Pick: $(picked) (sweep.txt); official rate: bench.txt; copy $OUT home"
