#!/usr/bin/env bash
# Session B, every GPU: the real search and nothing else. Checks every GPU with
# the build's --verify (in parallel, well under a minute), then runs
# `onionrig search` on each GPU that passed, every entry on CPUs of its own,
# until stopped: Ctrl-C once stops after in-flight work, twice aborts. Verified
# hits stream to OUT/search/search.jsonl, progress to OUT/search.log.
#
# Run it inside tmux (or under nohup): a dropped SSH connection must not end it.
#
# Usage: tools/rental/grind.sh OUT_DIR POINTS_JSON PREFIX [PREFIX ...]
# Knobs (environment): BUILD=base (use the build tune.sh picked), GPUS=all
# (or a list like 0-3,6).
set -euo pipefail
umask 077  # results tie an address to this account

usage="usage: grind.sh OUT_DIR POINTS_JSON PREFIX [PREFIX ...]"
OUT=$(realpath -m "${1:?$usage}")
POINTS=$(realpath "${2:?$usage}")
shift 2
[ $# -gt 0 ] || { echo "$usage" >&2; exit 2; }
mkdir -p "$OUT"
# shellcheck source-path=SCRIPTDIR source=lib.sh
. "$(dirname "$0")/lib.sh"
BUILD=${BUILD:-base}
builds | awk '{ print $1 }' | grep -qx -- "$BUILD" ||
    die "unknown build '$BUILD': the names are in tools/rental/builds.txt"

log "grind.sh: build $BUILD, prefixes $*"
normalize_visibility
ensure_rig
set_confinement
snapshot
select_builds "$BUILD"

count=$(gpu_agents | wc -l)
[ "$count" -gt 0 ] || die "rocminfo lists no GPU"
read -ra devices <<< "$("$ROOT/.venv/bin/python" -c 'import sys
from onionrig.cli import parse_cpus
print(*parse_cpus(sys.argv[1]))' "${GPUS:-0-$((count - 1))}")"

log "verifying $BUILD on GPUs ${devices[*]}, in parallel"
for g in "${devices[@]}"; do
    (on_gpu "$g" "$OUT/bin/$BUILD/entry" --verify "$ROOT/vectors" > "$OUT/verify-g$g.txt" 2>&1
     echo $? > "$OUT/verify-g$g.rc") &
done
wait
good=()
for g in "${devices[@]}"; do
    if [ "$(cat "$OUT/verify-g$g.rc")" = 0 ]; then
        good+=("$g")
    else
        log "GPU $g FAILED verify (verify-g$g.txt); leaving it out"
    fi
done
[ ${#good[@]} -gt 0 ] || die "no GPU passed verify"

start_telemetry "${good[@]}"
trap stop_telemetry EXIT
prefixes=()
for p in "$@"; do prefixes+=(--prefix "$p"); done
log "searching on GPUs ${good[*]}; hits -> $OUT/search/search.jsonl"
"$RIG" search "$OUT/bin/$BUILD/entry" --points "$POINTS" "${prefixes[@]}" \
    --devices "$(IFS=,; echo "${good[*]}")" --out "$OUT/search" "${CONFINE[@]}" 2>&1 |
    tee -i -a "$OUT/search.log"
log "search ended: $(grep -c '"kind": "hit"' "$OUT/search/search.jsonl" 2>/dev/null || echo 0)" \
    "verified hit(s) in search/search.jsonl"
