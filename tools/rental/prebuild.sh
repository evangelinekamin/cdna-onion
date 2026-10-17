#!/usr/bin/env bash
# Runs inside the ROCm container (kit.sh starts it, offline): fat binaries of
# every build in builds.txt and of the microbenchmark, into the kit's prebuilt/,
# with BUILDINFO.txt describing them.
#
# Usage: prebuild.sh KIT_DIR COMMIT ARCH...
set -euo pipefail
KIT=$1 COMMIT=$2
shift 2
ARCHS=("$@")
OUTDIR=$KIT/prebuilt
WORK=$(mktemp -d)
mkdir -p "$OUTDIR"

pids=()
while read -r name defs; do
    mkdir -p "$WORK/$name" "$OUTDIR/$name"
    cp -r "$KIT/entries/combined/src" "$KIT/entries/combined/Makefile" "$WORK/$name/"
    (make -C "$WORK/$name" entry GPU_ARCH="${ARCHS[*]}" DEFS="$defs" > "$WORK/$name.log" 2>&1 &&
        cp "$WORK/$name/entry" "$OUTDIR/$name/entry") &
    pids+=("$!")
done < <(grep -Ev '^[[:space:]]*(#|$)' "$KIT/tools/rental/builds.txt")
status=0
# The microbenchmark includes ../../entries/baseline/src, so keep that layout.
mkdir -p "$WORK/mb/tools" "$WORK/mb/entries/baseline"
cp -r "$KIT/tools/microbench" "$WORK/mb/tools/"
cp -r "$KIT/entries/baseline/src" "$WORK/mb/entries/baseline/"
if make -C "$WORK/mb/tools/microbench" hip HIP_ARCHS="${ARCHS[*]}" > "$WORK/microbench.log" 2>&1
then
    cp "$WORK/mb/tools/microbench/build/microbench-hip" "$OUTDIR/microbench-hip"
else
    status=1
fi
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
if [ "$status" != 0 ]; then
    tail -n 30 "$WORK"/*.log >&2
    exit 1
fi

{
    echo "commit: $COMMIT"
    echo "hip: $(hipcc --version 2>&1 | grep -m1 'HIP version')"
    echo "archs: ${ARCHS[*]}"
    echo "built: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    while read -r name defs; do
        echo "build $name sha256 $(sha256sum < "$OUTDIR/$name/entry" | cut -c1-64) defs ${defs:-none}"
    done < <(grep -Ev '^[[:space:]]*(#|$)' "$KIT/tools/rental/builds.txt")
    echo "microbench-hip sha256 $(sha256sum < "$OUTDIR/microbench-hip" | cut -c1-64)"
} > "$OUTDIR/BUILDINFO.txt"
cat "$OUTDIR/BUILDINFO.txt"
