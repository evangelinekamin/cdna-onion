#!/usr/bin/env bash
# Judge one arena submission: build it, then run the rig on it. Both happen inside
# a container that sees only this submission's directory, the rig (read-only),
# and the GPU, so competitor code never reaches anything else on this account.
# The build has no network, and the rig adds its own sandbox around the entry.
#
# Usage (the runner does this): sbatch judge.sbatch SUBMISSION_DIR check|bench
# SUBMISSION_DIR holds sub.tar.gz; this writes src/, out/, build.log, and result.json.
#
#SBATCH --job-name=onionrig-judge
#SBATCH --partition=fpga
#SBATCH --qos=public
#SBATCH --gres=gpu:mi200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=01:00:00

set -euo pipefail
SUB=$(realpath "$1")
KIND=$2
case "$KIND" in check|bench) ;; *) echo "kind must be check or bench" >&2; exit 2 ;; esac
RIG=$HOME/onionrig
SIF=$HOME/onionrig-arena/rocm-dev-7.2.4.sif
ARCH=gfx90a

cd "$SUB"
rm -rf src out build.log result.json
mkdir -p src out
# The rig is bound at its own path so its Python environment resolves inside.
# Only ROCR_VISIBLE_DEVICES passes through: HIP would apply Slurm's
# CUDA_VISIBLE_DEVICES on top of it and could hide the allocated GPU.
APPT=(apptainer exec --containall --no-home --cleanenv
      --env "ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-0}"
      --bind "$SUB:/sub" --bind "$RIG:$RIG:ro" --bind /dev/kfd --bind /dev/dri
      "$SIF")

# Unpack and build offline, from the submission's own sources only.
started=$(date +%s)
build_ok=0
if "${APPT[@]}" bash -c "cd /sub && tar -xzf sub.tar.gz -C src --no-same-owner &&
        cd src && timeout 900 unshare --user --map-root-user --net make entry GPU_ARCH=$ARCH" \
        > build.log 2>&1 \
   && [ -x src/entry ]; then
    build_ok=1
elif [ ! -x src/entry ]; then
    echo "(judge) no executable ./entry after 'make entry GPU_ARCH=$ARCH'" >> build.log
fi
build_seconds=$(( $(date +%s) - started ))

if [ "$build_ok" = 1 ]; then
    if [ "$KIND" = check ]; then
        "${APPT[@]}" "$RIG/.venv/bin/onionrig" check /sub/src/entry --vectors "$RIG/vectors" \
            --json /sub/out/check.json --out /sub/out > out/rig.txt 2>&1 || true
    else
        "${APPT[@]}" "$RIG/.venv/bin/onionrig" bench /sub/src/entry --out /sub/out \
            > out/rig.txt 2>&1 || true
    fi
fi
status=0
"${APPT[@]}" "$RIG/.venv/bin/python" -m onionrig.arena.judge \
    "$KIND" /sub/out "$build_ok" "$build_seconds" /sub/build.log /sub/result.json || status=$?
seff "$SLURM_JOBID" || true
exit "$status"
