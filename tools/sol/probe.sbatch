#!/usr/bin/env bash
# A/B probe on Sol's MI210: build the variants of a probe directory (made by an
# entry's probe/mkprobe.sh) and time them in interleaved rounds, which cancels
# most of the GPU's drift between jobs. A few minutes of GPU time, run by the
# operator; competitor submissions go through the judge (tools/arena/).
#
# Usage: sbatch --output=DIR/probe-%j.log probe.sbatch DIR
# DIR holds probe.tar.gz. The result is the last line of the log: per variant,
# the mean rate in G/s and the spread over rounds, as printed by probe/run.sh.
#
#SBATCH --job-name=onionrig-probe
#SBATCH --partition=fpga
#SBATCH --qos=public
#SBATCH --gres=gpu:mi200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=00:30:00

set -euo pipefail
DIR=$(realpath "$1")
SIF=$HOME/onionrig-arena/rocm-dev-7.2.4.sif
# The same container as the judge, seeing only this directory and the GPU.
APPT=(apptainer exec --containall --no-home --cleanenv
      --env "ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-0}"
      --bind "$DIR:/probe" --bind /dev/kfd --bind /dev/dri "$SIF")

cd "$DIR"
rm -rf src build.log && mkdir src
if ! "${APPT[@]}" bash -c "cd /probe && tar -xzf probe.tar.gz -C src --no-same-owner &&
        cd src && unshare --user --map-root-user --net make -j4 entry GPU_ARCH=gfx90a" \
        > build.log 2>&1; then
    echo "build failed:"; tail -20 build.log; exit 1
fi
# probe/run.sh exits 1 by design (it is not a conformant --verify).
"${APPT[@]}" bash -c "cd /probe/src && ./entry --verify" || true
