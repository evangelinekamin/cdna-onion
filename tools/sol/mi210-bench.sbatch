#!/usr/bin/env bash
# Sol MI210 benchmarking job: a short benchmarking run (about 5 minutes), not a search.
#
# Sol's MI210s are the two GPUs in node sg250. They live in the `fpga` partition,
# which accepts only the `public` QoS and gives a node to one user at a time,
# and Slurm calls their GPU type `mi200` (sg250 is the node, not the type).
#
# sg250 has ROCm installed system-wide (7.2.3 as of October 2026), so no module
# is loaded there. Where hipcc is missing, the job loads a `rocm` module; if that
# fails, its log lists the ROCm modules on offer. Resubmit with one of them:
#   sbatch --export=ALL,ROCM_MODULE=<name> tools/sol/mi210-bench.sbatch
#
#SBATCH --job-name=onionrig-mi210
#SBATCH --partition=fpga
#SBATCH --qos=public
#SBATCH --gres=gpu:mi200:1
#
# Sized from seff on the first run (4 minutes, about 1.1 cores busy, 330 MB peak;
# the HIP compile peaks at 220 MB): the GPU does the work. Four cores give the
# entry its two pinned cores and hit verification the other two. The hour covers
# the worst case, ten benchmark runs each after a full cool-down (about 42 min).
#SBATCH --cpus-per-task=4
#SBATCH --mem=2G
#SBATCH --time=01:00:00
#SBATCH --output=onionrig-%j.log

set -eo pipefail
# Before `set -u`: some Lmod versions read unset variables while loading.
if ! command -v hipcc >/dev/null; then
    module load "${ROCM_MODULE:-rocm}" || { module avail rocm 2>&1; exit 1; }
fi
set -u
cd "${SLURM_SUBMIT_DIR}"
OUT="results/mi210-${SLURM_JOB_ID}"
mkdir -p "$OUT"

# The work runs as job step 0, so its CPU and memory accounting is final once the
# step ends; seff on a job that is still running reads nothing for its batch step.
status=0
srun --cpus-per-task="$SLURM_CPUS_PER_TASK" tools/sol/run-all.sh "$OUT" 60 || status=$?
for _ in $(seq 12); do  # the step's record reaches the accounting database within seconds
    sacct -n -j "${SLURM_JOB_ID}.0" -o State 2>/dev/null \
        | grep -qE 'COMPLETED|FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL' && break
    sleep 5
done
seff "$SLURM_JOB_ID" > "$OUT/seff.txt" 2>&1 || true
exit "$status"
