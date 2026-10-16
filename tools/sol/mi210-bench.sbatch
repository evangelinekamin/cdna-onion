#!/usr/bin/env bash
# Sol MI210 benchmarking job: a short, sanctioned benchmarking run, not a search.
#
# Sol's MI210s are the two GPUs in node sg250. They live in the `fpga` partition,
# which accepts only the `public` QoS and gives a node to one user at a time,
# and Slurm calls their GPU type `mi200` (sg250 is the node, not the type).
#
# If `module load rocm` fails, this job's log lists the ROCm modules Sol offers.
# Resubmit with one of them:
#   sbatch --export=ALL,ROCM_MODULE=<name> tools/sol/mi210-bench.sbatch
#
#SBATCH --job-name=onionrig-mi210
#SBATCH --partition=fpga
#SBATCH --qos=public
#SBATCH --gres=gpu:mi200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=onionrig-%j.log

set -eo pipefail
# Loaded before `set -u`: some Lmod versions read unset variables while loading.
module load "${ROCM_MODULE:-rocm}" || { module avail rocm 2>&1; exit 1; }
set -u
cd "${SLURM_SUBMIT_DIR}"
exec tools/sol/run-all.sh "results/mi210-${SLURM_JOB_ID}" 60
