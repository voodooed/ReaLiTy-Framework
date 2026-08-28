#!/bin/bash
#SBATCH --job-name=reality-kitti2cadc
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=16:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# KITTI -> physics weather model (snow) -> PICGAN -> CADC, on one V100.
#
# Walltime: measured throughput is 23.5 frames/s for a full GAN step at batch 8
# (341 ms/step, 6.7 GiB), so the ~5,000-frame training convention runs at roughly
# 3.6 min/epoch -> ~12 h for 200 epochs. The request is 16 h to leave margin; if
# the job is killed anyway, simply resubmit -- `reality train` resumes from the
# last full checkpoint rather than restarting.
#
# Preparation (the weather model + projection over train/) happens once inside the same
# command and is cached, so later epochs and later jobs never repeat it.

set -euo pipefail

module load python/3.11 cuda/12.6            # adjust to the cluster's module names
conda activate reality                       # adjust to your environment

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
CONFIG="${REPO}/reality/configs/weather/kitti_to_cadc.yaml"

# Keep the prepared cache on fast node-local scratch: it is read every epoch.
# $SLURM_TMPDIR is node-local and wiped at job end, so a multi-job run should
# point CACHE_ROOT at persistent scratch instead to keep the cache across jobs.
CACHE_ROOT="${SLURM_TMPDIR:-/tmp}/reality-cache"
RUN_DIR="${REPO}/runs/kitti_to_cadc"

mkdir -p "${RUN_DIR}" "${CACHE_ROOT}" "${REPO}/logs"
cd "${REPO}"

echo "host       : $(hostname)"
echo "gpu        : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "config     : ${CONFIG}"
echo "cache root : ${CACHE_ROOT}"
echo "run dir    : ${RUN_DIR}"

# Fail fast if the GPU cannot actually run a kernel: cuda.is_available alone
# returns true for a build with no kernels for this card.
python - <<'PY'
import torch
assert torch.cuda.is_available, "no CUDA device visible"
torch.mm(torch.randn(64, 64, device="cuda"), torch.randn(64, 64, device="cuda"))
torch.cuda.synchronize
print(f"cuda ok: torch {torch.__version__}, {torch.cuda.get_device_name(0)}")
PY

# One command: prepare (cached) -> statistics over the whole prepared set -> train.
srun python -m reality train \
    --config "${CONFIG}" \
    --cache-root "${CACHE_ROOT}" \
    --checkpoint-every 5

echo "done. checkpoints and stats in the run directory:"
ls -lh "${RUN_DIR}" 2>/dev/null || true
