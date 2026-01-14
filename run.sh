#!/usr/bin/env bash
set -euo pipefail

module purge
module load gnu13/13.2.0 || true

export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MKL_THREADING_LAYER=GNU
export HF_HOME=/data/researchHome/pdou/.cache/hf
export HUGGINGFACE_HUB_CACHE=/data/researchHome/pdou/.cache/huggingface
export MPLCONFIGDIR=/tmp/$USER-mpl
# Stop CUDA probing and other surprises on login nodes
export CUDA_VISIBLE_DEVICES=""
export TRANSFORMERS_NO_TORCHVISION=1
mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$MPLCONFIGDIR"

unset PYTHONPATH || true

cd /data/researchHome/pdou/hpc_cluster_report

# Ensure venv exists (Rocky/RHEL: --copies avoids lib/lib64 weirdness)
if [[ ! -d my_env ]]; then
  python3 -m venv --copies my_env
fi
source my_env/bin/activate
# python -m ensurepip --upgrade || true
# python -m pip install -U pip setuptools wheel

# # Import check; if it fails, repair packages
# python -c "import numpy, pandas, pytz" || {
#   echo "[run.sh] Repairing scientific stack..."
#   pip install --no-cache-dir --force-reinstall \
#     numpy==1.26.4 pandas==2.2.2 pytz==2025.2
# }

# Run your programs
python collect_data.py
python system_info.py

# Submit Slurm job (compute7)
# /source /data/researchHome/pdou/hpc_cluster_report/.venv/bin/activate   # if not already

sbatch /data/researchHome/pdou/hpc_cluster_report/cluster_report.slurm
