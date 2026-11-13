#!/bin/bash
# YAXCHE - Benchmark CUDA + Ensamble Monte Carlo en GPU NVIDIA L40S (nukwa-l40s)
# Enviar desde la raiz del repositorio YAXCHE:
#     sbatch jobs/yaxche_gpu.sbatch

#SBATCH --job-name=yaxche-gpu
#SBATCH --partition=nukwa-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%x-%j.out

set -eu
[ "$(type -t module)" = function ] || source /etc/profile.d/modules.sh
module load mamba/Python4HPC

export CUPY_CACHE_DIR="/tmp/$USER/yaxche_cupy_cache"
export NUMBA_CACHE_DIR="/tmp/$USER/yaxche_numba_cache"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
mkdir -p "$CUPY_CACHE_DIR" "$NUMBA_CACHE_DIR" results

echo "== $(date -Is) Host: $(hostname) =="
nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total --format=csv

# 1. Benchmark coalescente vs strided + Ensamble Monte Carlo (200 replicas) sobre Guatemala real
python scripts/benchmark_kabre.py --mode gpu --npz data/processed/guatemala_real.npz --steps 48 --repeat 10 --mc-runs 200

# 2. Escala nacional (4500x4500 = 20.25M celdas, resolucion ~100m)
python scripts/benchmark_kabre.py --mode gpu --size 4500 --steps 48 --repeat 5 --mc-runs 0

