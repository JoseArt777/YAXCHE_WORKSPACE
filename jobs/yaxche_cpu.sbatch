#!/bin/bash
# YAXCHE - Benchmark de Escalamiento Fuerte CPU (Numba) en nodo exclusivo kura
# Enviar desde la raiz del repositorio YAXCHE:
#     sbatch jobs/yaxche_cpu.sbatch

#SBATCH --job-name=yaxche-cpu
#SBATCH --partition=kura
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --hint=nomultithread
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%x-%j.out

set -eu
[ "$(type -t module)" = function ] || source /etc/profile.d/modules.sh
module load mamba/Python4HPC

export NUMBA_CACHE_DIR="/tmp/$USER/yaxche_numba_cache"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMBA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
mkdir -p "$NUMBA_CACHE_DIR" results

echo "== $(date -Is) Host: $(hostname) CPUs: ${SLURM_CPUS_PER_TASK} =="

# 1. Escalamiento fuerte sobre datos reales de Guatemala (450x450)
python scripts/benchmark_kabre.py --mode cpu --npz data/processed/guatemala_real.npz --steps 48 --repeat 7

# 2. Escalamiento fuerte sobre malla de alta resolucion (2000x2000 = 4M celdas)
python scripts/benchmark_kabre.py --mode cpu --size 2000 --steps 24 --repeat 5

