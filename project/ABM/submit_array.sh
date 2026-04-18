#!/bin/bash
#SBATCH --job-name=abm_array
#SBATCH --account=coms038604
#SBATCH --partition=compute
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --array=0-15                  # 16 tasks (seeds 0-15)
#SBATCH --output=logs/slurm_%A_%a.out
#SBATCH --error=logs/slurm_%A_%a.err

set -eu

module load languages/python/3.12.3

cd "$SLURM_SUBMIT_DIR"

mkdir -p logs results

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Running on: $(hostname)"
echo "Start: $(date)"

python run.py

echo "End: $(date)"