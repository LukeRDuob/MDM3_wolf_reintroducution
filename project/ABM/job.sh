#!/bin/bash
#SBATCH --job-name=testing_experiments
#SBATCH --account=coms038604
#SBATCH --partition=compute
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --output=slurm-%x_%j.out
#SBATCH --error=slurm-%x_%j.out

set -eu

# Load Python
module load languages/python/3.12.3

# Go to your ABM folder
cd "$SLURM_SUBMIT_DIR"

# Run your model
python run.py