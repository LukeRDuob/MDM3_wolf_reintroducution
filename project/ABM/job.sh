#!/bin/bash
#SBATCH --job-name=coms038604
#SBATCH --partition=cpu
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=slurm-%x_%j.out

set -eu

# Load conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base   # or abm_env if you cloned it

# Go to your ABM folder
cd /user/home/sd23327/ABM

# Run your model
python experiments.py