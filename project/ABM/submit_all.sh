#!/bin/bash
set -eu

# Submit array
ARRAY_JOB=$(sbatch --parsable submit_array.sh)
echo "Array job submitted: ${ARRAY_JOB}"

# Submit combine job with dependency
COMBINE_JOB=$(sbatch --parsable \
    --dependency=afterok:${ARRAY_JOB} \
    --job-name=combine_results \
    --account=coms038604 \
    --partition=compute \
    --time=00:30:00 \
    --mem=8G \
    --cpus-per-task=1 \
    --output=logs/combine_%j.out \
    --error=logs/combine_%j.err \
    --wrap="module load languages/python/3.12.3; cd ${SLURM_SUBMIT_DIR:-$PWD}; python combine_results.py")

echo "Combine job submitted: ${COMBINE_JOB} (depends on ${ARRAY_JOB})"