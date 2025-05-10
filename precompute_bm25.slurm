#!/bin/bash
#SBATCH --job-name=BM25-Precompute
#SBATCH --account=ec403
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=16G
#SBATCH --partition=accel
#SBATCH --cpus-per-task=96
#SBATCH --output=bm25_precompute_%j.log
#SBATCH --error=bm25_precompute_%j.log

# Get the absolute path to the user's home directory version of the code
# The job was submitted from this directory
SUBMIT_DIR=$(pwd)

# Path to shared project directory
SHARED_PROJECT_PATH="/fp/projects01/ec403/IN5550_students/EivindogNora/FEVER-8-Shared-Task"

# Print diagnostics
echo "Job submitted from: ${SUBMIT_DIR}"
echo "SLURM working directory: $(pwd)"
echo "Shared project path: ${SHARED_PROJECT_PATH}"

# Always change to the submission directory to find our scripts
cd "${SUBMIT_DIR}"
echo "Changed to directory: $(pwd)"

# Load Python module if needed
module load Python/3.10.8-GCCcore-12.2.0

# Activate conda environment from shared location
source "${SHARED_PROJECT_PATH}/miniconda3/etc/profile.d/conda.sh"
conda activate hero

echo "Current environment:"
conda info --envs

echo "Files in current directory:"
ls -la

# Run the precomputation script
# Replace paths with your actual paths
python precompute_retrieval.py \
    --knowledge_store_dir "${SHARED_PROJECT_PATH}/knowledge_store/dev" \
    --bm25_output_dir "${SHARED_PROJECT_PATH}/knowledge_store/dev/precomputed_bm25" \
    --workers 96

echo "Job completed at: $(date)"