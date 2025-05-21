#!/bin/bash
#SBATCH --job-name=FEVER8
#SBATCH --account=ec403
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=16G
#SBATCH --partition=accel
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --output=out_comp_ret_%j.log
#SBATCH --error=out_comp_ret_%j.log

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

# Load required CUDA module for GPU tasks
module load CUDA/11.7.0

# Set environment variables for GPU
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

source ../config.sh
# Activate conda environment from shared location
source "${SHARED_PROJECT_PATH}/miniconda3/etc/profile.d/conda.sh"
conda activate hero

echo "Current environment:"
conda info --envs

echo "Files in current directory:"
ls -la

echo "Running the baseline system with arguments: $@"
# Use the full path to run_system.sh from the submission directory
"${SUBMIT_DIR}/python compare_retrieval.py --positive_file ${DATA_STORE}/multi_fc/dev_reranking_top_k_positive.json --negative_file ${DATA_STORE}/multi_fc/dev_reranking_top_k_negative.json --objective_file ${DATA_STORE}/multi_fc/dev_reranking_top_k_objective.json" "$@"

echo "Job completed"
