#!/bin/bash
#SBATCH --job-name=MultiFC_Auxiliary
#SBATCH --account=ec403
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=16G
#SBATCH --partition=accel
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --output=out_multi_fc_auxiliary_%j.log
#SBATCH --error=out_multi_fc_auxiliary_%j.log

# Print the current working directory
echo "Working directory: $(pwd)"
SUBMIT_DIR=$(pwd)
SHARED_PROJECT_PATH="/fp/projects01/ec403/IN5550_students/EivindogNora/FEVER-8-Shared-Task"

# Load required CUDA module for GPU tasks
module load CUDA/11.7.0

# Set environment variables for GPU
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

# Activate conda environment - this is needed regardless of setup
source "${SHARED_PROJECT_PATH}/miniconda3/etc/profile.d/conda.sh"
conda activate hero

echo "Current environment:"
conda info --envs

echo "Running multi-perspective auxiliary experiment with arguments: $@"

# Run the auxiliary experiment script
"${SUBMIT_DIR}/biased_system_inference.sh" "$@"

echo "Job completed"