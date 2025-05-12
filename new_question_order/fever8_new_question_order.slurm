#!/bin/bash
#SBATCH --job-name=FEVER8
#SBATCH --account=ec403
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=16G
#SBATCH --partition=accel
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --output=out_fever8_%j.log
#SBATCH --error=out_fever8_%j.log

# Default to not run setup unless specified
SETUP=${1:-0}

# Shift the first argument (SETUP) so we can pass remaining args
shift 2>/dev/null || true

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

# Setup phase - only runs if SETUP=1
if [ "$SETUP" -eq 1 ]; then
    echo "Running setup and installation..."

    # Load modules
    module purge  # Clear any previously loaded modules
    module load Python/3.10.8-GCCcore-12.2.0
    module load Miniconda3/22.11.1-1

    # We'll use the shared conda environment
    echo "Using shared conda environment at ${SHARED_PROJECT_PATH}/miniconda3"
    echo "Setup completed"
fi

# Set environment variables for GPU
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

# Activate conda environment from shared location
source "${SHARED_PROJECT_PATH}/miniconda3/etc/profile.d/conda.sh"
conda activate hero

echo "Current environment:"
conda info --envs

echo "Files in current directory:"
ls -la

echo "Running the baseline system with arguments: $@"
# Use the full path to run_system.sh from the submission directory
bash "${SUBMIT_DIR}/run_system_new_question_order.sh" "$@"

echo "Job completed"