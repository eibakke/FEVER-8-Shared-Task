#!/bin/bash
#SBATCH --job-name=DirectPred
#SBATCH --account=ec403
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=16G
#SBATCH --partition=accel
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --output=out_direct_pred_%j.log
#SBATCH --error=out_direct_pred_%j.log

# Default to not run setup unless specified
SETUP=${1:-0}

# Shift the first argument (SETUP) so we can pass remaining args
shift 2>/dev/null || true

# Print the current working directory
echo "Working directory: $(pwd)"

# Load required CUDA module for GPU tasks
module load CUDA/11.7.0

# Setup phase - only runs if SETUP=1
if [ "$SETUP" -eq 1 ]; then
    echo "Running setup and installation..."

    # Load modules
    module purge  # Clear any previously loaded modules
    module load Python/3.10.8-GCCcore-12.2.0
    module load Miniconda3/22.11.1-1

    # Setup miniconda
    mkdir -p miniconda3
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda3/miniconda.sh
    bash miniconda3/miniconda.sh -b -u -p miniconda3
    rm miniconda3/miniconda.sh

    source miniconda3/bin/activate
    conda init bash

    # Run installation script
    echo "Running installation script..."
    ./installation.sh

    echo "Setup completed"
fi

# Set environment variables for GPU
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

# Activate conda environment - this is needed regardless of setup
source miniconda3/etc/profile.d/conda.sh
conda activate hero

echo "Current environment:"
conda info --envs

echo "Running the direct prediction system with arguments: $@"
# Pass all remaining arguments to run_direct_prediction.sh
./run_direct_prediction.sh "$@"

echo "Job completed"
