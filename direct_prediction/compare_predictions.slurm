#!/bin/bash
#SBATCH --job-name=CompPred
#SBATCH --account=ec403
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=16G
#SBATCH --partition=accel
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --output=out_compare_pred_%j.log
#SBATCH --error=out_compare_pred_%j.log

source $(dirname "$0")/../config.sh

# Default settings
DIRECT_SYSTEM="direct_prediction"
BASELINE_SYSTEM="baseline"
SPLIT="dev"
OUTPUT_DIR="comparison_results"
USE_GPU=1
BATCH_SIZE=64
EMBEDDING_MODEL="all-MiniLM-L6-v2"

# Get the absolute path to the user's home directory version of the code
# The job was submitted from this directory
SUBMIT_DIR=$(pwd)
OUTPUT_DIR=${SUBMIT_DIR}

# Path to shared project directory
SHARED_PROJECT_PATH="/fp/projects01/ec403/IN5550_students/EivindogNora/FEVER-8-Shared-Task"


# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --direct=*)
      DIRECT_SYSTEM="${1#*=}"
      shift
      ;;
    --baseline=*)
      BASELINE_SYSTEM="${1#*=}"
      shift
      ;;
    --split=*)
      SPLIT="${1#*=}"
      shift
      ;;
    --output-dir=*)
      OUTPUT_DIR="${1#*=}"
      shift
      ;;
    --batch-size=*)
      BATCH_SIZE="${1#*=}"
      shift
      ;;
    --embedding-model=*)
      EMBEDDING_MODEL="${1#*=}"
      shift
      ;;
    *)
      echo "Unknown parameter: $1"
      echo "Usage: sbatch compare_predictions.slurm [--direct=SYSTEM] [--baseline=SYSTEM] [--split=SPLIT] [--output-dir=DIR] [--batch-size=SIZE] [--embedding-model=MODEL]"
      exit 1
      ;;
  esac
done

# Print the current working directory
echo "Working directory: $(pwd)"

# Load required CUDA module for GPU tasks
module load CUDA/11.7.0

# Set environment variables for GPU
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

# Activate conda environment
source "${SHARED_PROJECT_PATH}/miniconda3/etc/profile.d/conda.sh"
conda activate hero

# Setup data paths
DIRECT_FILE="${DATA_STORE}/${DIRECT_SYSTEM}/${SPLIT}_veracity_prediction.json"
BASELINE_FILE="${DATA_STORE}/${BASELINE_SYSTEM}/${SPLIT}_veracity_prediction.json"
GOLD_FILE="${DATA_STORE}/averitec/${SPLIT}.json"

# Check if files exist
if [ ! -f "$DIRECT_FILE" ]; then
  echo "Error: Direct prediction file not found at $DIRECT_FILE"
  exit 1
fi

if [ ! -f "$BASELINE_FILE" ]; then
  echo "Error: Baseline prediction file not found at $BASELINE_FILE"
  exit 1
fi

if [ ! -f "$GOLD_FILE" ]; then
  echo "Warning: Gold data file not found at $GOLD_FILE"
  echo "Continuing without correctness evaluation..."
  GOLD_ARG=""
else
  GOLD_ARG="--gold_file $GOLD_FILE"
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run the comparison script
echo "Running comparison analysis..."
echo "Direct system: $DIRECT_SYSTEM"
echo "Baseline system: $BASELINE_SYSTEM"
echo "Split: $SPLIT"
echo "Output directory: $OUTPUT_DIR"
echo "Batch size: $BATCH_SIZE"
echo "Embedding model: $EMBEDDING_MODEL"

python compare_predictions.py \
  --direct_file "$DIRECT_FILE" \
  --baseline_file "$BASELINE_FILE" \
  $GOLD_ARG \
  --output_dir "$OUTPUT_DIR" \
  --use_gpu \
  --batch_size $BATCH_SIZE \
  --embedding_model "$EMBEDDING_MODEL"

echo "Comparison analysis complete!"
echo "Results saved to $OUTPUT_DIR"