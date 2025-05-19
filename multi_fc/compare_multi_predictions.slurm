#!/bin/bash
#SBATCH --job-name=MultiCompPred
#SBATCH --account=ec403
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=16G
#SBATCH --partition=accel
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --output=out_multi_compare_pred_%j.log
#SBATCH --error=out_multi_compare_pred_%j.log

source $(pwd)/../config.sh

# Default settings
MULTI_SYSTEM="multi_perspective"
BASELINE_SYSTEM="baseline"
SPLIT="dev"
OUTPUT_DIR="multi_perspective_comparison_results"
USE_GPU=1
BATCH_SIZE=64
EMBEDDING_MODEL="all-MiniLM-L6-v2"
SKIP_JUSTIFICATION=0

# Get the absolute path to the user's home directory version of the code
# The job was submitted from this directory
SUBMIT_DIR=$(pwd)

# Path to shared project directory
SHARED_PROJECT_PATH="/fp/projects01/ec403/IN5550_students/EivindogNora/FEVER-8-Shared-Task"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --multi-system=*)
      MULTI_SYSTEM="${1#*=}"
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
    --skip-justification)
      SKIP_JUSTIFICATION=1
      shift
      ;;
    *)
      echo "Unknown parameter: $1"
      echo "Usage: sbatch multi_compare_predictions.slurm [--multi-system=SYSTEM] [--baseline=SYSTEM] [--split=SPLIT] [--output-dir=DIR] [--batch-size=SIZE] [--embedding-model=MODEL] [--skip-justification]"
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

# Setup data paths for multi-perspective files
POSITIVE_FILE="${DATA_STORE}/${MULTI_SYSTEM}/${SPLIT}_veracity_prediction_positive.json"
NEGATIVE_FILE="${DATA_STORE}/${MULTI_SYSTEM}/${SPLIT}_veracity_prediction_negative.json"
OBJECTIVE_FILE="${DATA_STORE}/${MULTI_SYSTEM}/${SPLIT}_veracity_prediction_objective.json"
BASELINE_FILE="${DATA_STORE}/${BASELINE_SYSTEM}/${SPLIT}_veracity_prediction.json"
GOLD_FILE="${DATA_STORE}/averitec/${SPLIT}.json"

# Check if perspective files exist
echo "Checking for perspective prediction files..."
for file_desc in "positive:$POSITIVE_FILE" "negative:$NEGATIVE_FILE" "objective:$OBJECTIVE_FILE"; do
  perspective=$(echo $file_desc | cut -d: -f1)
  file_path=$(echo $file_desc | cut -d: -f2)

  if [ ! -f "$file_path" ]; then
    echo "Error: $perspective perspective file not found at $file_path"
    exit 1
  else
    echo "Found $perspective file: $file_path"
  fi
done

# Check baseline file
if [ ! -f "$BASELINE_FILE" ]; then
  echo "Error: Baseline prediction file not found at $BASELINE_FILE"
  exit 1
else
  echo "Found baseline file: $BASELINE_FILE"
fi

# Check gold file
if [ ! -f "$GOLD_FILE" ]; then
  echo "Warning: Gold data file not found at $GOLD_FILE"
  echo "Continuing without correctness evaluation..."
  GOLD_ARG=""
else
  echo "Found gold file: $GOLD_FILE"
  GOLD_ARG="--gold_file $GOLD_FILE"
fi

# Create output directory with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
FULL_OUTPUT_DIR="${SUBMIT_DIR}/${OUTPUT_DIR}_${TIMESTAMP}"
mkdir -p "$FULL_OUTPUT_DIR"

# Prepare justification analysis argument
if [ $SKIP_JUSTIFICATION -eq 1 ]; then
  JUSTIFICATION_ARG="--skip_justification"
  echo "Skipping justification similarity analysis"
else
  JUSTIFICATION_ARG=""
  echo "Including justification similarity analysis"
fi

# Run the multi-perspective comparison script
echo "Running multi-perspective comparison analysis..."
echo "Multi-perspective system: $MULTI_SYSTEM"
echo "Baseline system: $BASELINE_SYSTEM"
echo "Split: $SPLIT"
echo "Output directory: $FULL_OUTPUT_DIR"
echo "Batch size: $BATCH_SIZE"
echo "Embedding model: $EMBEDDING_MODEL"
echo "GPU enabled: $USE_GPU"

python compare_multi_predictions.py \
  --positive_file "$POSITIVE_FILE" \
  --negative_file "$NEGATIVE_FILE" \
  --objective_file "$OBJECTIVE_FILE" \
  --baseline_file "$BASELINE_FILE" \
  $GOLD_ARG \
  --output_dir "$FULL_OUTPUT_DIR" \
  --use_gpu \
  --batch_size $BATCH_SIZE \
  --embedding_model "$EMBEDDING_MODEL" \
  $JUSTIFICATION_ARG

if [ $? -eq 0 ]; then
  echo "Multi-perspective comparison analysis complete!"
  echo "Results saved to $FULL_OUTPUT_DIR"

  # Print summary statistics
  echo ""
  echo "=== Quick Summary ==="
  if [ -f "$FULL_OUTPUT_DIR/multi_perspective_summary.md" ]; then
    echo "Summary report generated successfully"
    echo "Individual perspective reports available in:"
    echo "  - $FULL_OUTPUT_DIR/positive/"
    echo "  - $FULL_OUTPUT_DIR/negative/"
    echo "  - $FULL_OUTPUT_DIR/objective/"
  fi
else
  echo "Error: Multi-perspective comparison analysis failed!"
  exit 1
fi

# Optional: Show disk usage
echo ""
echo "Output directory size:"
du -sh "$FULL_OUTPUT_DIR"

echo "Job completed successfully!"