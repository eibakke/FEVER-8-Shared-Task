#!/bin/bash

# Source the configuration file to get shared paths
source $(dirname "$0")/../config.sh

# Default settings
SYSTEM_NAME="multi_perspective"
SPLIT="dev"
FC_TYPE="positive"
CODE_PATH=$(dirname "$0")

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --system=*)
      SYSTEM_NAME="${1#*=}"
      shift
      ;;
    --split=*)
      SPLIT="${1#*=}"
      shift
      ;;
    --fc-type=*)
      FC_TYPE="${1#*=}"
      shift
      ;;
    *)
      echo "Unknown parameter: $1"
      echo "Usage: ./run_positive_prediction.sh [--system=NAME] [--split=SPLIT] [--base-dir=DIR] [--fc-type=TYPE]"
      exit 1
      ;;
  esac
done


echo "Running ${FC_TYPE} fact-checking veracity prediction experiment..."
echo "System: ${SYSTEM_NAME}, Split: ${SPLIT}"
echo "Data store: ${DATA_STORE}"

# Load API keys from .env
if [ -f "${CODE_PATH}/../.env" ]; then
    source "${CODE_PATH}/../.env"
else
    echo "Warning: .env file not found at ${CODE_PATH}/../.env"
    echo "Please create it with: echo 'export HUGGING_FACE_HUB_TOKEN=your_token' > .env"
fi

if [ -z "$HUGGING_FACE_HUB_TOKEN" ]; then
    echo "Error: HUGGING_FACE_HUB_TOKEN environment variable is not set"
    exit 1
fi

export HUGGING_FACE_HUB_TOKEN

if [ -z "$OPENAI_API_KEY" ]; then
    echo "Error: OPENAI_API_KEY environment variable is not set"
    exit 1
fi

export OPENAI_API_KEY

# Set model paths
HERO_MODEL="humane-lab/Meta-Llama-3.1-8B-HerO"

# Input and output file paths
INPUT_QA_FILE="${DATA_STORE}/${SYSTEM_NAME}/${SPLIT}_top_k_qa_${FC_TYPE}.json"
OUTPUT_PREDICTION_FILE="${DATA_STORE}/${SYSTEM_NAME}/${SPLIT}_veracity_prediction_${FC_TYPE}.json"
OUTPUT_SUBMISSION_FILE="leaderboard_submission/submission_${FC_TYPE}.csv"

# Check if input file exists
if [ ! -f "$INPUT_QA_FILE" ]; then
    echo "Error: Input QA file not found: $INPUT_QA_FILE"
    echo "Make sure you have run the multi-perspective pipeline up to the question generation step."
    exit 1
fi

echo "Input QA file: $INPUT_QA_FILE"
echo "Output prediction file: $OUTPUT_PREDICTION_FILE"

# Step 1: Run veracity prediction on positive QA output
echo "Step 1: Running veracity prediction on ${FC_TYPE} QA output..."
python "${CODE_PATH}/../baseline/veracity_prediction_optimized.py" \
    --target_data "$INPUT_QA_FILE" \
    --output_file "$OUTPUT_PREDICTION_FILE" \
    --batch_size 8 \
    --model "$HERO_MODEL" || exit 1

echo "Veracity prediction completed successfully!"

# Step 2: Prepare leaderboard submission
echo "Step 2: Preparing leaderboard submission..."
mkdir -p "leaderboard_submission"
python "${CODE_PATH}/../prepare_leaderboard_submission.py" \
    --filename "$OUTPUT_PREDICTION_FILE" || exit 1

echo "Leaderboard submission prepared: $OUTPUT_SUBMISSION_FILE"

# Step 3: Evaluate results
echo "Step 3: Evaluating ${FC_TYPE} results..."
python "${CODE_PATH}/../averitec_evaluate.py" \
    --prediction_file "${CODE_PATH}/leaderboard_submission/submission.csv" \
    --label_file "${CODE_PATH}/leaderboard_submission/solution_${SPLIT}.csv" || exit 1

echo ""
echo "=== ${FC_TYPE} Fact-checking Experiment Completed ==="
echo "Prediction file: $OUTPUT_PREDICTION_FILE"
echo "Submission file: $OUTPUT_SUBMISSION_FILE"
echo ""
