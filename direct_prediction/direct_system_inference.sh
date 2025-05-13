#!/bin/bash

# Source the configuration file to get shared paths
source $(dirname "$0")/config.sh

# Default settings (can be overridden via command-line arguments)
SYSTEM_NAME="direct_prediction"
SPLIT="dev"
BASE_DIR="."
NUM_EXAMPLES=0
MODEL_PATH="/fp/projects01/ec403/hf_models/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --num-examples=*)
      NUM_EXAMPLES="${1#*=}"
      shift
      ;;
    --system=*)
      SYSTEM_NAME="${1#*=}"
      shift
      ;;
    --split=*)
      SPLIT="${1#*=}"
      shift
      ;;
    --base-dir=*)
      BASE_DIR="${1#*=}"
      shift
      ;;
    *)
      echo "Unknown parameter: $1"
      echo "Usage: ./run_direct_prediction.sh [--num-examples=N] [--system=NAME] [--split=SPLIT] [--base-dir=DIR]"
      exit 1
      ;;
  esac
done

# Setup code paths (local)
CODE_PATH=$(dirname "$0")


# If num_examples is set, create a smaller dataset
if [ $NUM_EXAMPLES -gt 0 ]; then
    echo "Creating smaller dataset with $NUM_EXAMPLES examples..."
    ORIG_SPLIT=$SPLIT
    SPLIT="${SPLIT}_small_${NUM_EXAMPLES}"

    # Check if jq is installed
    if ! command -v jq &> /dev/null; then
        echo "Error: jq is not installed. Please install it first."
        exit 1
    fi

    # Extract subset of the original dataset
    if [ ! -f "${DATA_STORE}/averitec/${SPLIT}.json" ]; then
        echo "Extracting ${NUM_EXAMPLES} examples from ${ORIG_SPLIT} dataset..."
        jq ".[0:${NUM_EXAMPLES}]" "${DATA_STORE}/averitec/${ORIG_SPLIT}.json" > "${DATA_STORE}/averitec/${SPLIT}.json"
    else
        echo "Using existing small dataset at ${DATA_STORE}/averitec/${SPLIT}.json"
    fi

    # Adjust batch size based on dataset size
    if [ $NUM_EXAMPLES -le 50 ]; then
        BATCH_SIZE=4
    else
        BATCH_SIZE=8
    fi
else
    # Default batch size for full dataset
    BATCH_SIZE=8
fi

echo "Starting direct prediction system for ${SYSTEM_NAME} on ${SPLIT} split..."
echo "Data store: ${DATA_STORE}"
echo "Batch size: ${BATCH_SIZE}"

# Load config file with tokens
if [ -f ".env" ]; then
    source "${CODE_PATH}/../.env"
else
    echo "Error: .env file not found"
    echo "Please create it with: echo 'export HUGGING_FACE_HUB_TOKEN=your_token' > .env"
    exit 1
fi

if [ -z "$HUGGING_FACE_HUB_TOKEN" ]; then
    echo "Error: HUGGING_FACE_HUB_TOKEN environment variable is not set"
    echo "Please set it by running: export HUGGING_FACE_HUB_TOKEN=your_token"
    exit 1
fi

export HUGGING_FACE_HUB_TOKEN

# Run direct prediction
echo "Running direct veracity prediction..."
python "${CODE_PATH}/direct_prediction.py" \
    --target_data "${DATA_STORE}/averitec/${SPLIT}.json" \
    --output_file "${DATA_STORE}/${SYSTEM_NAME}/${SPLIT}_veracity_prediction.json" \
    --batch_size $BATCH_SIZE \
    --model "$MODEL_PATH" || exit 1

# Run evaluation
echo "Preparing leaderboard submission..."
python "${CODE_PATH}/../prepare_leaderboard_submission.py" \
    --filename "${DATA_STORE}/${SYSTEM_NAME}/${SPLIT}_veracity_prediction.json" || exit 1

echo "Evaluating results..."
python "${CODE_PATH}/../averitec_evaluate.py" \
    --prediction_file "${CODE_PATH}/../leaderboard_submission/submission.csv" \
    --label_file "${CODE_PATH}/../leaderboard_submission/solution_${SPLIT}.csv" || exit 1

echo "All steps completed successfully!"
echo "Results saved to: ${DATA_STORE}/${SYSTEM_NAME}/${SPLIT}_veracity_prediction.json"
