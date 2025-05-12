#!/bin/bash

# Source the configuration file to get shared paths
source $(dirname "$0")/../config.sh

# Default settings (can be overridden via command-line arguments)
SYSTEM_NAME="new_order"  # Change this to "HerO", "Baseline", etc.
SPLIT="dev"             # Change this to "dev", or "test"
BASE_DIR="."            # Current directory
NUM_EXAMPLES=0          # Default: use full dataset (0 = full dataset)
RESUME_STEP=1           # Default: start from the beginning

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
    --resume-step=*)
      RESUME_STEP="${1#*=}"
      shift
      ;;
    *)
      echo "Unknown parameter: $1"
      echo "Usage: ./system_inference.sh [--num-examples=N] [--system=NAME] [--split=SPLIT] [--base-dir=DIR] [--resume-step=STEP]"
      exit 1
      ;;
  esac
done

# Setup code paths (local)
CODE_PATH=$(dirname "$0")

# If num_examples is set, create a smaller dataset
if [ $NUM_EXAMPLES -gt 0 ]; then
    echo "Processing smaller dataset with $NUM_EXAMPLES examples..."
    ORIG_SPLIT=$SPLIT
    SPLIT="${SPLIT}_small_${NUM_EXAMPLES}"

    # Create the necessary directories
    mkdir -p "${KNOWLEDGE_STORE}/${SPLIT}"

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

    # Copy knowledge store if it doesn't exist yet
    if [ ! -d "${KNOWLEDGE_STORE}/${SPLIT}" ] || [ -z "$(ls -A ${KNOWLEDGE_STORE}/${SPLIT})" ]; then
        echo "Copying knowledge store from ${ORIG_SPLIT} to ${SPLIT}..."
        cp -r "${KNOWLEDGE_STORE}/${ORIG_SPLIT}/." "${KNOWLEDGE_STORE}/${SPLIT}/"
    else
        echo "Using existing knowledge store at ${KNOWLEDGE_STORE}/${SPLIT}"
    fi

    # Adjust batch sizes based on dataset size
    if [ $NUM_EXAMPLES -le 10 ]; then
        RERANKING_BATCH_SIZE=16
        QUESTION_GEN_BATCH_SIZE=1
        VERACITY_BATCH_SIZE=1
    elif [ $NUM_EXAMPLES -le 50 ]; then
        RERANKING_BATCH_SIZE=32
        QUESTION_GEN_BATCH_SIZE=1
        VERACITY_BATCH_SIZE=1
    elif [ $NUM_EXAMPLES -le 100 ]; then
        RERANKING_BATCH_SIZE=64
        QUESTION_GEN_BATCH_SIZE=2
        VERACITY_BATCH_SIZE=2
    else
        RERANKING_BATCH_SIZE=128
        QUESTION_GEN_BATCH_SIZE=4
        VERACITY_BATCH_SIZE=4
    fi
else
    # Default batch sizes for full dataset
    RERANKING_BATCH_SIZE=64  # Reduced from 128 to 64
    QUESTION_GEN_BATCH_SIZE=4
    VERACITY_BATCH_SIZE=8
    mkdir -p "${KNOWLEDGE_STORE}/${SPLIT}"
fi

echo "Starting system inference for ${SYSTEM_NAME} on ${SPLIT} split..."
echo "Data store: ${DATA_STORE}"
echo "Knowledge store: ${KNOWLEDGE_STORE}"
echo "Code path: ${CODE_PATH}"
echo "Batch sizes - Reranking: ${RERANKING_BATCH_SIZE}, Question Gen: ${QUESTION_GEN_BATCH_SIZE}, Veracity: ${VERACITY_BATCH_SIZE}"
echo "Starting from step ${RESUME_STEP}"

# Load API keys from .env file
if [ -f "${CODE_PATH}/../.env" ]; then
    source "${CODE_PATH}/../.env"
else
    echo "Warning: .env file not found at ${CODE_PATH}/../.env"
    echo "Please create it with: echo 'export HUGGING_FACE_HUB_TOKEN=your_token' > .env"
    echo "Note: This may not be a problem if you're sourcing the environment variables from another location."
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

# Set model path
MODEL_PATH="/fp/projects01/ec403/hf_models/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"

# Execute each script based on resume step
if [ $RESUME_STEP -le 1 ]; then
    echo "Step 1: Generating hypothetical fact-checking documents..."
    python "${CODE_PATH}/fc_and_question_generator.py" \
        --target_data "${DATA_STORE}/averitec/${SPLIT}.json" \
        --json_output "${DATA_STORE}/${SYSTEM_NAME}/${SPLIT}_hyde_fc.json" \
        --model "$MODEL_PATH" || exit 1
fi

if [ $RESUME_STEP -le 2 ]; then
    echo "Step 2: Running retrieval..."
    python "${CODE_PATH}/retrival_optimized_new_question_order.py" \
        --knowledge_store_dir "${KNOWLEDGE_STORE}/${SPLIT}" \
        --retrieval_method "bm25_precomputed" \
        --precomputed_bm25_dir "${KNOWLEDGE_STORE}/${SPLIT}/precomputed_bm25" \
        --target_data "${DATA_STORE}/${SYSTEM_NAME}/${SPLIT}_hyde_fc.json" \
        --json_output "${DATA_STORE}/${SYSTEM_NAME}/${SPLIT}_retrieval_top_k.json" \
        --top_k 5000 || exit 1
fi

if [ $RESUME_STEP -le 3 ]; then
    echo "Step 3: Running reranking..."
    python "${CODE_PATH}/reranking_optimized_new_question_order.py" \
        --target_data "${DATA_STORE}/${SYSTEM_NAME}/${SPLIT}_retrieval_top_k.json" \
        --json_output "${DATA_STORE}/${SYSTEM_NAME}/${SPLIT}_reranking_top_k.json" \
        --retrieved_top_k 500 --batch_size $RERANKING_BATCH_SIZE || exit 1
fi

if [ $RESUME_STEP -le 4 ]; then
    echo "Step 4: Generating questions..."
    python "${CODE_PATH}/../baseline/question_generation_optimized.py" \
        --reference_corpus "${DATA_STORE}/averitec/${SPLIT}.json" \
        --top_k_target_knowledge "${DATA_STORE}/${SYSTEM_NAME}/${SPLIT}_reranking_top_k.json" \
        --output_questions "${DATA_STORE}/${SYSTEM_NAME}/${SPLIT}_top_k_qa.json" \
        --model "$MODEL_PATH" \
        --batch_size $QUESTION_GEN_BATCH_SIZE || exit 1
fi

if [ $RESUME_STEP -le 5 ]; then
    echo "Step 5: Running veracity prediction..."
    python "${CODE_PATH}/../baseline/veracity_prediction_optimized.py" \
        --target_data "${DATA_STORE}/${SYSTEM_NAME}/${SPLIT}_top_k_qa.json" \
        --output_file "${DATA_STORE}/${SYSTEM_NAME}/${SPLIT}_veracity_prediction.json" \
        --batch_size $VERACITY_BATCH_SIZE \
        --model "humane-lab/Meta-Llama-3.1-8B-HerO" || exit 1
fi

if [ $RESUME_STEP -le 6 ]; then
    echo "Step 6: Preparing leaderboard submission..."
    mkdir -p "${CODE_PATH}/leaderboard_submission"
    python "${CODE_PATH}/prepare_leaderboard_submission.py" \
        --filename "${DATA_STORE}/${SYSTEM_NAME}/${SPLIT}_veracity_prediction.json"  || exit 1
fi

if [ $RESUME_STEP -le 7 ]; then
    echo "Step 7: Evaluating results..."
    python "${CODE_PATH}/local_averitec_evaluate.py" \
        --prediction_file "${CODE_PATH}/../leaderboard_submission/submission.csv" \
        --label_file "${CODE_PATH}/../leaderboard_submission/solution_dev.csv" || exit 1
fi

echo "All steps completed successfully!"
echo "To analyze the results, run: python ${CODE_PATH}/.../analyze_pipeline.py --system $SYSTEM_NAME --split $SPLIT --summary"
