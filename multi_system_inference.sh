#!/bin/bash

# Source the configuration file to get shared paths
source $(dirname "$0")/config.sh

# Default settings (can be overridden via command-line arguments)
SYSTEM_NAME="multi_perspective"  # Change this to identify your system
SPLIT="dev"                      # Change this to "dev", or "test"
BASE_DIR="."                     # Current directory
NUM_EXAMPLES=0                   # Default: use full dataset (0 = full dataset)
SKIP_STEPS=""                    # Steps to skip (comma-separated)
FORCE_STEPS=""                   # Steps to force run even if output exists (comma-separated)
START_FROM=1                     # Start from this step (default: from beginning)

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
    --skip-steps=*)
      SKIP_STEPS="${1#*=}"
      shift
      ;;
    --force-steps=*)
      FORCE_STEPS="${1#*=}"
      shift
      ;;
    --start-from=*)
      START_FROM="${1#*=}"
      shift
      ;;
    *)
      echo "Unknown parameter: $1"
      echo "Usage: ./multi_system_inference.sh [--num-examples=N] [--system=NAME] [--split=SPLIT] [--base-dir=DIR] [--skip-steps=1,2,3] [--force-steps=4,5] [--start-from=N]"
      exit 1
      ;;
  esac
done

# Get code path (local)
CODE_PATH=$(dirname "$0")

# Create necessary directories
mkdir -p "${DATA_STORE}/averitec"
mkdir -p "${DATA_STORE}/${SYSTEM_NAME}"
mkdir -p "${HF_HOME}"

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

    KNOWLEDGE_STORE_PATH="${KNOWLEDGE_STORE}/${ORIG_SPLIT}"

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
    RERANKING_BATCH_SIZE=128
    QUESTION_GEN_BATCH_SIZE=4
    VERACITY_BATCH_SIZE=8
    KNOWLEDGE_STORE_PATH="${KNOWLEDGE_STORE}/${SPLIT}"
fi

echo "Starting multi-perspective system inference for ${SYSTEM_NAME} on ${SPLIT} split..."
echo "Data store: ${DATA_STORE}"
echo "Knowledge store: ${KNOWLEDGE_STORE_PATH}"
echo "Code path: ${CODE_PATH}"
echo "Batch sizes - Reranking: ${RERANKING_BATCH_SIZE}, Question Gen: ${QUESTION_GEN_BATCH_SIZE}, Veracity: ${VERACITY_BATCH_SIZE}"
echo "Starting from step ${START_FROM}"
if [ ! -z "$SKIP_STEPS" ]; then
    echo "Skipping steps: ${SKIP_STEPS}"
fi
if [ ! -z "$FORCE_STEPS" ]; then
    echo "Forcing steps: ${FORCE_STEPS}"
fi

# Function to check if a step should be run
should_run_step() {
    step_num=$1
    output_file=$2

    # Check if step is before START_FROM
    if [ $step_num -lt $START_FROM ]; then
        echo "Skipping step $step_num (starting from step $START_FROM)"
        return 1
    fi

    # Check if step is in SKIP_STEPS
    if [[ $SKIP_STEPS == *"$step_num"* || $SKIP_STEPS == *"all"* ]]; then
        echo "Skipping step $step_num (explicitly skipped)"
        return 1
    fi

    # Check if step is in FORCE_STEPS
    if [[ $FORCE_STEPS == *"$step_num"* || $FORCE_STEPS == *"all"* ]]; then
        echo "Running step $step_num (explicitly forced)"
        return 0
    fi

    # Check if output file exists
    if [ -f "$output_file" ]; then
        echo "Skipping step $step_num (output file exists: $output_file)"
        return 1
    fi

    # Run the step
    return 0
}

# Load API keys from .env
if [ -f "${CODE_PATH}/.env" ]; then
    source "${CODE_PATH}/.env"
else
    echo "Warning: .env file not found at ${CODE_PATH}/.env"
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

# Set model path
MODEL_PATH="/fp/projects01/ec403/hf_models/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
HERO_MODEL="humane-lab/Meta-Llama-3.1-8B-HerO"

# Define FC types
FC_TYPES=("positive" "negative" "objective")

# Step 1: Generate multi-type hypothetical fact-checking documents
STEP1_OUTPUT="${DATA_STORE}/${SYSTEM_NAME}/${SPLIT}_multi_hyde_fc.json"
if should_run_step 1 "$STEP1_OUTPUT"; then
    echo "Step 1: Generating multi-type hypothetical fact-checking documents..."
    python "${CODE_PATH}/multi_fc/multi_hyde_fc_generation.py" \
        --target_data "${DATA_STORE}/averitec/${SPLIT}.json" \
        --json_output "$STEP1_OUTPUT" \
        --model "$MODEL_PATH" || exit 1
fi

# Step 2: Extract each type of fact-checking document
STEP2_OUTPUT_PREFIX="${DATA_STORE}/${SYSTEM_NAME}/${SPLIT}_hyde_fc"
STEP2_OUTPUT="${STEP2_OUTPUT_PREFIX}_positive.json" # Check just one of the outputs
if should_run_step 2 "$STEP2_OUTPUT"; then
    echo "Step 2: Extracting different types of fact-checking documents..."
    python "${CODE_PATH}/multi_fc/extract_fc_types.py" \
        --input_file "$STEP1_OUTPUT" \
        --output_prefix "$STEP2_OUTPUT_PREFIX" \
        --types "${FC_TYPES[@]}" || exit 1
fi

# Arrays to collect QA outputs for merging
QA_OUTPUTS=()

# Process each type of fact-checking document
for fc_type in "${FC_TYPES[@]}"; do
    echo "Processing ${fc_type} fact-checking perspective..."

    # Step 3a: Run retrieval for this type
    STEP3A_OUTPUT="${DATA_STORE}/${SYSTEM_NAME}/${SPLIT}_retrieval_top_k_${fc_type}.json"
    if should_run_step "3a-${fc_type}" "$STEP3A_OUTPUT"; then
        echo "Step 3a: Running retrieval for ${fc_type} fact-checking..."
        python "${CODE_PATH}/baseline/retrieval_optimized.py" \
            --knowledge_store_dir "${KNOWLEDGE_STORE_PATH}" \
            --retrieval_method "bm25_precomputed" \
            --precomputed_bm25_dir "${KNOWLEDGE_STORE_PATH}/precomputed_bm25" \
            --target_data "${DATA_STORE}/${SYSTEM_NAME}/${SPLIT}_hyde_fc_${fc_type}.json" \
            --json_output "$STEP3A_OUTPUT" \
            --top_k 5000 || exit 1
    fi

    # Step 3b: Run reranking for this type
    STEP3B_OUTPUT="${DATA_STORE}/${SYSTEM_NAME}/${SPLIT}_reranking_top_k_${fc_type}.json"
    if should_run_step "3b-${fc_type}" "$STEP3B_OUTPUT"; then
        echo "Step 3b: Running reranking for ${fc_type} fact-checking..."
        python "${CODE_PATH}/baseline/reranking_optimized.py" \
            --target_data "$STEP3A_OUTPUT" \
            --json_output "$STEP3B_OUTPUT" \
            --retrieved_top_k 500 --batch_size $RERANKING_BATCH_SIZE || exit 1
    fi

    # Step 3c: Generate questions for this type
    STEP3C_OUTPUT="${DATA_STORE}/${SYSTEM_NAME}/${SPLIT}_top_k_qa_${fc_type}.json"
    if should_run_step "3c-${fc_type}" "$STEP3C_OUTPUT"; then
        echo "Step 3c: Generating questions for ${fc_type} fact-checking..."
        python "${CODE_PATH}/baseline/question_generation_optimized.py" \
            --reference_corpus "${DATA_STORE}/averitec/train_reference.json" \
            --top_k_target_knowledge "$STEP3B_OUTPUT" \
            --output_questions "$STEP3C_OUTPUT" \
            --model "$MODEL_PATH" \
            --batch_size $QUESTION_GEN_BATCH_SIZE || exit 1
    fi

    # Add this output to the list of QA outputs
    QA_OUTPUTS+=("$STEP3C_OUTPUT")
done

# Step 4: Merge question-answer pairs from all types
STEP4_OUTPUT="${DATA_STORE}/${SYSTEM_NAME}/${SPLIT}_merged_qa.json"
if should_run_step 4 "$STEP4_OUTPUT"; then
    echo "Step 4: Merging question-answer pairs from all perspectives..."
    python "${CODE_PATH}/multi_fc/merge_qa.py" \
        --qa_files "${QA_OUTPUTS[@]}" \
        --output_file "$STEP4_OUTPUT" \
        --types "${FC_TYPES[@]}" || exit 1
fi

# Step 5: Run veracity prediction with merged question-answer pairs
STEP5_OUTPUT="${DATA_STORE}/${SYSTEM_NAME}/${SPLIT}_veracity_prediction.json"
if should_run_step 5 "$STEP5_OUTPUT"; then
    echo "Step 5: Running veracity prediction with merged question-answer pairs..."
    python "${CODE_PATH}/multi_fc/multi_veracity_prediction.py" \
        --target_data "$STEP4_OUTPUT" \
        --output_file "$STEP5_OUTPUT" \
        --batch_size $VERACITY_BATCH_SIZE \
        --model "$HERO_MODEL" || exit 1
fi

# Step 6: Prepare leaderboard submission
STEP6_OUTPUT="${CODE_PATH}/leaderboard_submission/submission.csv"
mkdir -p "${CODE_PATH}/leaderboard_submission"
if should_run_step 6 "$STEP6_OUTPUT"; then
    echo "Step 6: Preparing leaderboard submission..."
    python "${CODE_PATH}/prepare_leaderboard_submission.py" \
        --filename "$STEP5_OUTPUT" \
        --output_dir "${CODE_PATH}/leaderboard_submission" || exit 1
fi

# Step 7: Evaluate results
if should_run_step 7 ""; then  # No specific output file for evaluation
    echo "Step 7: Evaluating results..."
    python "${CODE_PATH}/averitec_evaluate.py" \
        --prediction_file "$STEP6_OUTPUT" \
        --label_file "${CODE_PATH}/leaderboard_submission/solution_${SPLIT}.csv" || exit 1
fi

echo "All steps completed successfully!"
echo "To analyze the results, run: python ${CODE_PATH}/analyze_pipeline.py --system $SYSTEM_NAME --split $SPLIT --summary"