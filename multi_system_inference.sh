#!/bin/bash

# Default settings (can be overridden via command-line arguments)
SYSTEM_NAME="multi_perspective"  # Change this to identify your system
SPLIT="dev"                      # Change this to "dev", or "test"
BASE_DIR="."                     # Current directory
NUM_EXAMPLES=0                   # Default: use full dataset (0 = full dataset)

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
      echo "Usage: ./multi_system_inference.sh [--num-examples=N] [--system=NAME] [--split=SPLIT] [--base-dir=DIR]"
      exit 1
      ;;
  esac
done

# Setup data paths
DATA_STORE="${BASE_DIR}/data_store"
KNOWLEDGE_STORE="${BASE_DIR}/knowledge_store"
export HF_HOME="${BASE_DIR}/huggingface_cache"

# Create necessary directories
mkdir -p "${DATA_STORE}/averitec"
mkdir -p "${DATA_STORE}/${SYSTEM_NAME}"
mkdir -p "${HF_HOME}"

# If num_examples is set, create a smaller dataset
if [ $NUM_EXAMPLES -gt 0 ]; then
    echo "Creating smaller dataset with $NUM_EXAMPLES examples..."
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
        BATCH_SIZE=4
        RERANKING_BATCH_SIZE=16
        QUESTION_GEN_BATCH_SIZE=1
        VERACITY_BATCH_SIZE=1
    elif [ $NUM_EXAMPLES -le 50 ]; then
        BATCH_SIZE=8
        RERANKING_BATCH_SIZE=32
        QUESTION_GEN_BATCH_SIZE=1
        VERACITY_BATCH_SIZE=1
    elif [ $NUM_EXAMPLES -le 100 ]; then
        BATCH_SIZE=16
        RERANKING_BATCH_SIZE=64
        QUESTION_GEN_BATCH_SIZE=2
        VERACITY_BATCH_SIZE=2
    else
        BATCH_SIZE=32
        RERANKING_BATCH_SIZE=128
        QUESTION_GEN_BATCH_SIZE=4
        VERACITY_BATCH_SIZE=4
    fi
else
    # Default batch sizes for full dataset
    BATCH_SIZE=32
    RERANKING_BATCH_SIZE=128
    QUESTION_GEN_BATCH_SIZE=4
    VERACITY_BATCH_SIZE=8
    mkdir -p "${KNOWLEDGE_STORE}/${SPLIT}"
fi

echo "Starting multi-perspective system inference for ${SYSTEM_NAME} on ${SPLIT} split..."
echo "Data store: ${DATA_STORE}"
echo "Knowledge store: ${KNOWLEDGE_STORE}"
echo "Batch sizes - Main: ${BATCH_SIZE}, Reranking: ${RERANKING_BATCH_SIZE}, Question Gen: ${QUESTION_GEN_BATCH_SIZE}, Veracity: ${VERACITY_BATCH_SIZE}"

# Load config file with tokens
if [ -f ".env" ]; then
    source .env
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
echo "Step 1: Generating multi-type hypothetical fact-checking documents..."
python multi_fc/multi_hyde_fc_generation.py \
    --target_data "${DATA_STORE}/averitec/${SPLIT}.json" \
    --json_output "${DATA_STORE}/${SYSTEM_NAME}/${SPLIT}_multi_hyde_fc.json" \
    --model "$MODEL_PATH" \
    --batch_size $BATCH_SIZE || exit 1

# Step 2: Extract each type of fact-checking document
echo "Step 2: Extracting different types of fact-checking documents..."
python multi_fc/extract_fc_types.py \
    --input_file "${DATA_STORE}/${SYSTEM_NAME}/${SPLIT}_multi_hyde_fc.json" \
    --output_prefix "${DATA_STORE}/${SYSTEM_NAME}/${SPLIT}_hyde_fc" \
    --types "${FC_TYPES[@]}" || exit 1

# Arrays to collect QA outputs for merging
QA_OUTPUTS=()

# Process each type of fact-checking document
for fc_type in "${FC_TYPES[@]}"; do
    echo "Processing ${fc_type} fact-checking perspective..."

    # Step 3: Run retrieval for this type
    echo "Step 3a: Running retrieval for ${fc_type} fact-checking..."
    python baseline/retrieval_optimized.py \
        --knowledge_store_dir "${KNOWLEDGE_STORE}/${SPLIT}" \
        --target_data "${DATA_STORE}/${SYSTEM_NAME}/${SPLIT}_hyde_fc_${fc_type}.json" \
        --json_output "${DATA_STORE}/${SYSTEM_NAME}/${SPLIT}_retrieval_top_k_${fc_type}.json" \
        --top_k 5000 || exit 1

    # Step 4: Run reranking for this type
    echo "Step 3b: Running reranking for ${fc_type} fact-checking..."
    python baseline/reranking_optimized.py \
        --target_data "${DATA_STORE}/${SYSTEM_NAME}/${SPLIT}_retrieval_top_k_${fc_type}.json" \
        --json_output "${DATA_STORE}/${SYSTEM_NAME}/${SPLIT}_reranking_top_k_${fc_type}.json" \
        --retrieved_top_k 500 --batch_size $RERANKING_BATCH_SIZE || exit 1

    # Step 5: Generate questions for this type
    echo "Step 3c: Generating questions for ${fc_type} fact-checking..."
    python baseline/question_generation_optimized.py \
        --reference_corpus "${DATA_STORE}/averitec/${SPLIT}.json" \
        --top_k_target_knowledge "${DATA_STORE}/${SYSTEM_NAME}/${SPLIT}_reranking_top_k_${fc_type}.json" \
        --output_questions "${DATA_STORE}/${SYSTEM_NAME}/${SPLIT}_top_k_qa_${fc_type}.json" \
        --model "$MODEL_PATH" \
        --batch_size $QUESTION_GEN_BATCH_SIZE || exit 1

    # Add this output to the list of QA outputs
    QA_OUTPUTS+=("${DATA_STORE}/${SYSTEM_NAME}/${SPLIT}_top_k_qa_${fc_type}.json")
done

# Step 6: Merge question-answer pairs from all types
echo "Step 4: Merging question-answer pairs from all perspectives..."
python multi_fc/merge_qa.py \
    --qa_files "${QA_OUTPUTS[@]}" \
    --output_file "${DATA_STORE}/${SYSTEM_NAME}/${SPLIT}_merged_qa.json" \
    --types "${FC_TYPES[@]}" || exit 1

# Step 7: Run veracity prediction with merged question-answer pairs
echo "Step 5: Running veracity prediction with merged question-answer pairs..."
python multi_fc/multi_veracity_prediction.py \
    --target_data "${DATA_STORE}/${SYSTEM_NAME}/${SPLIT}_merged_qa.json" \
    --output_file "${DATA_STORE}/${SYSTEM_NAME}/${SPLIT}_veracity_prediction.json" \
    --batch_size $VERACITY_BATCH_SIZE \
    --model "$HERO_MODEL" || exit 1

# Step 8: Prepare leaderboard submission
echo "Step 6: Preparing leaderboard submission..."
python prepare_leaderboard_submission.py \
    --filename "${DATA_STORE}/${SYSTEM_NAME}/${SPLIT}_veracity_prediction.json" || exit 1

# Step 9: Evaluate results
echo "Step 7: Evaluating results..."
python baseline/averitec_evaluate_legacy.py \
    --prediction_file "${DATA_STORE}/${SYSTEM_NAME}/${SPLIT}_veracity_prediction.json" \
    --label_file "${DATA_STORE}/averitec/${SPLIT}.json" || exit 1

echo "All steps completed successfully!"
echo "To analyze the results, run: python analyze_pipeline.py --system $SYSTEM_NAME --split $SPLIT --summary"
