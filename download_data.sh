#!/bin/bash

# Source the configuration to get shared paths
source $(dirname "$0")/config.sh

# Create directories
mkdir -p "${DATA_STORE}/averitec"
mkdir -p "${KNOWLEDGE_STORE}/dev"
mkdir -p "${KNOWLEDGE_STORE}/test"

# Function to download a file if it doesn't exist
download_if_not_exists() {
    local url=$1
    local target=$2

    if [ ! -f "$target" ]; then
        echo "Downloading $(basename "$target")..."
        wget -O "$target" "$url"
        if [ $? -ne 0 ]; then
            echo "Error downloading $url"
            rm -f "$target"  # Remove partial file
            return 1
        fi
    else
        echo "$(basename "$target") already exists."
    fi
    return 0
}

# Function to extract a file if it doesn't exist
extract_if_not_exists() {
    local archive=$1
    local target_dir=$2

    # Check if the directory is empty
    if [ -z "$(ls -A "$target_dir" 2>/dev/null)" ]; then
        echo "Extracting $(basename "$archive") to $target_dir..."
        tar -xzf "$archive" -C "$target_dir" --strip-components=1
        if [ $? -ne 0 ]; then
            echo "Error extracting $archive"
            return 1
        fi
    else
        echo "$(basename "$target_dir") already has files."
    fi
    return 0
}

# Download data files
DEV_DATA_URL="https://github.com/Raldir/FEVER-8-Shared-Task/releases/download/v1.0/averitec_dev.json"
DEV_KNOWLEDGE_URL="https://github.com/Raldir/FEVER-8-Shared-Task/releases/download/v1.0/dev_knowledge_store.tar.gz"
TEST_KNOWLEDGE_URL="https://github.com/Raldir/FEVER-8-Shared-Task/releases/download/v1.0/test_knowledge_store.tar.gz"

# Download and extract dev files
download_if_not_exists "$DEV_DATA_URL" "${DATA_STORE}/averitec/dev.json" || exit 1

# Download knowledge store
download_if_not_exists "$DEV_KNOWLEDGE_URL" "${DATA_STORE}/dev_knowledge_store.tar.gz" || exit 1
download_if_not_exists "$TEST_KNOWLEDGE_URL" "${DATA_STORE}/test_knowledge_store.tar.gz" || exit 1

# Extract knowledge store
extract_if_not_exists "${DATA_STORE}/dev_knowledge_store.tar.gz" "${KNOWLEDGE_STORE}/dev" || exit 1
extract_if_not_exists "${DATA_STORE}/test_knowledge_store.tar.gz" "${KNOWLEDGE_STORE}/test" || exit 1

echo "Data download and extraction completed successfully."