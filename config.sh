#!/bin/bash

# Path to the shared project space
export SHARED_PROJECT_PATH="/fp/projects01/ec403/IN5550_students/EivindogNora/FEVER-8-Shared-Task"

# Paths to shared resources
export DATA_STORE="${SHARED_PROJECT_PATH}/data_store"
export KNOWLEDGE_STORE="${SHARED_PROJECT_PATH}/knowledge_store"
export HF_HOME="${SHARED_PROJECT_PATH}/huggingface_cache"
export NLTK_DATA="${SHARED_PROJECT_PATH}/nltk_data"
export CONDA_ENV_PATH="${SHARED_PROJECT_PATH}/miniconda3"

# Set up NLTK data path
export NLTK_DATA_PATH="${NLTK_DATA}"

# Include your personal .env file for API keys
if [ -f "$HOME/FEVER-8-Project/.env" ]; then
    source "$HOME/FEVER-8-Project/.env"
fi
