#!/bin/bash

# Initialize conda for script usage
source miniconda3/etc/profile.d/conda.sh


# Create and activate environment
conda create -n hero python=3.12 -y  # Added -y for non-interactive
sleep 2  # Increased sleep time for readability

conda activate hero
sleep 2

conda info --envs
sleep 2

which python
which pip

# Install packages
python -m pip install torch
python -m pip install -r requirements.txt
MAX_JOBS=8 python -m pip -v install flash-attn --no-build-isolation
python -m pip install https://vllm-wheels.s3.us-west-2.amazonaws.com/nightly/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl
python -m pip install transformers --upgrade
