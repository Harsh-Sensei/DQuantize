#!/bin/bash

# Get current datetime in readable format (YYYY-MM-DD_HH-MM-SS)
DATETIME=$(date +"%Y-%m-%d_%H-%M-%S")
OUTPUT_DIR="logs/${DATETIME}"
mkdir -p "${OUTPUT_DIR}"
# 16 examples for testing, actual number of examples is 1319
# Run the Python script with current script usage as arguments
CUDA_VISIBLE_DEVICES=3 uv run -m dquantize.run_llada \
    --name "llada" \
    --dataset "gsm8k" \
    --max_examples 8 \
    --min_length 32 \
    --max_length 256 \
    --output_dir "${OUTPUT_DIR}" \
    --steps 64 \
    --gen_length 256 \
    --block_length 32 \
    --temperature 0. \
    --cfg_scale 0. \
    --remasking "low_confidence" \
    --batch_size 8 | tee "${OUTPUT_DIR}/run_log.txt"


