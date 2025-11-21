#!/bin/bash

# Get current datetime in readable format (YYYY-MM-DD_HH-MM-SS)
DATETIME=$(date +"%Y-%m-%d_%H-%M-%S")
OUTPUT_DIR="logs/${DATETIME}"

# Run the Python script with current script usage as arguments
uv run -m dquantize.run_llada \
    --model "GSAI-ML/LLaDA-8B-Instruct" \
    --dataset "wikitext2" \
    --max_examples 128 \
    --min_length 32 \
    --max_length 128 \
    --output_dir "${OUTPUT_DIR}" \
    --steps 32 \
    --gen_length 128 \
    --block_length 32 \
    --temperature 0. \
    --cfg_scale 0. \
    --remasking "low_confidence" \
    --batch_size 8 | tee "${OUTPUT_DIR}/run_log.txt"

# Run the analysis script after generation completes
PROBS_FILE="${OUTPUT_DIR}/probs.jsonl"
if [ -f "${PROBS_FILE}" ]; then
    echo "Running analysis on ${PROBS_FILE}..."
    uv run -m analysis.prob_distr \
        --probs_file "${PROBS_FILE}" \
        --output_dir "${OUTPUT_DIR}/plots" | tee "${OUTPUT_DIR}/analysis_log.txt"
else
    echo "Warning: ${PROBS_FILE} not found. Skipping analysis."
fi

