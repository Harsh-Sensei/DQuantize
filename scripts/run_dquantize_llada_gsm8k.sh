#!/bin/bash

# Get current datetime in readable format (YYYY-MM-DD_HH-MM-SS)
DATETIME=$(date +"%Y-%m-%d_%H-%M-%S")
OUTPUT_DIR="logs/${DATETIME}"
mkdir -p "${OUTPUT_DIR}"

# Default quantized model path - update this to your actual path
QUANTIZED_MODEL_PATH="/home/scratch/hshah2/dquantize_cache/GSAI-ML/LLaDA-8B-Instruct-w4-g128.pt"

# Run the Python script with dquantize arguments
CUDA_VISIBLE_DEVICES=3,2 uv run -m dquantize.run_dquantize_llada \
    --model_name "GSAI-ML/LLaDA-8B-Instruct" \
    --quantized_model_path "${QUANTIZED_MODEL_PATH}" \
    --strategy "firstk" \
    --k 10 \
    --device_precise "cuda:0" \
    --device_quantized "cuda:1" \
    --dataset "gsm8k" \
    --q_backend "fake" \
    --max_examples 8 \
    --min_length 32 \
    --max_length 256 \
    --split "test" \
    --output_dir "${OUTPUT_DIR}" \
    --steps 64 \
    --gen_length 256 \
    --block_length 32 \
    --temperature 0. \
    --cfg_scale 0. \
    --batch_size 8 | tee "${OUTPUT_DIR}/run_log.txt"

