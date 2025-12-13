#!/bin/bash

# Get current datetime in readable format (YYYY-MM-DD_HH-MM-SS)
DATETIME=$(date +"%Y-%m-%d_%H-%M-%S")
OUTPUT_DIR="logs/all_strat_${DATETIME}"
mkdir -p "${OUTPUT_DIR}"

# Default quantized model path - update this to your actual path
QUANTIZED_MODEL_PATH="/home/scratch/hshah2/dquantize_cache/GSAI-ML/LLaDA-8B-Instruct-AWQ-w3-g128.pt"

# Run the Python script with dquantize arguments
CUDA_VISIBLE_DEVICES=0,1 uv run -m dquantize.run_dquantize_llada_all_strat \
    --model_name "GSAI-ML/LLaDA-8B-Instruct" \
    --quantized_model_path "${QUANTIZED_MODEL_PATH}" \
    --device_precise "cuda:0" \
    --device_quantized "cuda:1" \
    --q_backend "fake" \
    --dataset "gsm8k" \
    --max_examples 128 \
    --min_length 0 \
    --max_length 256 \
    --split "test" \
    --output_dir "${OUTPUT_DIR}" \
    --steps 64 \
    --gen_length 128 \
    --block_length 32 \
    --temperature 0. \
    --cfg_scale 0. \
    --batch_size 32 \
    --num_k_splits 4 
    
    # | tee "${OUTPUT_DIR}/run_log.txt"

