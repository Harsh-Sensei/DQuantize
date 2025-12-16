#!/bin/bash

# Get current datetime in readable format (YYYY-MM-DD_HH-MM-SS)
DATETIME=$(date +"%Y-%m-%d_%H-%M-%S")
OUTPUT_DIR="logs/quantization_analysis_${DATETIME}"

# Create output directory
mkdir -p "${OUTPUT_DIR}"
QUANTIZED_MODEL_PATH="/home/scratch/hshah2/dquantize_cache/GSAI-ML/LLaDA-8B-Instruct-AWQ-w4-g128.pt"
# Run the quantization analysis script
echo "Starting LLaDA quantization analysis..."
echo "Output directory: ${OUTPUT_DIR}"
echo "=================================="

CUDA_VISIBLE_DEVICES=2,3 uv run -m dquantize.run_llada_quantization_analysis \
    --model "GSAI-ML/LLaDA-8B-Instruct" \
    --quantized_model "${QUANTIZED_MODEL_PATH}" \
    --dataset "wikitext2" \
    --max_examples 32 \
    --min_length 32 \
    --max_length 256 \
    --output_dir "${OUTPUT_DIR}" \
    --steps 64 \
    --gen_length 128 \
    --block_length 32 \
    --temperature 0. \
    --cfg_scale 0. \
    --remasking "low_confidence" \
    --q_backend "fake" \
    --batch_size 32 | tee "${OUTPUT_DIR}/run_log.txt"

echo "=================================="
echo "Analysis complete!"
echo "Results saved to: ${OUTPUT_DIR}"
echo ""
echo "Output files:"
echo "  - ${OUTPUT_DIR}/quantization_analysis.jsonl  (detailed KL divergence and Jaccard similarity per step)"
echo "  - ${OUTPUT_DIR}/analysis_summary.json        (summary statistics)"
echo "  - ${OUTPUT_DIR}/timing_stats.json            (timing breakdown)"
echo "  - ${OUTPUT_DIR}/probs.jsonl                  (probability logs from precise model)"
echo "  - ${OUTPUT_DIR}/call_config.yaml             (configuration used)"