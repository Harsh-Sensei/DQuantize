#!/bin/bash

# Get current datetime in readable format (YYYY-MM-DD_HH-MM-SS)
DATETIME=$(date +"%Y-%m-%d_%H-%M-%S")
OUTPUT_DIR="logs/quantization_analysis_${DATETIME}"

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Run the quantization analysis script
echo "Starting LLaDA quantization analysis..."
echo "Output directory: ${OUTPUT_DIR}"
echo "=================================="

CUDA_VISIBLE_DEVICES=3,2 uv run -m dquantize.run_llada_quantization_analysis \
    --model "GSAI-ML/LLaDA-8B-Instruct" \
    --quantized_model "/home/scratch/hshah2/dquantize_cache/GSAI-ML/LLaDA-8B-Instruct-w4-g128.pt" \
    --dataset "wikitext2" \
    --max_examples 8 \
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