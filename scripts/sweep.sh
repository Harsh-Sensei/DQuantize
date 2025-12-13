#!/bin/bash

for steps in 8 16 32 64; do
    for block_length in 8 16 32 64; do
        for batch_size in 32 64; do
            output_dir="./outputs/steps${steps}_block${block_length}_batch${batch_size}"
            # sbatch launch_llada_inference.sh --steps $steps --block_length $block_length --batch_size $batch_size
            sbatch --job-name="s${steps}_b${block_length}_bs${batch_size}" \
            --output="./outputs/steps${steps}_block${block_length}_batch${batch_size}.out" \
            launch_llada_inference.sh \
            --output_dir="$output_dir" \
            --steps $steps \
            --block_length $block_length \
            --batch_size $batch_size
        done
    done
done