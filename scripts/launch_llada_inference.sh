#!/bin/bash
#SBATCH --job-name=gsm8ktest_vanilla_llada_128_steps      # Job name
#SBATCH --nodes=1                      # Number of nodes
#SBATCH --gres=gpu:1                    # GPUs per node (change if node has >1 GPU)
#SBATCH --cpus-per-task=8                # CPU cores per task
#SBATCH --mem=64G                        # Memory per node
#SBATCH --time=48:00:00                  # Walltime (hh:mm:ss)
#SBATCH --partition=general              # Partition/queue name
#SBATCH --output=logs/%x_%j.out   # Stdout log
#SBATCH --error=logs/%x_%j.err    # Stderr log

# Activate your conda environment
mkdir -p output
module load cuda-12.5
nvidia-smi

source /home/nidhih/miniconda3/etc/profile.d/conda.sh
conda activate colab

cd /home/nidhih/isgrl/dquantize/DQuantize/dquantize

python run_llada.py --dataset=gsm8k "$@" --max_examples 128 --max_length 256 

#!/bin/bash

