# Quantize the model using AWQ
export HF_DATASETS_TRUST_REMOTE_CODE=true
export HF_ALLOW_CODE_EVAL=1

DIRPATH="$(cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1 && pwd)"
MODEL_PATH='GSAI-ML/LLaDA-8B-Instruct'  # Replace with your model path
CACHE_PATH='/home/scratch/hshah2/dquantize_cache'

# Check if W_BIT argument is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <W_BIT>"
    echo "Example: $0 4"
    exit 1
fi

W_BIT=$1
Q_GROUP_SIZE=128

# model_path: the path to the pretrained model or instruct-tuned model
# w_bit: the weight bit-width for AWQ quantization
# q_group_size: the group size for quantization

CUDA_VISIBLE_DEVICES=2 python $DIRPATH/llm-awq/entry.py --model_path $MODEL_PATH \
    --w_bit $W_BIT --q_group_size $Q_GROUP_SIZE \
    --run_awq --dump_awq $CACHE_PATH/$MODEL_PATH-AWQ-w$W_BIT-g$Q_GROUP_SIZE.pt