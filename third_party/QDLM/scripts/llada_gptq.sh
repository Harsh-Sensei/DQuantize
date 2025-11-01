# Quantize the model using AutoGPTQ
export HF_DATASETS_TRUST_REMOTE_CODE=true
export HF_ALLOW_CODE_EVAL=1

DIRPATH="$(cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1 && pwd)"
MODEL_PATH='/path/to/your/model'  # Replace with your model path
W_BIT=3

# model_path: the path to the pretrained model or instruct-tuned model
# wbits: the weight bit-width for GPTQ quantization
# by default, we use 128 as the group size for GPTQ quantization

# general tasks
# --tasks hellaswag,piqa,winogrande,arc_easy,arc_challenge
python $DIRPATH/AutoGPTQ/quantize.py --model $MODEL_PATH --wbits $W_BIT --tasks piqa


# MMLU
python $DIRPATH/AutoGPTQ/quantize.py --model $MODEL_PATH --wbits $W_BIT --tasks mmlu --num_fewshot 5 --mc_num 1


# GSM8K
python $DIRPATH/AutoGPTQ/quantize.py --model $MODEL_PATH --wbits $W_BIT \
    --task gsm8k --gen_length 256 --steps 256 --block_length 32 --num_fewshot 4 \


# MATH
python $DIRPATH/AutoGPTQ/quantize.py --model $MODEL_PATH --wbits $W_BIT \
    --tasks minerva_math  --num_fewshot 0 --gen_length 256 --steps 256 --block_length 64 \


# HumanEval
python $DIRPATH/AutoGPTQ/quantize.py --model $MODEL_PATH --wbits $W_BIT \
    --task humaneval --gen_length 512 --steps 512 --block_length 32 --num_fewshot 0 \


# MBPP
python $DIRPATH/AutoGPTQ/quantize.py --model $MODEL_PATH --wbits $W_BIT \
    --task mbpp --gen_length 512 --steps 512 --block_length 32 --num_fewshot 3 \