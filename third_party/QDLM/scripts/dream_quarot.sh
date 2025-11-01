export HF_DATASETS_TRUST_REMOTE_CODE=true
export HF_ALLOW_CODE_EVAL=1

DIRPATH="$(cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1 && pwd)"
MODEL_PATH="/model/dream-7b-base"

# model_path: the path to the pretrained model
# a_bits: the activation bit-width for QuaRot quantization
# v_bits: the value bit-width for QuaRot quantization
# k_bits: the key bit-width for QuaRot quantization
# w_bits: the weight bit-width for QuaRot quantization
# w_clip: whether to apply weight clipping for QuaRot quantization
# we use symmetric per-channel quantization for weights and per-tensor quantization for activations (query states maintain fp16) in QuaRot
# you could revise quantization settings according to your needs, please refer to ./QuaRot/fake_quant/README.md

# general qa tasks
# --tasks piqa,winogrande,arc_easy,arc_challenge
python $DIRPATH/QuaRot/fake_quant/main.py --model $MODEL_PATH \
    --a_bits 16 --v_bits 16 --k_bits 16 --w_bits 16 --w_clip \
    --rotate --tasks piqa \