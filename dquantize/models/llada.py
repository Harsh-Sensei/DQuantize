"""
LLaDA model implementations.

Supports:
    - llada: Standard LLaDA model (bfloat16)
    - llada-gptq: GPTQ quantized
    - llada-awq: AWQ quantized  
    - llada-bnb-4bit: bitsandbytes 4-bit quantization
    - llada-bnb-8bit: bitsandbytes 8-bit quantization
"""

from typing import Optional, Type
import torch
from torch import Tensor

from .base import BaseModel, register_model, _MODEL_REGISTRY, get_torch_dtype

MASK_TOKEN_ID = 126336

def get_torch_dtype(dtype_str: str):
    """Convert a string dtype to torch.dtype."""
    if dtype_str.lower() in ("float32", "fp32"):
        return torch.float32
    elif dtype_str.lower() in ("float16", "fp16"):
        return torch.float16
    elif dtype_str.lower() in ("bfloat16", "bf16"):
        return torch.bfloat16
    else:
        raise ValueError(f"Unknown torch dtype: {dtype_str}")


class LLaDABaseModel(BaseModel):
    """
    Base class for all LLaDA variants.
    Handles loading, tokenization, and forward pass.
    """

    MASK_TOKEN_ID = MASK_TOKEN_ID

    @property
    def mask_token_id(self) -> int:
        return self.MASK_TOKEN_ID

    def _load_tokenizer(self, path: str, trust_remote_code: bool):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            path,
            trust_remote_code=trust_remote_code,
        )
        if tokenizer.padding_side != "left":
            tokenizer.padding_side = "left"

        # Sanity check: pad token must not equal mask token
        assert tokenizer.pad_token_id != self.MASK_TOKEN_ID, \
            "Pad token ID equals mask token ID - this will cause issues"

        return tokenizer

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Run model forward pass."""
        with torch.no_grad():
            outputs = self._model(input_ids, attention_mask=attention_mask)
            return outputs.logits


# --------------------------
# Standard LLaDA (bfloat16)
# --------------------------
@register_model("llada")
class LLaDAModel(LLaDABaseModel):
    def load(self) -> "LLaDAModel":
        from transformers import AutoModel
        # from .utils import get_torch_dtype

        print(f"Loading LLaDA model: {self.config.path}")
        torch_dtype = get_torch_dtype(self.config.torch_dtype)

        self._model = AutoModel.from_pretrained(
            self.config.path,
            trust_remote_code=self.config.trust_remote_code,
            torch_dtype=torch_dtype,
        ).to(self.config.device).eval()

        print("Loading tokenizer...")
        self._tokenizer = self._load_tokenizer(self.config.path, self.config.trust_remote_code)

        return self


# --------------------------
# BitsAndBytes Quantization
# --------------------------
class LLaDABnbModel(LLaDABaseModel):
    """Generic BitsAndBytes quantized LLaDA base class."""

    n_bits: int = 4  # override in subclasses

    def load(self) -> "LLaDABnbModel":
        from transformers import AutoModel, BitsAndBytesConfig
        # from .utils import get_torch_dtype

        print(f"Loading LLaDA model ({self.n_bits}-bit): {self.config.path}")

        bnb_config = BitsAndBytesConfig()
        if self.n_bits == 4:
            bnb_config.load_in_4bit = True
            bnb_config.bnb_4bit_compute_dtype = get_torch_dtype(self.config.torch_dtype)
            bnb_config.bnb_4bit_use_double_quant = self.config.quantization_config.get("double_quant", True)
            bnb_config.bnb_4bit_quant_type = self.config.quantization_config.get("quant_type", "nf4")
        elif self.n_bits == 8:
            bnb_config.load_in_8bit = True
            bnb_config.llm_int8_threshold = self.config.quantization_config.get("threshold", 6.0)
        else:
            raise ValueError(f"Unsupported bits: {self.n_bits}")

        self._model = AutoModel.from_pretrained(
            self.config.path,
            trust_remote_code=self.config.trust_remote_code,
            quantization_config=bnb_config,
            device_map="auto",
        ).eval()

        print("Loading tokenizer...")
        self._tokenizer = self._load_tokenizer(self.config.path, self.config.trust_remote_code)
        return self


@register_model("llada-bnb-4bit")
class LLaDABnb4BitModel(LLaDABnbModel):
    n_bits = 4


@register_model("llada-bnb-8bit")
class LLaDABnb8BitModel(LLaDABnbModel):
    n_bits = 8


# --------------------------
# GPTQ Quantization
# --------------------------
@register_model("llada-gptq")
class LLaDAGPTQModel(LLaDABaseModel):
    def load(self) -> "LLaDAGPTQModel":
        from transformers import AutoModel, GPTQConfig

        print(f"Loading LLaDA model (GPTQ): {self.config.path}")
        gptq_config = GPTQConfig(
            bits=self.config.quantization_config.get("bits", 4),
            dataset=self.config.quantization_config.get("dataset", None),
            tokenizer=None,
        )

        self._model = AutoModel.from_pretrained(
            self.config.path,
            trust_remote_code=self.config.trust_remote_code,
            quantization_config=gptq_config,
            device_map="auto",
        ).eval()

        print("Loading tokenizer...")
        self._tokenizer = self._load_tokenizer(self.config.path, self.config.trust_remote_code)
        return self


# --------------------------
# AWQ Quantization
# --------------------------
@register_model("llada-awq")
class LLaDAAwqModel(LLaDABaseModel):
    def load(self) -> "LLaDAAwqModel":
        from transformers import AutoModel, AwqConfig

        print(f"Loading LLaDA model (AWQ): {self.config.path}")
        awq_config = AwqConfig(
            bits=self.config.quantization_config.get("bits", 4),
            fuse_max_seq_len=self.config.quantization_config.get("fuse_max_seq_len", 512),
            do_fuse=self.config.quantization_config.get("do_fuse", True),
        )

        self._model = AutoModel.from_pretrained(
            self.config.path,
            trust_remote_code=self.config.trust_remote_code,
            quantization_config=awq_config,
            device_map="auto",
        ).eval()

        print("Loading tokenizer...")
        self._tokenizer = self._load_tokenizer(self.config.path, self.config.trust_remote_code)
        return self


# --------------------------
# Factory
# --------------------------
def load_model(config) -> BaseModel:
    """Factory function to load a model from config."""
    model_cls = _MODEL_REGISTRY.get(config.name)
    if model_cls is None:
        raise ValueError(f"Unknown model name: {config.name}")
    model = model_cls(config)
    return model.load()
