"""
Base model interface and model registry.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Any, Type, Dict
import torch
from torch import nn, Tensor

_MODEL_REGISTRY: Dict[str, Type] = {}

def register_model(name: str):
    # Register a new dataset
    def decorator(cls):
        _MODEL_REGISTRY[name] = cls
        return cls
    return decorator


class BaseModel(ABC):
    """
    Abstract base class for all models.
    
    Models must implement:
        - forward(): Run model inference
        - mask_token_id: The token ID used for masking
        - load(): Class method to load the model
    """
    
    def __init__(self, config):
        self.config = config
        self._model: Optional[nn.Module] = None
        self._tokenizer: Optional[Any] = None
    
    @property
    def model(self) -> nn.Module:
        """Get the underlying model."""
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        return self._model
    
    @property
    def tokenizer(self) -> Any:
        """Get the tokenizer."""
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not loaded. Call load() first.")
        return self._tokenizer
    
    @property
    def device(self) -> torch.device:
        """Get the model's device."""
        return next(self.model.parameters()).device
    
    @property
    @abstractmethod
    def mask_token_id(self) -> int:
        """Return the mask token ID."""
        pass
    
    @abstractmethod
    def load(self) -> "BaseModel":
        """
        Load the model and tokenizer.
        Returns self for chaining.
        """
        pass
    
    @abstractmethod
    def forward(
        self, 
        input_ids: Tensor, 
        attention_mask: Optional[Tensor] = None
    ) -> Tensor:
        """
        Run model forward pass.
        
        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)
            attention_mask: Optional attention mask of shape (batch_size, seq_len)
        
        Returns:
            logits: Output logits of shape (batch_size, seq_len, vocab_size)
        """
        pass
    
    def encode(
        self, 
        texts: list[str], 
        add_special_tokens: bool = False,
        padding: bool = True,
        return_tensors: str = "pt"
    ) -> dict:
        """
        Encode texts to token IDs.
        
        Args:
            texts: List of text strings
            add_special_tokens: Whether to add special tokens
            padding: Whether to pad sequences
            return_tensors: Return tensor type ("pt" for PyTorch)
        
        Returns:
            Dictionary with 'input_ids' and 'attention_mask'
        """
        return self.tokenizer(
            texts,
            add_special_tokens=add_special_tokens,
            padding=padding,
            return_tensors=return_tensors
        )
    
    def decode(
        self, 
        token_ids: Tensor, 
        skip_special_tokens: bool = True
    ) -> list[str]:
        """
        Decode token IDs to text.
        
        Args:
            token_ids: Token IDs of shape (batch_size, seq_len)
            skip_special_tokens: Whether to skip special tokens
        
        Returns:
            List of decoded text strings
        """
        return self.tokenizer.batch_decode(
            token_ids, 
            skip_special_tokens=skip_special_tokens
        )
    
    def apply_chat_template(
        self,
        messages: list[dict],
        add_generation_prompt: bool = True,
        tokenize: bool = False
    ) -> str:
        """
        Apply chat template to messages.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            add_generation_prompt: Whether to add generation prompt
            tokenize: Whether to tokenize the result
        
        Returns:
            Formatted prompt string
        """
        return self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            tokenize=tokenize
        )
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(path={self.config.path}, device={self.config.device})"



def load_model(config) -> BaseModel:
    """Factory function to load a model from config."""
    model_cls: Type[BaseModel] = model_registry.get(config.name)
    model = model_cls(config)
    return model.load()



def get_torch_dtype(dtype_str: str) -> torch.dtype:
    """Convert string dtype to torch.dtype."""
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if dtype_str not in dtype_map:
        raise ValueError(f"Unknown dtype: {dtype_str}. Available: {list(dtype_map.keys())}")
    return dtype_map[dtype_str]