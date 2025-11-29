import torch
import numpy as np
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Callable, Optional, Tuple
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import os
import sys

# Import apply_awq from the AWQ quantization module
awq_path = os.path.join(os.path.dirname(__file__), '..', 'third_party', 'QDLM', 'llm-awq')
if awq_path not in sys.path:
    sys.path.insert(0, awq_path)
from awq.quantize.pre_quant import apply_awq
from awq.quantize.quantizer import real_quantize_model_weight, pseudo_quantize_model_weight
import re
import time

@dataclass
class DQuantizeConfig:
    """Configuration for dynamic quantization strategies and generation parameters."""
    k: int = 64  # Number of steps for 'firstk' and 'lastk' strategies
    steps: int = 128  # Sampling steps per block
    gen_length: int = 128  # Total generation length
    block_length: int = 128  # Length of each generation block
    temperature: float = 0.  # Sampling temperature
    cfg_scale: float = 0.  # Classifier-free guidance scale
    logits_eos_inf: bool = False  # Set EOS logits to -inf
    confidence_eos_eot_inf: bool = False  # Set EOS/EoT confidence to -inf
    batch_size: int = 8  # Maximum batch size for processing
    apply_chat_template: bool = True  # Apply chat template for instruct models
    mask_id: int = 126336  # Token ID for mask token


def add_gumbel_noise(logits, temperature):
    """
    Add Gumbel noise for sampling categorical distributions.
    Uses float64 for numerical stability.
    """
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def infer_quantization_params_from_filename(filename: str) -> Tuple[int, int]:
    """
    Infer w_bit and q_group_size from AWQ metadata filename.
    
    Expected filename patterns:
    - "*-w4-g128.pt" -> w_bit=4, q_group_size=128
    - "*-w3-g128.pt" -> w_bit=3, q_group_size=128
    - "*-w4-g64.pt" -> w_bit=4, q_group_size=64
    
    Args:
        filename: Path to AWQ metadata file
        
    Returns:
        Tuple of (w_bit, q_group_size)
        
    Raises:
        ValueError: If cannot infer parameters from filename
    """
    basename = os.path.basename(filename)
    
    # Try pattern: -w4-g128 or -w4g128
    pattern1 = r'-w(\d+)-g(\d+)'
    match = re.search(pattern1, basename)
    if match:
        return int(match.group(1)), int(match.group(2))
    
    # Try pattern: -w4g128 (no dash between w and g)
    pattern2 = r'-w(\d+)g(\d+)'
    match = re.search(pattern2, basename)
    if match:
        return int(match.group(1)), int(match.group(2))
    
    # Try pattern: w4-g128 or w4g128 (no leading dash)
    pattern3 = r'w(\d+)-g(\d+)'
    match = re.search(pattern3, basename)
    if match:
        return int(match.group(1)), int(match.group(2))
    
    pattern4 = r'w(\d+)g(\d+)'
    match = re.search(pattern4, basename)
    if match:
        return int(match.group(1)), int(match.group(2))
    
    raise ValueError(
        f"Cannot infer w_bit and q_group_size from filename: {filename}. "
        f"Expected pattern like '*-w4-g128.pt' or '*-w4g128.pt'"
    )


def get_num_transfer_tokens(mask_index, steps):
    """
    Precompute the number of tokens that need to be transitioned at each step.
    For linear noise schedule, the expected number should be consistent across steps.
    """
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    
    num_transfer_tokens = torch.zeros(
        mask_num.size(0), steps, 
        device=mask_index.device, 
        dtype=torch.int64
    ) + base
    
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1
    
    return num_transfer_tokens


class DQuantizeModelLLada:
    """
    Dynamic Quantization wrapper for LLaDA models.
    
    This class manages two models (precise and quantized) and dynamically
    selects which model to use at each generation step based on the strategy.
    """
    
    def __init__(
        self,
        model_name: str,
        quantized_model_path: str,
        strategy: str = "all",
        dquantize_config: Optional[DQuantizeConfig] = None,
        device_precise: str = 'cuda:0',
        device_quantized: str = 'cuda:1',
        torch_dtype = torch.bfloat16,
        q_backend: str = 'real'
    ):
        """
        Initialize the DQuantizeModelLLada.
        
        Args:
            model_name: HuggingFace model name or path for the base model
            quantized_model_path: Path to AWQ metadata file (.pt)
            strategy: Strategy for model selection ('all', 'firstk', 'lastk')
            dquantize_config: Configuration for strategies (contains k parameter)
            device_precise: Device for precise model (default: 'cuda:0')
            device_quantized: Device for quantized model (default: 'cuda:1')
            torch_dtype: Data type for models (default: torch.bfloat16)
            q_backend: Quantization backend ('real' or 'fake', default: 'real')
        """
        # Validate inputs
        if strategy not in ['all', 'firstk', 'lastk']:
            raise ValueError(f"Invalid strategy: {strategy}. Must be 'all', 'firstk', or 'lastk'")
        
        if q_backend not in ['real', 'fake']:
            raise ValueError(f"Invalid q_backend: {q_backend}. Must be 'real' or 'fake'")
        
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. This class requires GPU support.")
        
        num_gpus = torch.cuda.device_count()
        if num_gpus < 2:
            raise RuntimeError(f"At least 2 GPUs are required, but only {num_gpus} GPU(s) available.")
        
        self.model_name = model_name
        self.strategy = strategy
        self.dquantize_config = dquantize_config or DQuantizeConfig()
        self.device_precise = device_precise
        self.device_quantized = device_quantized
        self.torch_dtype = torch_dtype
        self.q_backend = q_backend
        
        # Load models
        print(f"Loading precise model on {device_precise}: {model_name}")
        self.precise_model = AutoModel.from_pretrained(
            model_name, 
            trust_remote_code=True, 
            torch_dtype=torch_dtype
        ).to(device_precise).eval()
        
        print(f"Loading base model for quantization on {device_quantized}: {model_name}")
        self.quantized_model = AutoModel.from_pretrained(
            model_name, 
            trust_remote_code=True, 
            torch_dtype=torch_dtype
        ).to(device_quantized).eval()
        
        # Apply AWQ quantization
        # NOTE: apply_awq modifies the model by applying activation scaling and clipping.
        # This changes the model's behavior even without weight quantization (q_backend='fake').
        # The scaling factors are applied to layer norms, activations (GELU, SiLU), and linear layers
        # to prepare the model for quantization. This is why outputs may differ from the precise model
        # even when weight quantization is not applied.
        print(f"Loading AWQ metadata from: {quantized_model_path}")
        if not os.path.exists(quantized_model_path):
            raise FileNotFoundError(f"AWQ metadata file not found: {quantized_model_path}")
        
        awq_results = torch.load(quantized_model_path, map_location='cpu')
        apply_awq(self.quantized_model, awq_results)
        
        # Infer w_bit and q_group_size from filename (needed for both real and fake quantization)
        try:
            w_bit, q_group_size = infer_quantization_params_from_filename(quantized_model_path)
            print(f"Inferred quantization parameters: w_bit={w_bit}, q_group_size={q_group_size}")
        except ValueError as e:
            raise ValueError(
                f"Failed to infer quantization parameters from filename: {e}. "
                f"Please ensure the filename follows the pattern '*-w4-g128.pt' or '*-w4g128.pt'"
            )
        
        # Create quantization config
        q_config = {
            "zero_point": True,  # AWQ uses zero_point by default
            "q_group_size": q_group_size,
        }
        
        # Apply weight quantization based on q_backend
        if self.q_backend == 'real':
            print("Applying real weight quantization...")
            # Apply real quantization
            real_quantize_model_weight(
                self.quantized_model, 
                w_bit=w_bit, 
                q_config=q_config
            )
            print(f"Applied real weight quantization (w_bit={w_bit}, q_group_size={q_group_size})")
        elif self.q_backend == 'fake':
            print("Applying pseudo weight quantization...")
            # Apply pseudo quantization
            pseudo_quantize_model_weight(
                self.quantized_model,
                w_bit=w_bit,
                q_config=q_config
            )
            print(f"Applied pseudo weight quantization (w_bit={w_bit}, q_group_size={q_group_size})")
        else:
            raise ValueError(f"Invalid q_backend: {self.q_backend}. Must be 'real' or 'fake'")
        
        # Ensure quantized model is on correct device
        print(f"Moving quantized model to {device_quantized}...")
        self.quantized_model = self.quantized_model.to(device_quantized)
        
        for name, param in self.quantized_model.named_parameters():
            if param.device != torch.device(device_quantized):
                param.data = param.data.to(device_quantized)
        for name, buffer in self.quantized_model.named_buffers():
            if buffer.device != torch.device(device_quantized):
                buffer.data = buffer.data.to(device_quantized)
        
        print(f"Quantization complete. Model on {device_quantized}")
        
        # Load tokenizer
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # LLaDA works better with left padding
        if self.tokenizer.padding_side != 'left':
            self.tokenizer.padding_side = 'left'
        
        # Validate padding token
        assert self.tokenizer.pad_token_id != 126336, "Padding token conflicts with mask token"
        
        # Set selection function based on strategy
        self.selection_fn = self._get_selection_function()
        
        print(f"Initialized DQuantizeModelLLada with strategy: {strategy}")
    
    def _get_selection_function(self) -> Callable[[int, int], bool]:
        """
        Get the appropriate selection function based on strategy.
        
        Returns:
            A function that takes (current_step, total_steps) and returns
            True if quantized model should be used, False for precise model.
        """
        if self.strategy == 'all':
            return self._select_all_quantized
        elif self.strategy == 'firstk':
            return self._select_firstk
        elif self.strategy == 'lastk':
            return self._select_lastk
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def _select_all_quantized(self, current_step: int, total_steps: int) -> bool:
        """Always use quantized model."""
        return True
    
    def _select_firstk(self, current_step: int, total_steps: int) -> bool:
        """Use quantized model for first k steps."""
        return current_step < self.dquantize_config.k
    
    def _select_lastk(self, current_step: int, total_steps: int) -> bool:
        """Use quantized model for last k steps."""
        return current_step >= (total_steps - self.dquantize_config.k)
    
    @torch.no_grad()
    def _generate_batch(
        self,
        prompt: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Generate tokens for a single batch using dynamic quantization.
        
        Args:
            prompt: Input token IDs of shape (batch_size, L)
            attention_mask: Optional attention mask
        
        Returns:
            Generated token IDs of shape (batch_size, L + gen_length)
        """
        # Get parameters from config
        steps = self.dquantize_config.steps
        gen_length = self.dquantize_config.gen_length
        block_length = self.dquantize_config.block_length
        temperature = self.dquantize_config.temperature
        cfg_scale = self.dquantize_config.cfg_scale
        mask_id = self.dquantize_config.mask_id
        logits_eos_inf = self.dquantize_config.logits_eos_inf
        confidence_eos_eot_inf = self.dquantize_config.confidence_eos_eot_inf
        # Initialize output with masks
        x = torch.full(
            (prompt.shape[0], prompt.shape[1] + gen_length), 
            mask_id, 
            dtype=torch.long
        ).to(self.device_precise)
        x[:, :prompt.shape[1]] = prompt.clone()
        
        if attention_mask is not None:
            attention_mask = torch.cat([
                attention_mask, 
                torch.ones((prompt.shape[0], gen_length), 
                          dtype=attention_mask.dtype, 
                          device=self.device_precise)
            ], dim=-1)
        
        prompt_index = (x != mask_id)
        
        # Validate block configuration
        assert gen_length % block_length == 0
        num_blocks = gen_length // block_length
        
        assert steps % num_blocks == 0
        steps_per_block = steps // num_blocks
        
        
        # Generate block by block
        for num_block in tqdm(range(num_blocks), desc="Generating blocks"):
            block_start = prompt.shape[1] + num_block * block_length
            block_end = prompt.shape[1] + (num_block + 1) * block_length
            
            block_mask_index = (x[:, block_start:block_end] == mask_id)
            num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)
            
            for step_idx in tqdm(range(steps_per_block), desc="Generating steps"):
                # Determine which model to use
                start_pre_model_time = time.time()
                use_quantized = self.selection_fn(step_idx, steps_per_block)
                model = self.quantized_model if use_quantized else self.precise_model
                model_device = self.device_quantized if use_quantized else self.device_precise
                
                # Move inputs to appropriate device
                x_input = x.to(model_device)
                attention_mask_input = attention_mask.to(model_device) if attention_mask is not None else None
                prompt_index_input = prompt_index.to(model_device)
                
                # Model inference
                mask_index = (x_input == mask_id)
                print(f"Time taken for pre-model operations: {time.time() - start_pre_model_time} seconds")
                start_model_inference_time = time.time()
                if cfg_scale > 0.:
                    un_x = x_input.clone()
                    un_x[prompt_index_input] = mask_id
                    x_concat = torch.cat([x_input, un_x], dim=0)
                    
                    if attention_mask_input is not None:
                        attention_mask_concat = torch.cat([attention_mask_input, attention_mask_input], dim=0)
                    else:
                        attention_mask_concat = None
                    
                    logits = model(x_concat, attention_mask=attention_mask_concat).logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    logits = model(x_input, attention_mask=attention_mask_input).logits
                print(f"Time taken for model inference: {time.time() - start_model_inference_time} seconds")
                start_sampling_time = time.time()
                if logits_eos_inf:
                    logits[:, :, 126081] = -torch.inf
                
                # Sampling
                logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)
                
                if confidence_eos_eot_inf:
                    logits_with_noise[:, :, 126081] = logits[:, :, 126348] = -torch.inf
                
                # Compute confidence
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                )
                
                x0_p[:, block_end:] = -np.inf
                
                x0 = torch.where(mask_index, x0, x_input)
                confidence = torch.where(mask_index, x0_p, -np.inf)
                
                # Token selection (low confidence remasking)
                transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                
                for j in range(confidence.shape[0]):
                    _, select_index = torch.topk(
                        confidence[j], 
                        k=num_transfer_tokens[j, step_idx] + 1
                    )
                    transfer_index[j, select_index[:-1]] = True
                
                # Move results back to precise device and update x
                x_input[transfer_index] = x0[transfer_index]
                x = x_input.to(self.device_precise)
                print(f"Time taken for sampling: {time.time() - start_sampling_time} seconds")
        return x
    
    def generate(
        self,
        prompts: List[str],
    ) -> List[str]:
        """
        Generate text completions for a batch of prompts.
        
        Args:
            prompts: List of input text prompts
        
        Returns:
            List of generated text completions
        """
        # Get parameters from config
        batch_size = self.dquantize_config.batch_size
        apply_chat_template = self.dquantize_config.apply_chat_template
        
        # Apply chat template if needed
        if apply_chat_template and 'instruct' in self.model_name.lower():
            messages = [{"role": "user", "content": prompt} for prompt in prompts]
            formatted_prompts = [
                self.tokenizer.apply_chat_template(
                    [message], 
                    add_generation_prompt=True, 
                    tokenize=False
                ) for message in messages
            ]
        else:
            formatted_prompts = prompts
        
        # Process in batches
        all_outputs = []
        num_batches = (len(formatted_prompts) + batch_size - 1) // batch_size
        
        for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(formatted_prompts))
            batch_prompts = formatted_prompts[start_idx:end_idx]
            
            # Tokenize
            encoded = self.tokenizer(
                batch_prompts,
                add_special_tokens=False,
                padding=True,
                return_tensors="pt"
            )
            
            input_ids = encoded['input_ids'].to(self.device_precise)
            attention_mask = encoded['attention_mask'].to(self.device_precise)
            
            # Generate
            output_ids = self._generate_batch(
                prompt=input_ids,
                attention_mask=attention_mask,
            )
            
            # Extract generated portion (remove prompt)
            generated_ids = output_ids[:, input_ids.shape[1]:]
            
            # Decode
            batch_outputs = self.tokenizer.batch_decode(
                generated_ids, 
                skip_special_tokens=True
            )
            
            all_outputs.extend(batch_outputs)
        
        return all_outputs


def main():
    """Example usage of DQuantizeModelLLada."""
    # Configuration
    model_name = 'GSAI-ML/LLaDA-8B-Instruct'
    quantized_model_path = '/home/scratch/hshah2/dquantize_cache/GSAI-ML/LLaDA-8B-Instruct-w4-g128.pt'  # Replace with actual path
    
    # Create toy prompts
    prompts = [
        "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?",
        "Joy can read 8 pages of a book in 20 minutes. How many hours will it take her to read 120 pages?",
        "Randy has 60 mango trees on his farm. He also has 5 less than half as many coconut trees as mango trees. How many trees does Randy have in all on his farm?"
    ]
    
    # Initialize with 'firstk' strategy
    config = DQuantizeConfig(
        k=64,
        steps=128,
        gen_length=128,
        block_length=32,
        temperature=0.,
        batch_size=2,
    )
    dq_model = DQuantizeModelLLada(
        model_name=model_name,
        quantized_model_path=quantized_model_path,
        strategy='firstk',
        dquantize_config=config,
    )
    
    # Generate
    outputs = dq_model.generate(prompts=prompts)
    
    # Print results
    for i, (prompt, output) in enumerate(zip(prompts, outputs)):
        print(f"\n{'='*80}")
        print(f"Example {i+1}")
        print(f"{'='*80}")
        print(f"Prompt: {prompt}")
        print(f"\nGenerated: {output}")


if __name__ == '__main__':
    main()