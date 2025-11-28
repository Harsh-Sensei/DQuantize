import torch
import json
import numpy as np
import torch.nn.functional as F
import argparse
import os
import yaml
from datetime import datetime
from datasets import load_dataset
from tqdm import tqdm
import time
from collections import defaultdict
import pandas as pd

# Import apply_awq from the AWQ quantization module
import sys
awq_path = os.path.join(os.path.dirname(__file__), '..', 'third_party', 'QDLM', 'llm-awq')
if awq_path not in sys.path:
    sys.path.insert(0, awq_path)
from awq.quantize.pre_quant import apply_awq

from transformers import AutoModel, AutoTokenizer


def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    Uses float64 for numerical stability.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    '''
    Precompute the number of tokens that need to be transitioned at each step.
    '''
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


@torch.no_grad()
def generate_batch(
        model, prompt, attention_mask=None, steps=128, gen_length=128, 
        block_length=128, temperature=0., cfg_scale=0., 
        mask_id=126336, logits_eos_inf=False, confidence_eos_eot_inf=False):
    '''
    Generate text using the precise model.
    
    Args:
        model: Mask predictor model
        prompt: Input tensor of shape (batch_size, L)
        attention_mask: Optional attention mask
        steps: Sampling steps per block
        gen_length: Generated answer length
        block_length: Block length
        temperature: Sampling temperature
        cfg_scale: Classifier-free guidance scale
        mask_id: Token ID of [MASK]
        logits_eos_inf: Whether to set EOS logits to -inf
        confidence_eos_eot_inf: Whether to set confidence of EOS/EoT to -inf
    
    Returns:
        x: Generated tokens of shape (batch_size, L + gen_length)
    '''
    # Initialize output with all masks
    x = torch.full(
        (prompt.shape[0], prompt.shape[1] + gen_length), 
        mask_id, 
        dtype=torch.long
    ).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()
    
    if attention_mask is not None:
        attention_mask = torch.cat([
            attention_mask, 
            torch.ones((prompt.shape[0], gen_length), 
                      dtype=attention_mask.dtype, 
                      device=model.device)
        ], dim=-1)
    
    prompt_index = (x != mask_id)
    
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    
    assert steps % num_blocks == 0
    steps = steps // num_blocks
    
    # Generate block by block
    for num_block in tqdm(range(num_blocks), desc="Generating blocks", position=0):
        block_start = prompt.shape[1] + num_block * block_length
        block_end = prompt.shape[1] + (num_block + 1) * block_length
        
        block_mask_index = (x[:, block_start:block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        
        for i in range(steps):
            mask_index = (x == mask_id)
            
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                if attention_mask is not None:
                    attention_mask_ = torch.cat([attention_mask, attention_mask], dim=0)
                else:
                    attention_mask_ = None
                
                logits = model(x_, attention_mask=attention_mask_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x, attention_mask=attention_mask).logits
            
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
            
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)
            
            # Token selection
            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(
                    confidence[j], 
                    k=num_transfer_tokens[j, i] + 1
                )
                transfer_index[j, select_index[:-1]] = True
            
            x[transfer_index] = x0[transfer_index]
    
    return x


@torch.no_grad()
def forward_process(batch, prompt_index, mask_id, batch_size, device):
    """
    Forward diffusion process: randomly mask tokens in the target portion.
    
    Args:
        batch: Input sequences of shape (batch_size, seq_len)
        prompt_index: Boolean mask indicating prompt positions
        mask_id: Token ID for mask token
        batch_size: Batch size
        device: Device to use
    
    Returns:
        noisy_batch: Batch with random tokens masked
        p_mask: Probability of masking (k/target_len) for each position
    """
    b, l = batch.shape
    
    target_len = (l - prompt_index.sum()).item()
    
    # Sample random masking level k from [1, target_len]
    k = torch.randint(1, target_len + 1, (), device=device)
    
    # Create different masking levels for each sample in batch
    x = torch.round(
        torch.linspace(
            float(k), 
            k + (b - 1) * (target_len / b), 
            steps=b, 
            device=device
        )
    ).long()
    x = ((x - 1) % target_len) + 1
    assert x.min() >= 1 and x.max() <= target_len
    
    # Create random mask patterns
    indices = torch.arange(target_len, device=device).repeat(b, 1)
    is_mask = indices < x.unsqueeze(1)
    
    # Randomly permute mask positions for each sample
    for i in range(b):
        is_mask[i] = is_mask[i][torch.randperm(target_len, device=device)]
    
    # Prepend zeros for prompt portion
    is_mask = torch.cat((
        torch.zeros(b, prompt_index.sum(), dtype=torch.bool, device=device), 
        is_mask
    ), dim=1)
    
    # Apply masking
    noisy_batch = torch.where(is_mask, mask_id, batch)
    
    # Compute masking probability
    p_mask = (x / target_len).unsqueeze(1).repeat(1, l)
    
    return noisy_batch, p_mask


@torch.no_grad()
def get_loglikelihood(
        model, prefix, target, mask_id, mc_num=100, 
        batch_size=10, cfg_scale=0., device='cuda'):
    """
    Compute log-likelihood of target given prefix using Monte Carlo estimation.
    
    Args:
        model: Mask predictor model
        prefix: Prefix tokens (1D tensor)
        target: Target tokens (1D tensor)
        mask_id: Token ID for mask token
        mc_num: Number of Monte Carlo samples
        batch_size: Batch size for MC estimation
        cfg_scale: Classifier-free guidance scale
        device: Device to use
    
    Returns:
        log_likelihood: Estimated log-likelihood
    """
    # Create full sequence
    seq = torch.cat([prefix, target]).unsqueeze(0)
    seq = seq.repeat(batch_size, 1).to(device)
    
    prompt_index = torch.arange(seq.shape[1], device=device) < len(prefix)
    
    loss_acc = []
    
    num_batches = mc_num // batch_size
    for _ in tqdm(range(num_batches), desc="MC samples", leave=False, position=1):
        # Forward process: add noise
        perturbed_seq, p_mask = forward_process(
            seq, prompt_index, mask_id, batch_size, device
        )
        
        mask_indices = perturbed_seq == mask_id
        
        # Get model predictions
        if cfg_scale > 0.:
            un_x = perturbed_seq.clone()
            un_x[prompt_index.unsqueeze(0).repeat(batch_size, 1)] = mask_id
            x_ = torch.cat([perturbed_seq, un_x], dim=0)
            
            logits = model(x_).logits
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
        else:
            logits = model(perturbed_seq).logits
        
        logits = logits.to(device)
        
        # Compute cross-entropy loss weighted by masking probability
        loss = F.cross_entropy(
            logits[mask_indices], 
            seq[mask_indices], 
            reduction='none'
        ) / p_mask[mask_indices]
        
        loss = loss.sum() / batch_size
        loss_acc.append(loss.item())
    
    # Return negative average loss as log-likelihood
    return -sum(loss_acc) / len(loss_acc)


def load_dataset_examples(dataset_name, tokenizer, max_examples, max_length, min_length=0):
    """Load examples from a dataset."""
    if dataset_name == 'toy':
        prompts = [
            "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?",
            "Joy can read 8 pages of a book in 20 minutes. How many hours will it take her to read 120 pages?",
            "Randy has 60 mango trees on his farm. He also has 5 less than half as many coconut trees as mango trees. How many trees does Randy have in all on his farm?"
        ]
        
        if min_length > 0:
            filtered_prompts = []
            for prompt in prompts:
                tokens = tokenizer.encode(prompt, add_special_tokens=False)
                if len(tokens) >= min_length:
                    if max_length > 0 and len(tokens) > max_length:
                        tokens = tokens[:max_length]
                        prompt = tokenizer.decode(tokens, skip_special_tokens=True)
                    filtered_prompts.append(prompt)
            prompts = filtered_prompts
        
        prompts = prompts[:max_examples]
        return prompts
    
    elif dataset_name == 'wikitext2':
        print("Loading wikitext2 dataset...")
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
        texts = []
        
        try:
            dataset_len = len(dataset)
            total = min(max_examples * 2, dataset_len)
        except (TypeError, AttributeError):
            total = max_examples * 2
        
        pbar = tqdm(enumerate(dataset), desc="Loading examples", total=total if total > 0 else None)
        for i, example in pbar:
            if len(texts) >= max_examples:
                break
            text = example['text'].strip()
            if len(text) > 0:
                tokens = tokenizer.encode(text, add_special_tokens=False)
                
                if min_length > 0 and len(tokens) < min_length:
                    continue
                
                if max_length > 0 and len(tokens) > max_length:
                    tokens = tokens[:max_length]
                    text = tokenizer.decode(tokens, skip_special_tokens=True)
                
                texts.append(text)
                pbar.set_postfix({"loaded": len(texts), "target": max_examples})
        pbar.close()
        return texts[:max_examples]
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Must be 'toy' or 'wikitext2'")


def main():
    parser = argparse.ArgumentParser(description='Run LLaDA likelihood analysis')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='GSAI-ML/LLaDA-8B-Instruct',
                        help='Precise model name or path')
    parser.add_argument('--quantized_model', type=str, required=True,
                        help='Path to AWQ metadata file (.pt)')
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, choices=['toy', 'wikitext2'], default='toy',
                        help='Dataset to load')
    parser.add_argument('--max_examples', type=int, default=3,
                        help='Maximum number of examples to process')
    parser.add_argument('--max_length', type=int, default=0,
                        help='Maximum length of each example in tokens')
    parser.add_argument('--min_length', type=int, default=0,
                        help='Minimum length of each example in tokens')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory to save results')
    
    # Generation arguments
    parser.add_argument('--steps', type=int, default=128,
                        help='Sampling steps')
    parser.add_argument('--gen_length', type=int, default=128,
                        help='Generated answer length')
    parser.add_argument('--block_length', type=int, default=32,
                        help='Block length')
    parser.add_argument('--temperature', type=float, default=0.,
                        help='Sampling temperature')
    parser.add_argument('--cfg_scale', type=float, default=0.,
                        help='Classifier-free guidance scale')
    parser.add_argument('--logits_eos_inf', action='store_true',
                        help='Set logits of EOS token to -inf')
    parser.add_argument('--confidence_eos_eot_inf', action='store_true',
                        help='Set confidence of EOS/EoT to -inf')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for generation')
    
    # Likelihood estimation arguments
    parser.add_argument('--mc_num', type=int, default=100,
                        help='Number of Monte Carlo samples for likelihood estimation')
    parser.add_argument('--mc_batch_size', type=int, default=10,
                        help='Batch size for Monte Carlo sampling')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save configuration
    config = {
        'model': args.model,
        'quantized_model': args.quantized_model,
        'dataset': args.dataset,
        'max_examples': args.max_examples,
        'max_length': args.max_length,
        'min_length': args.min_length,
        'steps': args.steps,
        'gen_length': args.gen_length,
        'block_length': args.block_length,
        'temperature': args.temperature,
        'cfg_scale': args.cfg_scale,
        'logits_eos_inf': args.logits_eos_inf,
        'confidence_eos_eot_inf': args.confidence_eos_eot_inf,
        'batch_size': args.batch_size,
        'mc_num': args.mc_num,
        'mc_batch_size': args.mc_batch_size,
    }
    
    config_path = os.path.join(args.output_dir, 'likelihood_config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"Configuration saved to: {config_path}")
    
    # Check GPU availability
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This script requires GPU support.")
    
    num_gpus = torch.cuda.device_count()
    if num_gpus < 2:
        raise RuntimeError(f"At least 2 GPUs are required, but only {num_gpus} GPU(s) available.")
    
    print(f"Found {num_gpus} GPU(s)")
    device_precise = 'cuda:0'
    device_quantized = 'cuda:1'
    
    # Load models
    print(f"Loading precise model on {device_precise}: {args.model}")
    precise_model = AutoModel.from_pretrained(
        args.model, 
        trust_remote_code=True, 
        torch_dtype=torch.bfloat16
    ).to(device_precise).eval()
    
    print(f"Loading base model for quantization on {device_quantized}: {args.model}")
    quantized_model = AutoModel.from_pretrained(
        args.model, 
        trust_remote_code=True, 
        torch_dtype=torch.bfloat16
    ).to(device_quantized).eval()
    
    # Apply AWQ quantization
    print(f"Loading AWQ metadata from: {args.quantized_model}")
    if not os.path.exists(args.quantized_model):
        raise FileNotFoundError(f"AWQ metadata file not found: {args.quantized_model}")
    
    awq_results = torch.load(args.quantized_model, map_location='cpu')
    apply_awq(quantized_model, awq_results)
    
    print(f"Moving quantized model to {device_quantized}...")
    quantized_model = quantized_model.to(device_quantized)
    
    for name, param in quantized_model.named_parameters():
        if param.device != torch.device(device_quantized):
            param.data = param.data.to(device_quantized)
    for name, buffer in quantized_model.named_buffers():
        if buffer.device != torch.device(device_quantized):
            buffer.data = buffer.data.to(device_quantized)
    
    print(f"Applied AWQ quantization to model on {device_quantized}")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    
    if tokenizer.padding_side != 'left':
        tokenizer.padding_side = 'left'
    
    assert tokenizer.pad_token_id != 126336
    
    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    raw_prompts = load_dataset_examples(
        args.dataset, tokenizer, args.max_examples, 
        args.max_length, args.min_length
    )
    
    # Apply chat template if instruct model
    print("Preparing prompts...")
    if 'instruct' in args.model.lower():
        messages = [{"role": "user", "content": prompt} for prompt in raw_prompts]
        prompts = [
            tokenizer.apply_chat_template(
                [message], 
                add_generation_prompt=True, 
                tokenize=False
            ) for message in messages
        ]
    else:
        prompts = raw_prompts
    
    print("Tokenizing prompts...")
    encoded_outputs = tokenizer(
        prompts,
        add_special_tokens=False,
        padding=True,
        return_tensors="pt"
    )
    input_ids = encoded_outputs['input_ids'].to(device_precise)
    attention_mask = encoded_outputs['attention_mask'].to(device_precise)
    
    # Generate text using precise model
    print("\n" + "="*80)
    print("STEP 1: GENERATING TEXT WITH PRECISE MODEL")
    print("="*80)
    
    num_samples = input_ids.shape[0]
    num_batches = (num_samples + args.batch_size - 1) // args.batch_size
    
    all_outputs = []
    all_prompts = []
    all_targets = []
    
    for batch_idx in tqdm(range(num_batches), desc="Generating batches"):
        start_idx = batch_idx * args.batch_size
        end_idx = min(start_idx + args.batch_size, num_samples)
        
        batch_input_ids = input_ids[start_idx:end_idx]
        batch_attention_mask = attention_mask[start_idx:end_idx]
        
        batch_output = generate_batch(
            model=precise_model,
            prompt=batch_input_ids,
            attention_mask=batch_attention_mask,
            steps=args.steps,
            gen_length=args.gen_length,
            block_length=args.block_length,
            temperature=args.temperature,
            cfg_scale=args.cfg_scale,
            logits_eos_inf=args.logits_eos_inf,
            confidence_eos_eot_inf=args.confidence_eos_eot_inf,
        )
        
        all_outputs.append(batch_output)
        
        # Store prefix and target separately
        for i in range(batch_output.shape[0]):
            prefix_tokens = batch_input_ids[i]
            generated_tokens = batch_output[i, batch_input_ids.shape[1]:]
            
            # Remove padding from prefix
            prefix_tokens = prefix_tokens[batch_attention_mask[i].bool()]
            
            # Remove masks and EOS from generated tokens
            mask_id = 126336
            eos_id = 126081
            valid_mask = (generated_tokens != mask_id) & (generated_tokens != eos_id)
            generated_tokens = generated_tokens[valid_mask]
            
            all_prompts.append(prefix_tokens)
            all_targets.append(generated_tokens)
    
    # Concatenate all outputs for decoding
    full_outputs = torch.cat(all_outputs, dim=0)
    generated_text = tokenizer.batch_decode(
        full_outputs[:, input_ids.shape[1]:], 
        skip_special_tokens=True
    )
    
    print("\nGenerated samples:")
    for i, text in enumerate(generated_text[:3]):  # Show first 3
        print(f"\nSample {i+1}:")
        print(text)
        print('-' * 50)
    
    # Save generated text
    generated_path = os.path.join(args.output_dir, 'generated_text.jsonl')
    with open(generated_path, 'w') as f:
        for i, text in enumerate(generated_text):
            f.write(json.dumps({
                'sample_idx': i,
                'prompt': raw_prompts[i],
                'generated': text
            }) + '\n')
    print(f"\nGenerated text saved to: {generated_path}")
    
    # Compute likelihoods
    print("\n" + "="*80)
    print("STEP 2: COMPUTING LOG-LIKELIHOODS")
    print("="*80)
    
    likelihood_results = []
    
    for i in tqdm(range(len(all_prompts)), desc="Computing likelihoods"):
        prefix = all_prompts[i]
        target = all_targets[i]
        
        if len(target) == 0:
            print(f"Warning: Sample {i} has no generated tokens. Skipping.")
            continue
        
        print(f"\nProcessing sample {i+1}/{len(all_prompts)}")
        print(f"  Prefix length: {len(prefix)} tokens")
        print(f"  Target length: {len(target)} tokens")
        
        # Compute likelihood with precise model
        print("  Computing precise model likelihood...")
        ll_precise = get_loglikelihood(
            model=precise_model,
            prefix=prefix,
            target=target,
            mask_id=126336,
            mc_num=args.mc_num,
            batch_size=args.mc_batch_size,
            cfg_scale=args.cfg_scale,
            device=device_precise
        )
        
        # Compute likelihood with quantized model
        print("  Computing quantized model likelihood...")
        prefix_q = prefix.to(device_quantized)
        target_q = target.to(device_quantized)
        
        ll_quantized = get_loglikelihood(
            model=quantized_model,
            prefix=prefix_q,
            target=target_q,
            mask_id=126336,
            mc_num=args.mc_num,
            batch_size=args.mc_batch_size,
            cfg_scale=args.cfg_scale,
            device=device_quantized
        )
        
        # Compute difference
        ll_diff = ll_precise - ll_quantized
        
        result = {
            'sample_idx': i,
            'prefix_length': len(prefix),
            'target_length': len(target),
            'log_likelihood_precise': float(ll_precise),
            'log_likelihood_quantized': float(ll_quantized),
            'log_likelihood_diff': float(ll_diff),
        }
        
        likelihood_results.append(result)
        
        print(f"  LL (precise):    {ll_precise:.6f}")
        print(f"  LL (quantized):  {ll_quantized:.6f}")
        print(f"  Difference:      {ll_diff:.6f}")
    
    # Save likelihood results
    likelihood_path = os.path.join(args.output_dir, 'likelihood_analysis.jsonl')
    with open(likelihood_path, 'w') as f:
        for result in likelihood_results:
            f.write(json.dumps(result) + '\n')
    print(f"\nLikelihood results saved to: {likelihood_path}")
    
    # Compute and save summary statistics
    print("\n" + "="*80)
    print("LIKELIHOOD ANALYSIS SUMMARY")
    print("="*80)
    
    ll_precise_vals = [r['log_likelihood_precise'] for r in likelihood_results]
    ll_quantized_vals = [r['log_likelihood_quantized'] for r in likelihood_results]
    ll_diff_vals = [r['log_likelihood_diff'] for r in likelihood_results]
    
    summary = {
        'num_samples': len(likelihood_results),
        'log_likelihood_precise': {
            'mean': float(np.mean(ll_precise_vals)),
            'std': float(np.std(ll_precise_vals)),
            'min': float(np.min(ll_precise_vals)),
            'max': float(np.max(ll_precise_vals)),
        },
        'log_likelihood_quantized': {
            'mean': float(np.mean(ll_quantized_vals)),
            'std': float(np.std(ll_quantized_vals)),
            'min': float(np.min(ll_quantized_vals)),
            'max': float(np.max(ll_quantized_vals)),
        },
        'log_likelihood_difference': {
            'mean': float(np.mean(ll_diff_vals)),
            'std': float(np.std(ll_diff_vals)),
            'min': float(np.min(ll_diff_vals)),
            'max': float(np.max(ll_diff_vals)),
        },
    }
    
    summary_path = os.path.join(args.output_dir, 'likelihood_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nPrecise Model Log-Likelihood:")
    print(f"  Mean: {summary['log_likelihood_precise']['mean']:.6f} ± {summary['log_likelihood_precise']['std']:.6f}")
    print(f"  Range: [{summary['log_likelihood_precise']['min']:.6f}, {summary['log_likelihood_precise']['max']:.6f}]")
    
    print(f"\nQuantized Model Log-Likelihood:")
    print(f"  Mean: {summary['log_likelihood_quantized']['mean']:.6f} ± {summary['log_likelihood_quantized']['std']:.6f}")
    print(f"  Range: [{summary['log_likelihood_quantized']['min']:.6f}, {summary['log_likelihood_quantized']['max']:.6f}]")
    
    print(f"\nLog-Likelihood Difference (Precise - Quantized):")
    print(f"  Mean: {summary['log_likelihood_difference']['mean']:.6f} ± {summary['log_likelihood_difference']['std']:.6f}")
    print(f"  Range: [{summary['log_likelihood_difference']['min']:.6f}, {summary['log_likelihood_difference']['max']:.6f}]")
    
    print(f"\nSummary statistics saved to: {summary_path}")
    print("="*80)
    
    print(f"\nAll results saved to: {args.output_dir}")
    print("\nOutput files:")
    print(f"  - {likelihood_path}   (detailed likelihood per sample)")
    print(f"  - {summary_path}      (summary statistics)")
    print(f"  - {generated_path}    (generated text)")
    print(f"  - {config_path}       (configuration)")


if __name__ == '__main__':
    main()