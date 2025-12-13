import sys
sys.path.append("./dquantize")
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
from tasks import load_dataset_examples
from models import load_model
from utils.postprocess import extract_final_answer_from_output, show_accuracy
from types import SimpleNamespace

print(torch.cuda.is_available())  # Should return True
print(torch.cuda.get_device_name(0))  # Should show "NVIDIA L40S"
print(torch.version.cuda)

from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler


# LLaDA model name -> default model path mapping
LLADA_MODEL_PATHS = {
    "llada": "GSAI-ML/LLaDA-8B-Instruct",              # Standard bfloat16 LLaDA
    # "llada-gptq": "path/to/llada-gptq",            # GPTQ quantized
    # "llada-awq": "path/to/llada-awq",              # AWQ quantized
    # "llada-bnb-4bit": "path/to/llada-bnb-4bit",    # bitsandbytes 4-bit quantization
    # "llada-bnb-8bit": "path/to/llada-bnb-8bit",    # bitsandbytes 8-bit quantization
}

# Adapted from LLaDA codebase
def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens
    
@torch.no_grad()
def _generate_single_batch(model, prompt, attention_mask=None, steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336, logits_eos_inf=False, 
             confidence_eos_eot_inf=False, batch_offset=0, enable_profiling=False, prof=None):
    '''
    Internal function to generate for a single batch of samples.
    
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (batch_size, L).
        attention_mask: Optional attention mask tensor of shape (batch_size, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The token id of [MASK] is 126336.
        logits_eos_inf: Whether to set the logits of EOS token to -inf.
        confidence_eos_eot_inf: Whether to set the confidence of EOS and EoT token to -inf.
        batch_offset: Offset for batch indices in logs (for tracking original sample indices).
        enable_profiling: Whether profiling is enabled.
        prof: Profiler instance if enabled.
    
    Returns:
        x: Generated tokens tensor of shape (batch_size, L + gen_length).
        logs: List of log entries for this batch.
        timing_stats: Dictionary of timing statistics for this batch.
    '''
    # Initialize timing statistics
    timing_stats = defaultdict(float)
    logs = []
    
    # define output with all masks
    # shape, fill value, dtype, device
    with record_function("initialization"):
        init_start = time.time()
        x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
        x[:, :prompt.shape[1]] = prompt.clone() # copy the prompt

        if attention_mask is not None:
            attention_mask = torch.cat([attention_mask, torch.ones((prompt.shape[0], gen_length), dtype=attention_mask.dtype, device=model.device)], dim=-1)

        prompt_index = (x != mask_id)

        assert gen_length % block_length == 0
        num_blocks = gen_length // block_length

        assert steps % num_blocks == 0
        steps = steps // num_blocks
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        timing_stats['initialization'] = time.time() - init_start

    # Outer progress bar for blocks
    block_pbar = tqdm(range(num_blocks), desc="Blocks", position=0, leave=True)
    for num_block in block_pbar:
        block_pbar.set_description(f"Block {num_block + 1}/{num_blocks}")
        
        with record_function(f"block_{num_block}_setup"):
            setup_start = time.time()
            block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
            num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            timing_stats['block_setup'] += time.time() - setup_start
        
        # Inner progress bar for steps within each block
        step_pbar = tqdm(range(steps), desc=f"  Steps (block {num_block + 1})", position=1, leave=False)
        for i in step_pbar:
            # Model inference
            with record_function(f"block_{num_block}_step_{i}_model_inference"):
                inference_start = time.time()
                mask_index = (x == mask_id)
                with torch.no_grad():
                    if cfg_scale > 0.:
                        un_x = x.clone()
                        un_x[prompt_index] = mask_id
                        x_ = torch.cat([x, un_x], dim=0)
                        if attention_mask is not None:
                            attention_mask_ = torch.cat([attention_mask, attention_mask], dim=0)
                        
                        logits = model(x_, attention_mask=attention_mask_).logits
                        logits, un_logits = torch.chunk(logits, 2, dim=0)
                        logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                    else:
                        logits = model(x, attention_mask=attention_mask).logits
                if logits_eos_inf:
                    logits[:, :, 126081] = -torch.inf
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                timing_stats['model_inference'] += time.time() - inference_start

            # Sampling and token selection
            with record_function(f"block_{num_block}_step_{i}_sampling"):
                sampling_start = time.time()
                
                logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1) # b, l
                
                if confidence_eos_eot_inf:
                    logits_with_noise[:, :, 126081] = logits[:, :, 126348] = -torch.inf

                if remasking == 'low_confidence':
                    p = F.softmax(logits, dim=-1)
                    x0_p = torch.squeeze(
                        torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
                elif remasking == 'random':
                    x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
                else:
                    raise NotImplementedError(remasking)

                x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, -np.inf)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                timing_stats['sampling'] += time.time() - sampling_start

            # Token selection
            with record_function(f"block_{num_block}_step_{i}_selection"):
                selection_start = time.time()
                
                transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                for j in range(confidence.shape[0]):
                    probs, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i] + 1) # for the first unselected token
                    assert probs.shape == (num_transfer_tokens[j, i] + 1,)
                    transfer_index[j, select_index[:-1]] = True
                    logs.append({
                        "block": num_block,
                        "iter": i,
                        "batch_idx": batch_offset + j,  # Use batch_offset to track original sample index
                        "probs": probs.cpu().tolist(),
                        "indices_in_x":  select_index.cpu().tolist(),
                        "num_selected_tokens": int(num_transfer_tokens[j, i].item()),
                    })
                x[transfer_index] = x0[transfer_index]
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                timing_stats['selection'] += time.time() - selection_start
                
        step_pbar.close()
    block_pbar.close()

    return x, logs, timing_stats

@torch.no_grad()
def generate(model, prompt, attention_mask=None, steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336, logits_eos_inf=False, 
             confidence_eos_eot_inf=False, output_dir=None, enable_profiling=False, profile_dir=None, batch_size=8):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (N, L) where N is the number of samples.
        attention_mask: Optional attention mask tensor of shape (N, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
        logits_eos_inf: Whether to set the logits of EOS token to -inf. See Appendix B.4 of LLaDA for details
        confidence_eos_eot_inf: Whether to set the confidence of EOS and EoT token to -inf. See Appendix B.4 of LLaDA for details
        output_dir: Directory to save output files.
        enable_profiling: Whether to enable PyTorch profiler
        profile_dir: Directory to save profiling results
        batch_size: Number of samples to process in each batch (default: 8)
    '''
    # Initialize timing statistics
    total_timing_stats = defaultdict(float)
    
    # Setup profiler if enabled
    prof = None
    if enable_profiling:
        print(f"Profiling enabled. Results will be saved to: {profile_dir}")
        prof = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            on_trace_ready=tensorboard_trace_handler(profile_dir)
        )
        prof.__enter__()
    
    generation_start = time.time()
    
    num_samples = prompt.shape[0]
    all_outputs = []
    all_logs = []
    
    # Process samples in batches
    num_batches = (num_samples + batch_size - 1) // batch_size
    batch_pbar = tqdm(range(num_batches), desc="Processing batches", position=0, leave=True)
    
    for batch_idx in batch_pbar:
        print(f"Batch {batch_idx + 1}/{num_batches}")
        batch_pbar.set_description(f"Batch {batch_idx + 1}/{num_batches}")
        
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_samples)
        
        # Extract batch
        batch_prompt = prompt[start_idx:end_idx]
        batch_attention_mask = attention_mask[start_idx:end_idx] if attention_mask is not None else None
        # Generate for this batch
        batch_output, batch_logs, batch_timing_stats = _generate_single_batch(
            model=model,
            prompt=batch_prompt,
            attention_mask=batch_attention_mask,
            steps=steps,
            gen_length=gen_length,
            block_length=block_length,
            temperature=temperature,
            cfg_scale=cfg_scale,
            remasking=remasking,
            mask_id=mask_id,
            logits_eos_inf=logits_eos_inf,
            confidence_eos_eot_inf=confidence_eos_eot_inf,
            batch_offset=start_idx,  # Track original sample indices
            enable_profiling=enable_profiling,
            prof=prof
        )
        
        all_outputs.append(batch_output)
        all_logs.extend(batch_logs)
        
        # Aggregate timing stats
        for key, value in batch_timing_stats.items():
            total_timing_stats[key] += value
    
    batch_pbar.close()
    
    # Concatenate all outputs
    x = torch.cat(all_outputs, dim=0)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total_generation_time = time.time() - generation_start

    # Stop profiler if enabled
    if prof is not None:
        prof.__exit__(None, None, None)
        
        # Export key averages table
        print("\n" + "="*80)
        print("PROFILER SUMMARY (Top 20 operations by CUDA time)")
        print("="*80)
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
        
        # Save detailed table to file
        table_path = os.path.join(profile_dir, "profile_summary.txt")
        with open(table_path, "w") as f:
            f.write(prof.key_averages().table(sort_by="cuda_time_total"))
        print(f"\nFull profiler summary saved to: {table_path}")

    # Calculate percentages
    print("\n" + "="*80)
    print("TIMING BREAKDOWN")
    print("="*80)
    print(f"Total generation time: {total_generation_time:.4f}s")
    print(f"\nDetailed breakdown:")
    print(f"  Initialization:    {total_timing_stats['initialization']:8.4f}s ({total_timing_stats['initialization']/total_generation_time*100:5.2f}%)")
    print(f"  Block setup:       {total_timing_stats['block_setup']:8.4f}s ({total_timing_stats['block_setup']/total_generation_time*100:5.2f}%)")
    print(f"  Model inference:   {total_timing_stats['model_inference']:8.4f}s ({total_timing_stats['model_inference']/total_generation_time*100:5.2f}%)")
    print(f"  Sampling:          {total_timing_stats['sampling']:8.4f}s ({total_timing_stats['sampling']/total_generation_time*100:5.2f}%)")
    print(f"  Token selection:   {total_timing_stats['selection']:8.4f}s ({total_timing_stats['selection']/total_generation_time*100:5.2f}%)")
    
    overhead = total_generation_time - sum(total_timing_stats.values())
    print(f"  Other/Overhead:    {overhead:8.4f}s ({overhead/total_generation_time*100:5.2f}%)")
    print("="*80)

    # Save timing stats
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        probs_path = os.path.join(output_dir, "probs.jsonl")
        timing_path = os.path.join(output_dir, "timing_stats.json")
    else:
        probs_path = "probs.jsonl"
        timing_path = "timing_stats.json"
    
    with open(probs_path, "w") as f:
        for row in all_logs:
            f.write(json.dumps(row) + "\n")
    
    # Save timing statistics
    timing_summary = {
        "total_generation_time": total_generation_time,
        "breakdown": dict(total_timing_stats),
        "percentages": {
            "initialization": total_timing_stats['initialization']/total_generation_time*100,
            "block_setup": total_timing_stats['block_setup']/total_generation_time*100,
            "model_inference": total_timing_stats['model_inference']/total_generation_time*100,
            "sampling": total_timing_stats['sampling']/total_generation_time*100,
            "selection": total_timing_stats['selection']/total_generation_time*100,
            "overhead": overhead/total_generation_time*100
        }
    }
    with open(timing_path, "w") as f:
        json.dump(timing_summary, f, indent=2)
    print(f"\nTiming statistics saved to: {timing_path}")

    return x


def main():
    parser = argparse.ArgumentParser(description='Run LLaDA generation')
    
    # Model arguments
    # parser.add_argument('--model', type=str, default='GSAI-ML/LLaDA-8B-Instruct',
    #                     help='Model name or path to load from HuggingFace')
    parser.add_argument('--name', type=str, default='llada', help="Model name")
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, choices=['toy', 'wikitext2', 'gsm8k'], default='toy',
                        help='Dataset to load: "toy" or "wikitext2"')
    parser.add_argument('--max_examples', type=int, default=3,
                        help='Maximum number of examples to process')
    parser.add_argument('--max_length', type=int, default=0,
                        help='Maximum length of each example in tokens (0 for no limit)')
    parser.add_argument('--min_length', type=int, default=0,
                        help='Minimum length of each example in tokens (0 for no limit)')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory to save results and config')
    
    # Profiling arguments
    parser.add_argument('--profile', action='store_true',
                        help='Enable PyTorch profiling with TensorBoard')
    
    # Generation arguments
    parser.add_argument('--steps', type=int, default=128,
                        help='Sampling steps')
    parser.add_argument('--gen_length', type=int, default=128,
                        help='Generated answer length')
    parser.add_argument('--block_length', type=int, default=32,
                        help='Block length for semi-autoregressive remasking')
    parser.add_argument('--temperature', type=float, default=0.,
                        help='Categorical distribution sampling temperature')
    parser.add_argument('--cfg_scale', type=float, default=0.,
                        help='Unsupervised classifier-free guidance scale')
    parser.add_argument('--remasking', type=str, choices=['low_confidence', 'random'], default='low_confidence',
                        help='Remasking strategy: "low_confidence" or "random"')
    parser.add_argument('--logits_eos_inf', action='store_true',
                        help='Set logits of EOS token to -inf')
    parser.add_argument('--confidence_eos_eot_inf', action='store_true',
                        help='Set confidence of EOS and EoT token to -inf')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Number of samples to process in each batch (default: 8)')
    
    args = parser.parse_args()

    device = 'cuda'
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup profiling directory if enabled
    profile_dir = None
    if args.profile:
        profile_dir = os.path.join(args.output_dir, 'profile')
        os.makedirs(profile_dir, exist_ok=True)
    
    # Save configuration
    config = config = SimpleNamespace(
        name=args.name,
        path=LLADA_MODEL_PATHS[args.name],
        dataset=args.dataset,
        max_examples=args.max_examples,
        max_length=args.max_length,
        min_length=args.min_length,
        steps=args.steps,
        gen_length=args.gen_length,
        block_length=args.block_length,
        temperature=args.temperature,
        cfg_scale=args.cfg_scale,
        remasking=args.remasking,
        logits_eos_inf=args.logits_eos_inf,
        confidence_eos_eot_inf=args.confidence_eos_eot_inf,
        profile=args.profile,
        batch_size=args.batch_size,
        torch_dtype = 'bfloat16', # change it if required
        trust_remote_code = True, # change to True if loading custom models
        device = device,
    )
    
    config_path = os.path.join(args.output_dir, 'call_config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    

    print(f"Loading model: {config.name}")
    model_cls = load_model(config)
    model = model_cls._model
    tokenizer = model_cls._tokenizer

    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    data = load_dataset_examples(args.dataset, tokenizer, args.max_examples, args.max_length, args.min_length)
    raw_prompts = [d['question'] for d in data]
    targets = [d['target'] for d in data]
    # For instruct models, apply chat template
    if 'instruct' in LLADA_MODEL_PATHS[args.name].lower():
        messages = [{"role": "user", "content": prompt} for prompt in raw_prompts]
        prompts = [tokenizer.apply_chat_template([message], add_generation_prompt=True, tokenize=False) for message in messages]
    else:
        prompts = raw_prompts

    print("Tokenizing prompts...abc")
    encoded_outputs = tokenizer(
        prompts,
        add_special_tokens=False,
        padding=True,
        return_tensors="pt"
    )
    input_ids = encoded_outputs['input_ids'].to(device)
    attention_mask = encoded_outputs['attention_mask'].to(device)
    
    print(f"Starting generation (steps={args.steps}, gen_length={args.gen_length}, block_length={args.block_length})...")

    out = generate(
        model, 
        input_ids, 
        attention_mask, 
        steps=args.steps, 
        gen_length=args.gen_length, 
        block_length=args.block_length, 
        temperature=args.temperature, 
        cfg_scale=args.cfg_scale, 
        remasking=args.remasking,
        logits_eos_inf=args.logits_eos_inf,
        confidence_eos_eot_inf=args.confidence_eos_eot_inf,
        output_dir=args.output_dir,
        enable_profiling=args.profile,
        profile_dir=profile_dir,
        batch_size=args.batch_size
    )
    
    print("\nGeneration complete! Decoding outputs...")
    output = tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)
    print("\nGenerated outputs:")
    for i, o in enumerate(output):
        print(f"\nExample {i + 1}:")
        print(o)
        print('-' * 50)
    print(f"\nResults saved to: {args.output_dir}")


    # Eval logic (for GSM8k)
    final_outputs = [extract_final_answer_from_output(o) for o in output]
    # Exact Match
    show_accuracy(final_outputs, targets) 


    if args.profile:
        print(f"\nTo view profiling results in TensorBoard, run:")
        print(f"  tensorboard --logdir={profile_dir}")

if __name__ == '__main__':
    main()