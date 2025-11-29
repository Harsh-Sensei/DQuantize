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
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple

# Import apply_awq from the AWQ quantization module
import sys
awq_path = os.path.join(os.path.dirname(__file__), '..', 'third_party', 'QDLM', 'llm-awq')
if awq_path not in sys.path:
    sys.path.insert(0, awq_path)
from awq.quantize.pre_quant import apply_awq
from awq.quantize.quantizer import real_quantize_model_weight, pseudo_quantize_model_weight
import re

print(torch.cuda.is_available())  # Should return True
print(torch.cuda.get_device_name(0))  # Should show "NVIDIA L40S"
print(torch.version.cuda)


from transformers import AutoModel, AutoTokenizer
from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler

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


def compute_kl_divergence(p_logits, q_logits):
    """
    Compute KL divergence between two probability distributions.
    
    Args:
        p_logits: Logits from the first model (precise)
        q_logits: Logits from the second model (quantized)
    
    Returns:
        KL divergence value
    """
    p = F.softmax(p_logits, dim=-1)
    q = F.softmax(q_logits, dim=-1)
    
    # KL(P || Q) = sum(P * log(P/Q))
    kl_div = F.kl_div(q.log(), p, reduction='none').sum(dim=-1)
    return kl_div


def compute_jaccard_similarity(set1, set2):
    """
    Compute Jaccard similarity between two sets.
    
    Args:
        set1: First set of indices
        set2: Second set of indices
    
    Returns:
        Jaccard similarity (intersection / union)
    """
    set1 = set(set1)
    set2 = set(set2)
    
    if len(set1) == 0 and len(set2) == 0:
        return 1.0
    
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    if union == 0:
        return 0.0
    
    return intersection / union


def plot_quantization_metrics_per_batch(df, output_dir='plots', mean_gen_length=None, std_gen_length=None):
    """
    Plot KL divergence and Jaccard similarity for each batch separately.
    
    Args:
        df: DataFrame with analysis logs
        output_dir: Directory to save plots
        mean_gen_length: Mean generation length (for horizontal line)
        std_gen_length: Std dev of generation length (for horizontal lines)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Group by batch_idx and plot each metric
    for batch_idx, subdf in df.groupby("batch_idx"):
        subdf = subdf.sort_values(["block", "iter"])
        # Compute x-axis indices (integer positions)
        x_vals = range(len(subdf))

        # Find start of each block (where block value changes)
        block_starts = []
        block_labels = []
        prev_block = None
        
        for i, (idx, row) in enumerate(subdf.iterrows()):
            if row["block"] != prev_block:
                block_starts.append(i)
                block_labels.append(str(int(row["block"])))
                prev_block = row["block"]
        
        # Calculate midpoints of each block for label positioning
        block_midpoints = []
        for i in range(len(block_starts)):
            start = block_starts[i]
            end = block_starts[i + 1] if i + 1 < len(block_starts) else len(subdf)
            midpoint = (start + end) / 2
            block_midpoints.append(midpoint)

        def add_block_markers_and_labels(ax):
            """Add vertical dashed lines at each new block start and label x-axis with block IDs."""
            # Add vertical lines at block boundaries
            for idx in block_starts[1:]:  # Skip first one (it's at 0)
                ax.axvline(x=idx, linestyle="--", alpha=0.5, color='gray')
            
            # Set x-axis ticks at block midpoints with block labels
            ax.set_xticks(block_midpoints)
            ax.set_xticklabels(block_labels)
            ax.set_xlabel(f"Block ID ({subdf['iter'].max()+1} Iterations each)")

        # --- KL Divergence ---
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x_vals, subdf["avg_kl_divergence"], color='red', linewidth=2)
        add_block_markers_and_labels(ax)
        
        # Add horizontal lines for generation length statistics
        if mean_gen_length is not None and std_gen_length is not None:
            ax.axhline(y=mean_gen_length, color='green', linestyle=':', linewidth=2, alpha=0.7, label=f'Mean Gen Length: {mean_gen_length:.1f}')
            ax.axhline(y=mean_gen_length + std_gen_length, color='green', linestyle=':', linewidth=1.5, alpha=0.5, label=f'Mean ± Std: {mean_gen_length + std_gen_length:.1f}')
            ax.axhline(y=mean_gen_length - std_gen_length, color='green', linestyle=':', linewidth=1.5, alpha=0.5, label=f'Mean ± Std: {mean_gen_length - std_gen_length:.1f}')
            ax.legend()
        
        ax.set_title(f"Batch {batch_idx}: KL Divergence (Precise || Quantized)")
        ax.set_ylabel("KL Divergence")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, f"batch_{batch_idx}_kl_divergence.png"), dpi=150)
        plt.close(fig)

        # --- Jaccard Similarity ---
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x_vals, subdf["jaccard_similarity"], color='blue', linewidth=2)
        add_block_markers_and_labels(ax)
        
        # Add horizontal lines for generation length statistics
        if mean_gen_length is not None and std_gen_length is not None:
            ax.axhline(y=mean_gen_length, color='green', linestyle=':', linewidth=2, alpha=0.7, label=f'Mean Gen Length: {mean_gen_length:.1f}')
            ax.axhline(y=mean_gen_length + std_gen_length, color='green', linestyle=':', linewidth=1.5, alpha=0.5, label=f'Mean ± Std: {mean_gen_length + std_gen_length:.1f}')
            ax.axhline(y=mean_gen_length - std_gen_length, color='green', linestyle=':', linewidth=1.5, alpha=0.5, label=f'Mean ± Std: {mean_gen_length - std_gen_length:.1f}')
            ax.legend()
        
        ax.set_title(f"Batch {batch_idx}: Jaccard Similarity of Selected Tokens")
        ax.set_ylabel("Jaccard Similarity")
        ax.set_ylim([0, 1.05])  # Jaccard is between 0 and 1
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, f"batch_{batch_idx}_jaccard_similarity.png"), dpi=150)
        plt.close(fig)


def plot_aggregated_quantization_metrics(df, output_dir='plots', mean_gen_length=None, std_gen_length=None):
    """
    Plot aggregated statistics (mean and variance) across all batches for KL divergence and Jaccard similarity.
    
    Args:
        df: DataFrame with analysis logs
        output_dir: Directory to save plots
        mean_gen_length: Mean generation length (for horizontal line)
        std_gen_length: Std dev of generation length (for horizontal lines)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Sort by block and iter to ensure proper ordering
    df = df.sort_values(["block", "iter"]).reset_index(drop=True)
    
    # Group by block and iter to aggregate across batches
    aggregated = df.groupby(["block", "iter"]).agg({
        "avg_kl_divergence": ["mean", "std", "var"],
        "jaccard_similarity": ["mean", "std", "var"]
    }).reset_index()
    
    # Flatten column names
    aggregated.columns = ['_'.join(col).strip('_') if col[1] else col[0] for col in aggregated.columns.values]
    
    # Compute x-axis indices
    x_vals = range(len(aggregated))
    
    # Find start of each block (where block value changes)
    block_starts = []
    block_labels = []
    prev_block = None
    
    for i, row in aggregated.iterrows():
        if row["block"] != prev_block:
            block_starts.append(i)
            block_labels.append(str(int(row["block"])))
            prev_block = row["block"]
    
    # Calculate midpoints of each block for label positioning
    block_midpoints = []
    for i in range(len(block_starts)):
        start = block_starts[i]
        end = block_starts[i + 1] if i + 1 < len(block_starts) else len(aggregated)
        midpoint = (start + end) / 2
        block_midpoints.append(midpoint)
    
    def add_block_markers_and_labels(ax):
        """Add vertical dashed lines at each new block start and label x-axis with block IDs."""
        # Add vertical lines at block boundaries
        for idx in block_starts[1:]:  # Skip first one (it's at 0)
            ax.axvline(x=idx, linestyle="--", alpha=0.5, color='gray')
        
        # Set x-axis ticks at block midpoints with block labels
        ax.set_xticks(block_midpoints)
        ax.set_xticklabels(block_labels)
        max_iter = aggregated['iter'].max() + 1
        ax.set_xlabel(f"Block ID ({max_iter} Iterations each)")
    
    # --- KL Divergence (aggregated) ---
    fig, ax = plt.subplots(figsize=(12, 6))
    mean_vals = aggregated["avg_kl_divergence_mean"]
    std_vals = aggregated["avg_kl_divergence_std"].fillna(0)  # Fill NaN with 0
    
    ax.plot(x_vals, mean_vals, label='Mean', linewidth=2, color='red')
    ax.fill_between(x_vals, 
                    mean_vals - std_vals, 
                    mean_vals + std_vals, 
                    alpha=0.3, label='±1 Std Dev', color='red')
    
    # Add horizontal lines for generation length statistics
    if mean_gen_length is not None and std_gen_length is not None:
        ax.axhline(y=mean_gen_length, color='green', linestyle=':', linewidth=2, alpha=0.7, label=f'Mean Gen Length: {mean_gen_length:.1f}')
        ax.axhline(y=mean_gen_length + std_gen_length, color='green', linestyle=':', linewidth=1.5, alpha=0.5)
        ax.axhline(y=mean_gen_length - std_gen_length, color='green', linestyle=':', linewidth=1.5, alpha=0.5)
    
    add_block_markers_and_labels(ax)
    ax.set_title("Aggregated: KL Divergence (Precise || Quantized) - Mean ± Std Dev across batches")
    ax.set_ylabel("KL Divergence")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "aggregated_kl_divergence.png"), dpi=150)
    plt.close(fig)
    
    # --- Jaccard Similarity (aggregated) ---
    fig, ax = plt.subplots(figsize=(12, 6))
    mean_vals = aggregated["jaccard_similarity_mean"]
    std_vals = aggregated["jaccard_similarity_std"].fillna(0)
    
    ax.plot(x_vals, mean_vals, label='Mean', linewidth=2, color='blue')
    ax.fill_between(x_vals, 
                    mean_vals - std_vals, 
                    mean_vals + std_vals, 
                    alpha=0.3, label='±1 Std Dev', color='blue')
    
    # Add horizontal lines for generation length statistics
    if mean_gen_length is not None and std_gen_length is not None:
        ax.axhline(y=mean_gen_length, color='green', linestyle=':', linewidth=2, alpha=0.7, label=f'Mean Gen Length: {mean_gen_length:.1f}')
        ax.axhline(y=mean_gen_length + std_gen_length, color='green', linestyle=':', linewidth=1.5, alpha=0.5)
        ax.axhline(y=mean_gen_length - std_gen_length, color='green', linestyle=':', linewidth=1.5, alpha=0.5)
    
    add_block_markers_and_labels(ax)
    ax.set_title("Aggregated: Jaccard Similarity of Selected Tokens - Mean ± Std Dev across batches")
    ax.set_ylabel("Jaccard Similarity")
    ax.set_ylim([0, 1.05])
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "aggregated_jaccard_similarity.png"), dpi=150)
    plt.close(fig)
    
    # --- Combined plot: Both metrics ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # KL Divergence subplot
    mean_vals = aggregated["avg_kl_divergence_mean"]
    std_vals = aggregated["avg_kl_divergence_std"].fillna(0)
    ax1.plot(x_vals, mean_vals, label='Mean', linewidth=2, color='red')
    ax1.fill_between(x_vals, 
                     mean_vals - std_vals, 
                     mean_vals + std_vals, 
                     alpha=0.3, label='±1 Std Dev', color='red')
    
    # Add horizontal lines for generation length statistics
    if mean_gen_length is not None and std_gen_length is not None:
        ax1.axhline(y=mean_gen_length, color='green', linestyle=':', linewidth=2, alpha=0.7, label=f'Mean Gen Length: {mean_gen_length:.1f}')
        ax1.axhline(y=mean_gen_length + std_gen_length, color='green', linestyle=':', linewidth=1.5, alpha=0.5)
        ax1.axhline(y=mean_gen_length - std_gen_length, color='green', linestyle=':', linewidth=1.5, alpha=0.5)
    
    for idx in block_starts[1:]:
        ax1.axvline(x=idx, linestyle="--", alpha=0.5, color='gray')
    ax1.set_ylabel("KL Divergence")
    ax1.set_title("KL Divergence (Precise || Quantized)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Jaccard Similarity subplot
    mean_vals = aggregated["jaccard_similarity_mean"]
    std_vals = aggregated["jaccard_similarity_std"].fillna(0)
    ax2.plot(x_vals, mean_vals, label='Mean', linewidth=2, color='blue')
    ax2.fill_between(x_vals, 
                     mean_vals - std_vals, 
                     mean_vals + std_vals, 
                     alpha=0.3, label='±1 Std Dev', color='blue')
    
    # Add horizontal lines for generation length statistics
    if mean_gen_length is not None and std_gen_length is not None:
        ax2.axhline(y=mean_gen_length, color='green', linestyle=':', linewidth=2, alpha=0.7, label=f'Mean Gen Length: {mean_gen_length:.1f}')
        ax2.axhline(y=mean_gen_length + std_gen_length, color='green', linestyle=':', linewidth=1.5, alpha=0.5)
        ax2.axhline(y=mean_gen_length - std_gen_length, color='green', linestyle=':', linewidth=1.5, alpha=0.5)
    
    for idx in block_starts[1:]:
        ax2.axvline(x=idx, linestyle="--", alpha=0.5, color='gray')
    ax2.set_xticks(block_midpoints)
    ax2.set_xticklabels(block_labels)
    max_iter = aggregated['iter'].max() + 1
    ax2.set_xlabel(f"Block ID ({max_iter} Iterations each)")
    ax2.set_ylabel("Jaccard Similarity")
    ax2.set_ylim([0, 1.05])
    ax2.set_title("Jaccard Similarity of Selected Tokens")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "aggregated_combined_metrics.png"), dpi=150)
    plt.close(fig)
    
    # Print summary statistics
    print("\n" + "="*80)
    print("AGGREGATED PLOTTING STATISTICS")
    print("="*80)
    print(f"Number of batches: {df['batch_idx'].nunique()}")
    print(f"Number of timesteps: {len(aggregated)}")
    print(f"\nKL Divergence:")
    print(f"  Mean across all steps: {aggregated['avg_kl_divergence_mean'].mean():.6f} ± {aggregated['avg_kl_divergence_mean'].std():.6f}")
    print(f"  Avg Std Dev: {aggregated['avg_kl_divergence_std'].mean():.6f}")
    print(f"\nJaccard Similarity:")
    print(f"  Mean across all steps: {aggregated['jaccard_similarity_mean'].mean():.6f} ± {aggregated['jaccard_similarity_mean'].std():.6f}")
    print(f"  Avg Std Dev: {aggregated['jaccard_similarity_std'].mean():.6f}")
    print("="*80)

    
@torch.no_grad()
def _generate_single_batch_with_quantization_analysis(
        model, quantized_model, prompt, attention_mask=None, steps=128, gen_length=128, 
        block_length=128, temperature=0., cfg_scale=0., remasking='low_confidence', 
        mask_id=126336, logits_eos_inf=False, confidence_eos_eot_inf=False, 
        batch_offset=0, enable_profiling=False, prof=None, device_quantized='cuda:1'):
    '''
    Internal function to generate for a single batch of samples with quantization analysis.
    
    Args:
        model: Precise mask predictor.
        quantized_model: Quantized mask predictor.
        prompt: A tensor of shape (batch_size, L).
        attention_mask: Optional attention mask tensor of shape (batch_size, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. Must be 'low_confidence' for this analysis.
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
        analysis_logs: List of quantization analysis entries for this batch.
    '''
    # Validate remasking strategy
    if remasking != 'low_confidence':
        raise ValueError("This analysis script only supports 'low_confidence' remasking strategy")
    
    # Initialize timing statistics
    timing_stats = defaultdict(float)
    logs = []
    analysis_logs = []
    
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
            # Model inference - PRECISE MODEL
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
                        
                        logits_precise = model(x_, attention_mask=attention_mask_).logits
                        logits_precise, un_logits = torch.chunk(logits_precise, 2, dim=0)
                        logits_precise = un_logits + (cfg_scale + 1) * (logits_precise - un_logits)
                    else:
                        logits_precise = model(x, attention_mask=attention_mask).logits

                if logits_eos_inf:
                    logits_precise[:, :, 126081] = -torch.inf
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                timing_stats['model_inference'] += time.time() - inference_start

            # Model inference - QUANTIZED MODEL
            with record_function(f"block_{num_block}_step_{i}_quantized_model_inference"):
                quant_inference_start = time.time()
                with torch.no_grad():
                    # Move inputs to quantized model's device
                    x_quantized = x.to(device_quantized)
                    attention_mask_quantized = attention_mask.to(device_quantized) if attention_mask is not None else None
                    prompt_index_quantized = prompt_index.to(device_quantized)
                    
                    if cfg_scale > 0.:
                        un_x = x_quantized.clone()
                        un_x[prompt_index_quantized] = mask_id
                        x_ = torch.cat([x_quantized, un_x], dim=0)
                        if attention_mask_quantized is not None:
                            attention_mask_ = torch.cat([attention_mask_quantized, attention_mask_quantized], dim=0)
                        
                        logits_quantized = quantized_model(x_, attention_mask=attention_mask_).logits
                        logits_quantized, un_logits_q = torch.chunk(logits_quantized, 2, dim=0)
                        logits_quantized = un_logits_q + (cfg_scale + 1) * (logits_quantized - un_logits_q)
                    else:
                        logits_quantized = quantized_model(x_quantized, attention_mask=attention_mask_quantized).logits
                    
                    # Move logits back to precise model's device for comparison
                    logits_quantized = logits_quantized.to(model.device)

                if logits_eos_inf:
                    logits_quantized[:, :, 126081] = -torch.inf
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                timing_stats['quantized_model_inference'] += time.time() - quant_inference_start

            # Compute KL divergence for masked positions
            with record_function(f"block_{num_block}_step_{i}_kl_computation"):
                kl_start = time.time()
                
                # Compute KL divergence for all positions
                kl_divs = compute_kl_divergence(logits_precise, logits_quantized)  # shape: (batch, seq_len)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                timing_stats['kl_computation'] += time.time() - kl_start

            # Sampling and token selection - PRECISE MODEL
            with record_function(f"block_{num_block}_step_{i}_sampling"):
                sampling_start = time.time()
                
                logits_with_noise = add_gumbel_noise(logits_precise, temperature=temperature)
                x0_precise = torch.argmax(logits_with_noise, dim=-1) # b, l

                if confidence_eos_eot_inf:
                    logits_with_noise[:, :, 126081] = logits_precise[:, :, 126348] = -torch.inf

                # Compute confidence for precise model
                p_precise = F.softmax(logits_precise, dim=-1)
                x0_p_precise = torch.squeeze(
                    torch.gather(p_precise, dim=-1, index=torch.unsqueeze(x0_precise, -1)), -1) # b, l

                x0_p_precise[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

                x0_precise = torch.where(mask_index, x0_precise, x)
                confidence_precise = torch.where(mask_index, x0_p_precise, -np.inf)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                timing_stats['sampling'] += time.time() - sampling_start

            # Sampling for QUANTIZED MODEL
            with record_function(f"block_{num_block}_step_{i}_quantized_sampling"):
                quant_sampling_start = time.time()
                
                # logits_quantized is already on model.device from previous step
                logits_with_noise_q = add_gumbel_noise(logits_quantized, temperature=temperature)
                x0_quantized = torch.argmax(logits_with_noise_q, dim=-1)
                
                if confidence_eos_eot_inf:
                    logits_with_noise_q[:, :, 126081] = logits_quantized[:, :, 126348] = -torch.inf

                # Compute confidence for quantized model
                p_quantized = F.softmax(logits_quantized, dim=-1)
                x0_p_quantized = torch.squeeze(
                    torch.gather(p_quantized, dim=-1, index=torch.unsqueeze(x0_quantized, -1)), -1)

                x0_p_quantized[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

                x0_quantized = torch.where(mask_index, x0_quantized, x)
                confidence_quantized = torch.where(mask_index, x0_p_quantized, -np.inf)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                timing_stats['quantized_sampling'] += time.time() - quant_sampling_start

            # Token selection and Jaccard similarity computation
            with record_function(f"block_{num_block}_step_{i}_selection"):
                selection_start = time.time()
                
                transfer_index = torch.zeros_like(x0_precise, dtype=torch.bool, device=x0_precise.device)
                
                for j in range(confidence_precise.shape[0]):
                    # Compute average KL divergence for this sample's masked positions
                    masked_kl_divs_j = kl_divs[j][mask_index[j]]
                    avg_kl_div_j = masked_kl_divs_j.mean().item() if masked_kl_divs_j.numel() > 0 else 0.0
                    
                    # Select tokens for precise model
                    probs_precise, select_index_precise = torch.topk(
                        confidence_precise[j], k=num_transfer_tokens[j, i] + 1)
                    assert probs_precise.shape == (num_transfer_tokens[j, i] + 1,)
                    transfer_index[j, select_index_precise[:-1]] = True
                    
                    # Select tokens for quantized model
                    probs_quantized, select_index_quantized = torch.topk(
                        confidence_quantized[j], k=num_transfer_tokens[j, i] + 1)
                    
                    # Compute Jaccard similarity between selected indices
                    selected_indices_precise = select_index_precise[:-1].cpu().tolist()
                    selected_indices_quantized = select_index_quantized[:-1].cpu().tolist()
                    
                    jaccard_sim = compute_jaccard_similarity(
                        selected_indices_precise, 
                        selected_indices_quantized
                    )
                    
                    # Log analysis data
                    analysis_logs.append({
                        "block": num_block,
                        "iter": i,
                        "batch_idx": batch_offset + j,
                        "avg_kl_divergence": avg_kl_div_j,
                        "jaccard_similarity": jaccard_sim,
                        "num_selected_tokens": int(num_transfer_tokens[j, i].item()),
                        "selected_indices_precise": selected_indices_precise,
                        "selected_indices_quantized": selected_indices_quantized,
                    })
                    
                    logs.append({
                        "block": num_block,
                        "iter": i,
                        "batch_idx": batch_offset + j,
                        "probs": probs_precise.cpu().tolist(),
                        "indices_in_x":  select_index_precise.cpu().tolist(),
                        "num_selected_tokens": int(num_transfer_tokens[j, i].item()),
                    })
                
                # Use precise model's selection for continuing generation
                x[transfer_index] = x0_precise[transfer_index]
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                timing_stats['selection'] += time.time() - selection_start
                
        step_pbar.close()
    block_pbar.close()

    return x, logs, timing_stats, analysis_logs


@torch.no_grad()
def generate_with_quantization_analysis(
        model, quantized_model, prompt, attention_mask=None, steps=128, gen_length=128, 
        block_length=128, temperature=0., cfg_scale=0., remasking='low_confidence', 
        mask_id=126336, logits_eos_inf=False, confidence_eos_eot_inf=False, 
        output_dir=None, enable_profiling=False, profile_dir=None, batch_size=8,
        device_quantized='cuda:1'):
    '''
    Args:
        model: Precise mask predictor.
        quantized_model: Quantized mask predictor.
        prompt: A tensor of shape (N, L) where N is the number of samples.
        attention_mask: Optional attention mask tensor of shape (N, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. Must be 'low_confidence'.
        mask_id: The toke id of [MASK] is 126336.
        logits_eos_inf: Whether to set the logits of EOS token to -inf.
        confidence_eos_eot_inf: Whether to set the confidence of EOS and EoT token to -inf.
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
    all_analysis_logs = []
    
    # Process samples in batches
    num_batches = (num_samples + batch_size - 1) // batch_size
    batch_pbar = tqdm(range(num_batches), desc="Processing batches", position=0, leave=True)
    
    for batch_idx in batch_pbar:
        batch_pbar.set_description(f"Batch {batch_idx + 1}/{num_batches}")
        
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_samples)
        
        # Extract batch
        batch_prompt = prompt[start_idx:end_idx]
        batch_attention_mask = attention_mask[start_idx:end_idx] if attention_mask is not None else None
        
        # Generate for this batch with quantization analysis
        batch_output, batch_logs, batch_timing_stats, batch_analysis_logs = _generate_single_batch_with_quantization_analysis(
            model=model,
            quantized_model=quantized_model,
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
            batch_offset=start_idx,
            enable_profiling=enable_profiling,
            prof=prof,
            device_quantized=device_quantized
        )
        
        all_outputs.append(batch_output)
        all_logs.extend(batch_logs)
        all_analysis_logs.extend(batch_analysis_logs)
        
        # Aggregate timing stats
        for key, value in batch_timing_stats.items():
            total_timing_stats[key] += value
    
    batch_pbar.close()
    
    # Concatenate all outputs
    x = torch.cat(all_outputs, dim=0)
    
    # Count generated tokens (excluding masks and EOS) for each sample
    eos_id = 126081
    prompt_length = prompt.shape[1]
    generated_portion = x[:, prompt_length:]  # Shape: (num_samples, gen_length)
    
    # Count tokens that are not mask_id and not EOS for each sample
    is_valid_token = (generated_portion != mask_id) & (generated_portion != eos_id)
    generation_lengths = is_valid_token.sum(dim=1).cpu().numpy()  # Shape: (num_samples,)
    
    # Calculate statistics
    mean_gen_length = float(np.mean(generation_lengths))
    std_gen_length = float(np.std(generation_lengths))
    
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

    # Calculate timing statistics
    print("\n" + "="*80)
    print("TIMING BREAKDOWN")
    print("="*80)
    print(f"Total generation time: {total_generation_time:.4f}s")
    print(f"\nDetailed breakdown:")
    print(f"  Initialization:            {total_timing_stats['initialization']:8.4f}s ({total_timing_stats['initialization']/total_generation_time*100:5.2f}%)")
    print(f"  Block setup:               {total_timing_stats['block_setup']:8.4f}s ({total_timing_stats['block_setup']/total_generation_time*100:5.2f}%)")
    print(f"  Precise model inference:   {total_timing_stats['model_inference']:8.4f}s ({total_timing_stats['model_inference']/total_generation_time*100:5.2f}%)")
    print(f"  Quantized model inference: {total_timing_stats['quantized_model_inference']:8.4f}s ({total_timing_stats['quantized_model_inference']/total_generation_time*100:5.2f}%)")
    print(f"  KL computation:            {total_timing_stats['kl_computation']:8.4f}s ({total_timing_stats['kl_computation']/total_generation_time*100:5.2f}%)")
    print(f"  Sampling:                  {total_timing_stats['sampling']:8.4f}s ({total_timing_stats['sampling']/total_generation_time*100:5.2f}%)")
    print(f"  Quantized sampling:        {total_timing_stats['quantized_sampling']:8.4f}s ({total_timing_stats['quantized_sampling']/total_generation_time*100:5.2f}%)")
    print(f"  Token selection:           {total_timing_stats['selection']:8.4f}s ({total_timing_stats['selection']/total_generation_time*100:5.2f}%)")
    
    overhead = total_generation_time - sum(total_timing_stats.values())
    print(f"  Other/Overhead:            {overhead:8.4f}s ({overhead/total_generation_time*100:5.2f}%)")
    print("="*80)

    # Compute aggregate statistics for quantization analysis
    print("\n" + "="*80)
    print("QUANTIZATION ANALYSIS SUMMARY")
    print("="*80)
    
    all_kl_divs = [log['avg_kl_divergence'] for log in all_analysis_logs]
    all_jaccard_sims = [log['jaccard_similarity'] for log in all_analysis_logs]
    
    if len(all_kl_divs) > 0:
        avg_kl_div = np.mean(all_kl_divs)
        std_kl_div = np.std(all_kl_divs)
        print(f"Average KL Divergence: {avg_kl_div:.6f} ± {std_kl_div:.6f}")
    
    if len(all_jaccard_sims) > 0:
        avg_jaccard = np.mean(all_jaccard_sims)
        std_jaccard = np.std(all_jaccard_sims)
        print(f"Average Jaccard Similarity: {avg_jaccard:.6f} ± {std_jaccard:.6f}")
    
    print(f"\nGeneration Length Statistics:")
    print(f"  Mean: {mean_gen_length:.2f} tokens")
    print(f"  Std Dev: {std_gen_length:.2f} tokens")
    print(f"  Min: {float(np.min(generation_lengths)):.2f} tokens")
    print(f"  Max: {float(np.max(generation_lengths)):.2f} tokens")
    
    print("="*80)

    # Save all results
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        probs_path = os.path.join(output_dir, "probs.jsonl")
        analysis_path = os.path.join(output_dir, "quantization_analysis.jsonl")
        timing_path = os.path.join(output_dir, "timing_stats.json")
        summary_path = os.path.join(output_dir, "analysis_summary.json")
    else:
        probs_path = "probs.jsonl"
        analysis_path = "quantization_analysis.jsonl"
        timing_path = "timing_stats.json"
        summary_path = "analysis_summary.json"
    
    # Save probability logs
    with open(probs_path, "w") as f:
        for row in all_logs:
            f.write(json.dumps(row) + "\n")
    
    # Save quantization analysis logs
    with open(analysis_path, "w") as f:
        for row in all_analysis_logs:
            f.write(json.dumps(row) + "\n")
    print(f"Quantization analysis saved to: {analysis_path}")
    
    # Save timing statistics
    timing_summary = {
        "total_generation_time": total_generation_time,
        "breakdown": dict(total_timing_stats),
        "percentages": {
            "initialization": total_timing_stats['initialization']/total_generation_time*100,
            "block_setup": total_timing_stats['block_setup']/total_generation_time*100,
            "model_inference": total_timing_stats['model_inference']/total_generation_time*100,
            "quantized_model_inference": total_timing_stats['quantized_model_inference']/total_generation_time*100,
            "kl_computation": total_timing_stats['kl_computation']/total_generation_time*100,
            "sampling": total_timing_stats['sampling']/total_generation_time*100,
            "quantized_sampling": total_timing_stats['quantized_sampling']/total_generation_time*100,
            "selection": total_timing_stats['selection']/total_generation_time*100,
            "overhead": overhead/total_generation_time*100
        }
    }
    with open(timing_path, "w") as f:
        json.dump(timing_summary, f, indent=2)
    print(f"Timing statistics saved to: {timing_path}")
    
    # Save analysis summary
    analysis_summary = {
        "kl_divergence": {
            "mean": float(np.mean(all_kl_divs)) if len(all_kl_divs) > 0 else None,
            "std": float(np.std(all_kl_divs)) if len(all_kl_divs) > 0 else None,
            "min": float(np.min(all_kl_divs)) if len(all_kl_divs) > 0 else None,
            "max": float(np.max(all_kl_divs)) if len(all_kl_divs) > 0 else None,
        },
        "jaccard_similarity": {
            "mean": float(np.mean(all_jaccard_sims)) if len(all_jaccard_sims) > 0 else None,
            "std": float(np.std(all_jaccard_sims)) if len(all_jaccard_sims) > 0 else None,
            "min": float(np.min(all_jaccard_sims)) if len(all_jaccard_sims) > 0 else None,
            "max": float(np.max(all_jaccard_sims)) if len(all_jaccard_sims) > 0 else None,
        },
        "generation_length": {
            "mean": mean_gen_length,
            "std": std_gen_length,
            "min": float(np.min(generation_lengths)),
            "max": float(np.max(generation_lengths)),
            "all_lengths": generation_lengths.tolist()
        },
        "total_steps_analyzed": len(all_analysis_logs)
    }
    with open(summary_path, "w") as f:
        json.dump(analysis_summary, f, indent=2)
    print(f"Analysis summary saved to: {summary_path}")

    # Generate plots
    if len(all_analysis_logs) > 0:
        print("\n" + "="*80)
        print("GENERATING PLOTS")
        print("="*80)
        
        # Load analysis logs into DataFrame for plotting
        df_analysis = pd.DataFrame(all_analysis_logs)
        
        plots_dir = os.path.join(output_dir, 'plots') if output_dir is not None else 'plots'
        
        # Plot per-batch metrics
        print("Generating per-batch plots...")
        plot_quantization_metrics_per_batch(df_analysis, plots_dir, mean_gen_length, std_gen_length)
        print(f"Per-batch plots saved to: {plots_dir}")
        
        # Plot aggregated metrics if we have multiple batches
        if df_analysis['batch_idx'].nunique() > 1:
            print("\nGenerating aggregated plots...")
            plot_aggregated_quantization_metrics(df_analysis, plots_dir, mean_gen_length, std_gen_length)
            print(f"Aggregated plots saved to: {plots_dir}")
        else:
            print("\nOnly one batch found. Skipping aggregated plots.")
        
        print("="*80)

    return x


def load_dataset_examples(dataset_name, tokenizer, max_examples, max_length, min_length=0):
    """
    Load examples from a dataset.
    
    Args:
        dataset_name: 'toy' or 'wikitext2'
        tokenizer: Tokenizer to use
        max_examples: Maximum number of examples to load
        max_length: Maximum length of each example (0 for no limit)
        min_length: Minimum length of each example in tokens (0 for no limit)
    """
    if dataset_name == 'toy':
        # Create toy dataset with simple prompts
        prompts = [
            "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?",
            "Joy can read 8 pages of a book in 20 minutes. How many hours will it take her to read 120 pages?",
            "Randy has 60 mango trees on his farm. He also has 5 less than half as many coconut trees as mango trees. How many trees does Randy have in all on his farm?"
        ]
        # Filter by min_length if specified
        if min_length > 0:
            filtered_prompts = []
            for prompt in prompts:
                tokens = tokenizer.encode(prompt, add_special_tokens=False)
                if len(tokens) >= min_length:
                    # Truncate if max_length is specified
                    if max_length > 0 and len(tokens) > max_length:
                        tokens = tokens[:max_length]
                        prompt = tokenizer.decode(tokens, skip_special_tokens=True)
                    filtered_prompts.append(prompt)
            prompts = filtered_prompts
        
        # Repeat or truncate to max_examples
        prompts = prompts[:max_examples]
        return prompts
    elif dataset_name == 'wikitext2':
        # Load wikitext2 from huggingface
        print("Loading wikitext2 dataset...")
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
        texts = []
        # Try to get dataset length, but handle cases where it might not be available
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
            if len(text) > 0:  # Skip empty texts
                # Check token length for filtering
                tokens = tokenizer.encode(text, add_special_tokens=False)
                
                # Filter by min_length if specified
                if min_length > 0 and len(tokens) < min_length:
                    continue
                
                # Truncate by token length if max_length is specified
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
    parser = argparse.ArgumentParser(description='Run LLaDA generation with quantization analysis')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='GSAI-ML/LLaDA-8B-Instruct',
                        help='Precise model name or path to load from HuggingFace')
    parser.add_argument('--quantized_model', type=str, required=True,
                        help='Path to AWQ metadata file (.pt) for quantized model')
    parser.add_argument('--q_backend', type=str, default='real', choices=['real', 'fake'],
                        help='Quantization backend: "real" for actual quantization, "fake" for calibration only')
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, choices=['toy', 'wikitext2'], default='toy',
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
    parser.add_argument('--remasking', type=str, choices=['low_confidence'], default='low_confidence',
                        help='Remasking strategy: must be "low_confidence" for this analysis')
    parser.add_argument('--logits_eos_inf', action='store_true',
                        help='Set logits of EOS token to -inf')
    parser.add_argument('--confidence_eos_eot_inf', action='store_true',
                        help='Set confidence of EOS and EoT token to -inf')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Number of samples to process in each batch (default: 8)')
    
    args = parser.parse_args()
    
    # Validate remasking strategy
    if args.remasking != 'low_confidence':
        raise ValueError("This analysis script only supports 'low_confidence' remasking strategy")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup profiling directory if enabled
    profile_dir = None
    if args.profile:
        profile_dir = os.path.join(args.output_dir, 'profile')
        os.makedirs(profile_dir, exist_ok=True)
    
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
        'remasking': args.remasking,
        'logits_eos_inf': args.logits_eos_inf,
        'confidence_eos_eot_inf': args.confidence_eos_eot_inf,
        'profile': args.profile,
        'batch_size': args.batch_size,
        'q_backend': args.q_backend,
    }
    
    config_path = os.path.join(args.output_dir, 'call_config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Check GPU availability and ensure at least 2 GPUs
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This script requires GPU support.")
    
    num_gpus = torch.cuda.device_count()
    if num_gpus < 2:
        raise RuntimeError(f"At least 2 GPUs are required, but only {num_gpus} GPU(s) available.")
    
    print(f"Found {num_gpus} GPU(s)")
    device_precise = 'cuda:0'
    device_quantized = 'cuda:1'
    
    print(f"Loading precise model on {device_precise}: {args.model}")
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device_precise).eval()
    
    # Load quantized model: load base model and apply AWQ metadata
    print(f"Loading base model for quantization on {device_quantized}: {args.model}")
    quantized_model = AutoModel.from_pretrained(args.model, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device_quantized).eval()
    
    # Load and apply AWQ metadata
    # NOTE: apply_awq modifies the model by applying activation scaling and clipping.
    # This changes the model's behavior even without weight quantization (q_backend='fake').
    # The scaling factors are applied to layer norms, activations (GELU, SiLU), and linear layers
    # to prepare the model for quantization. This is why outputs may differ from the precise model
    # even when weight quantization is not applied.
    print(f"Loading AWQ metadata from: {args.quantized_model}")
    if not os.path.exists(args.quantized_model):
        raise FileNotFoundError(f"AWQ metadata file not found: {args.quantized_model}")
    
    awq_results = torch.load(args.quantized_model, map_location='cpu')
    apply_awq(quantized_model, awq_results)
    
    # Infer w_bit and q_group_size from filename (needed for both real and fake quantization)
    try:
        w_bit, q_group_size = infer_quantization_params_from_filename(args.quantized_model)
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
    if args.q_backend == 'real':
        print("Applying real weight quantization...")
        # Apply real quantization
        real_quantize_model_weight(
            quantized_model, 
            w_bit=w_bit, 
            q_config=q_config
        )
        print(f"Applied real weight quantization (w_bit={w_bit}, q_group_size={q_group_size})")
    elif args.q_backend == 'fake':
        print("Applying pseudo weight quantization...")
        # Apply pseudo quantization
        pseudo_quantize_model_weight(
            quantized_model,
            w_bit=w_bit,
            q_config=q_config
        )
        print(f"Applied pseudo weight quantization (w_bit={w_bit}, q_group_size={q_group_size})")
    else:
        raise ValueError(f"Invalid q_backend: {args.q_backend}. Must be 'real' or 'fake'")
    
    # After applying AWQ, ensure all model parameters are on the correct device
    # (apply_scale moves modules to CPU during processing, so we need to move them back)
    print(f"Moving quantized model to {device_quantized}...")
    quantized_model = quantized_model.to(device_quantized)
    
    # Double-check: ensure all parameters and buffers are on the correct device
    for name, param in quantized_model.named_parameters():
        if param.device != torch.device(device_quantized):
            param.data = param.data.to(device_quantized)
    for name, buffer in quantized_model.named_buffers():
        if buffer.device != torch.device(device_quantized):
            buffer.data = buffer.data.to(device_quantized)
    
    print(f"Quantization complete. Model on {device_quantized}")
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # The LLaDA architecture theoretically supports both left-padding and right-padding. 
    # However, the sampling code implementation is simpler with left-padding.
    if tokenizer.padding_side != 'left':
        tokenizer.padding_side = 'left'

    # If the padding ID equals the mask ID, you need to modify our generate function to achieve correct inference.
    assert tokenizer.pad_token_id != 126336

    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    raw_prompts = load_dataset_examples(args.dataset, tokenizer, args.max_examples, args.max_length, args.min_length)
    
    # For instruct models, apply chat template
    print("Preparing prompts...")
    if 'instruct' in args.model.lower():
        messages = [{"role": "user", "content": prompt} for prompt in raw_prompts]
        prompts = [tokenizer.apply_chat_template([message], add_generation_prompt=True, tokenize=False) for message in messages]
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
    
    print(f"Starting generation with quantization analysis (steps={args.steps}, gen_length={args.gen_length}, block_length={args.block_length})...")

    out = generate_with_quantization_analysis(
        model=model,
        quantized_model=quantized_model,
        prompt=input_ids,
        attention_mask=attention_mask,
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
        batch_size=args.batch_size,
        device_quantized=device_quantized
    )
    
    print("\nGeneration complete! Decoding outputs...")
    output = tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)
    print("\nGenerated outputs:")
    for i, o in enumerate(output):
        print(f"\nExample {i + 1}:")
        print(o)
        print('-' * 50)
    print(f"\nResults saved to: {args.output_dir}")
    
    if args.profile:
        print(f"\nTo view profiling results in TensorBoard, run:")
        print(f"  tensorboard --logdir={profile_dir}")


if __name__ == '__main__':
    main()