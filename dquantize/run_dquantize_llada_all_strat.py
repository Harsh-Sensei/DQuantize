import argparse
import random
import os
import yaml
import json
import matplotlib.pyplot as plt
import numpy as np
from types import SimpleNamespace
from dquantize.dquantize_llada import DQuantizeModelLLada, DQuantizeConfig
from dquantize.tasks.harness import TaskHarness
import torch


torch.manual_seed(73)
np.random.seed(73)
random.seed(73)


def calculate_total_steps(gen_length: int, block_length: int, steps: int) -> int:
    """
    Calculate total number of steps across all blocks.
    
    Args:
        gen_length: Total generation length
        block_length: Length of each block
        steps: Steps per block (before division by num_blocks)
    
    Returns:
        Total number of steps
    """
    assert gen_length % block_length == 0, "gen_length must be divisible by block_length"
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0, "steps must be divisible by num_blocks"
    steps_per_block = steps // num_blocks
    return steps_per_block


def run_experiment(
    model_name: str,
    quantized_model_path: str,
    strategy: str,
    k: int,
    dquantize_config: DQuantizeConfig,
    device_precise: str,
    device_quantized: str,
    q_backend: str,
    dataset: str,
    max_examples: int,
    max_length: int,
    min_length: int,
    split: str,
    output_subdir: str,
) -> float:
    """
    Run a single experiment and return accuracy.
    
    Returns:
        Accuracy as a float (0.0 to 1.0), or None if targets not available
    """
    print(f"\n{'='*80}")
    print(f"Running experiment: strategy={strategy}, k={k}")
    print(f"{'='*80}")
    
    # Update config with k value
    config = DQuantizeConfig(
        k=k,
        steps=dquantize_config.steps,
        gen_length=dquantize_config.gen_length,
        block_length=dquantize_config.block_length,
        temperature=dquantize_config.temperature,
        cfg_scale=dquantize_config.cfg_scale,
        logits_eos_inf=dquantize_config.logits_eos_inf,
        confidence_eos_eot_inf=dquantize_config.confidence_eos_eot_inf,
        batch_size=dquantize_config.batch_size,
        apply_chat_template=dquantize_config.apply_chat_template,
        mask_id=dquantize_config.mask_id,
    )
    
    # Initialize model
    dq_model = DQuantizeModelLLada(
        model_name=model_name,
        quantized_model_path=quantized_model_path,
        strategy=strategy,
        dquantize_config=config,
        device_precise=device_precise,
        device_quantized=device_quantized,
        q_backend=q_backend,
    )
    
    # Create task harness
    harness = TaskHarness(
        model=dq_model,
        tokenizer=dq_model.tokenizer,
        output_dir=output_subdir,
        dataset_name=dataset,
        max_examples=max_examples,
        max_length=max_length,
        min_length=min_length,
        split=split,
    )
    
    # Run evaluation
    results = harness.run()
    
    return results.get('accuracy')


def main():
    parser = argparse.ArgumentParser(description='Run DQuantize LLaDA experiments with all strategies')
    
    # Model arguments
    parser.add_argument('--model_name', type=str, default='GSAI-ML/LLaDA-8B-Instruct',
                        help='Model name or path to load from HuggingFace')
    parser.add_argument('--quantized_model_path', type=str, required=True,
                        help='Path to AWQ metadata file (.pt)')
    
    # Device arguments
    parser.add_argument('--device_precise', type=str, default='cuda:0',
                        help='Device for precise model')
    parser.add_argument('--device_quantized', type=str, default='cuda:1',
                        help='Device for quantized model')
    
    # Quantization arguments
    parser.add_argument('--q_backend', type=str, default='real', choices=['real', 'fake'],
                        help='Quantization backend: "real" for actual quantization, "fake" for calibration only')
    
    # Generation arguments
    parser.add_argument('--steps', type=int, default=128,
                        help='Sampling steps per block')
    parser.add_argument('--gen_length', type=int, default=128,
                        help='Total generation length')
    parser.add_argument('--block_length', type=int, default=128,
                        help='Length of each generation block')
    parser.add_argument('--temperature', type=float, default=0.,
                        help='Sampling temperature')
    parser.add_argument('--cfg_scale', type=float, default=0.,
                        help='Classifier-free guidance scale')
    parser.add_argument('--logits_eos_inf', action='store_true',
                        help='Set EOS logits to -inf')
    parser.add_argument('--confidence_eos_eot_inf', action='store_true',
                        help='Set EOS/EoT confidence to -inf')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Maximum batch size for processing')
    parser.add_argument('--apply_chat_template', action='store_true', default=True,
                        help='Apply chat template for instruct models')
    parser.add_argument('--no_apply_chat_template', dest='apply_chat_template', action='store_false',
                        help='Do not apply chat template')
    parser.add_argument('--mask_id', type=int, default=126336,
                        help='Token ID for mask token')
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='gsm8k', choices=['gsm8k'],
                        help='Dataset to load (only gsm8k supported)')
    parser.add_argument('--max_examples', type=int, default=8,
                        help='Maximum number of examples to process')
    parser.add_argument('--max_length', type=int, default=0,
                        help='Maximum length of each example in tokens (0 for no limit)')
    parser.add_argument('--min_length', type=int, default=0,
                        help='Minimum length of each example in tokens (0 for no limit)')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'test'],
                        help='Dataset split to use')
    
    # Experiment arguments
    parser.add_argument('--num_k_splits', type=int, default=10,
                        help='Number of k values to test (will test 0%, 10%, 20%, ..., 100% of total steps)')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory to save results and config')
    
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    plots_dir = os.path.join(args.output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Create base DQuantizeConfig
    dquantize_config = DQuantizeConfig(
        k=0,  # Will be overridden for each experiment
        steps=args.steps,
        gen_length=args.gen_length,
        block_length=args.block_length,
        temperature=args.temperature,
        cfg_scale=args.cfg_scale,
        logits_eos_inf=args.logits_eos_inf,
        confidence_eos_eot_inf=args.confidence_eos_eot_inf,
        batch_size=args.batch_size,
        apply_chat_template=args.apply_chat_template,
        mask_id=args.mask_id,
    )
    
    # Calculate total steps
    total_steps = calculate_total_steps(
        args.gen_length,
        args.block_length,
        args.steps
    )
    print(f"Total steps: {total_steps}")
    
    # Calculate k values based on percentages
    percentages = np.linspace(0, 100, args.num_k_splits + 1)
    k_values = [int(total_steps * p / 100) for p in percentages]
    k_values = sorted(list(set(k_values)))  # Remove duplicates and sort
    
    print(f"Testing k values: {k_values}")
    print(f"Corresponding percentages: {[int(p) for p in percentages if int(total_steps * p / 100) in k_values]}")
    
    # Save configuration
    config = SimpleNamespace(
        model_name=args.model_name,
        quantized_model_path=args.quantized_model_path,
        device_precise=args.device_precise,
        device_quantized=args.device_quantized,
        q_backend=args.q_backend,
        steps=args.steps,
        gen_length=args.gen_length,
        block_length=args.block_length,
        temperature=args.temperature,
        cfg_scale=args.cfg_scale,
        logits_eos_inf=args.logits_eos_inf,
        confidence_eos_eot_inf=args.confidence_eos_eot_inf,
        batch_size=args.batch_size,
        apply_chat_template=args.apply_chat_template,
        mask_id=args.mask_id,
        dataset=args.dataset,
        max_examples=args.max_examples,
        max_length=args.max_length,
        min_length=args.min_length,
        split=args.split,
        num_k_splits=args.num_k_splits,
        total_steps=total_steps,
        k_values=k_values,
    )
    
    config_path = os.path.join(args.output_dir, 'config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(vars(config), f, default_flow_style=False)
    
    # Store results
    results = {
        'firstk': {},
        'lastk': {},
        'all': None,
    }
    
    # Run experiments for firstk
    print("\n" + "="*80)
    print("RUNNING FIRSTK EXPERIMENTS")
    print("="*80)
    for k in k_values:
        output_subdir = os.path.join(args.output_dir, f'firstk_k{k}')
        os.makedirs(output_subdir, exist_ok=True)
        
        accuracy = run_experiment(
            model_name=args.model_name,
            quantized_model_path=args.quantized_model_path,
            strategy='firstk',
            k=k,
            dquantize_config=dquantize_config,
            device_precise=args.device_precise,
            device_quantized=args.device_quantized,
            q_backend=args.q_backend,
            dataset=args.dataset,
            max_examples=args.max_examples,
            max_length=args.max_length,
            min_length=args.min_length,
            split=args.split,
            output_subdir=output_subdir,
        )
        results['firstk'][k] = accuracy
        print(f"firstk k={k}: accuracy={accuracy:.4f}" if accuracy is not None else f"firstk k={k}: accuracy=None")
        
    # Run experiments for lastk
    print("\n" + "="*80)
    print("RUNNING LASTK EXPERIMENTS")
    print("="*80)
    for k in k_values:
        output_subdir = os.path.join(args.output_dir, f'lastk_k{k}')
        os.makedirs(output_subdir, exist_ok=True)
        
        accuracy = run_experiment(
            model_name=args.model_name,
            quantized_model_path=args.quantized_model_path,
            strategy='lastk',
            k=k,
            dquantize_config=dquantize_config,
            device_precise=args.device_precise,
            device_quantized=args.device_quantized,
            q_backend=args.q_backend,
            dataset=args.dataset,
            max_examples=args.max_examples,
            max_length=args.max_length,
            min_length=args.min_length,
            split=args.split,
            output_subdir=output_subdir,
        )
        results['lastk'][k] = accuracy
        print(f"lastk k={k}: accuracy={accuracy:.4f}" if accuracy is not None else f"lastk k={k}: accuracy=None")
    
    # Run experiment for all
    print("\n" + "="*80)
    print("RUNNING ALL EXPERIMENT")
    print("="*80)
    output_subdir = os.path.join(args.output_dir, 'all')
    os.makedirs(output_subdir, exist_ok=True)
    
    print(f"Running all experiment")
    accuracy = run_experiment(
        model_name=args.model_name,
        quantized_model_path=args.quantized_model_path,
        strategy='all',
        k=0,  # Not used for 'all' strategy
        dquantize_config=dquantize_config,
        device_precise=args.device_precise,
        device_quantized=args.device_quantized,
        q_backend=args.q_backend,
        dataset=args.dataset,
        max_examples=args.max_examples,
        max_length=args.max_length,
        min_length=args.min_length,
        split=args.split,
        output_subdir=output_subdir,
    )
    results['all'] = accuracy
    print(f"all: accuracy={accuracy:.4f}" if accuracy is not None else "all: accuracy=None")
    
    # Save results
    results_path = os.path.join(args.output_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    
    # Create plots
    print("\n" + "="*80)
    print("CREATING PLOTS")
    print("="*80)
    
    # Prepare data for plotting
    firstk_k = sorted([k for k in results['firstk'].keys() if results['firstk'][k] is not None])
    firstk_acc = [results['firstk'][k] for k in firstk_k]
    firstk_percentages = [100 * k / total_steps for k in firstk_k]
    
    lastk_k = sorted([k for k in results['lastk'].keys() if results['lastk'][k] is not None])
    lastk_acc = [results['lastk'][k] for k in lastk_k]
    lastk_percentages = [100 * k / total_steps for k in lastk_k]
    
    # Plot 1: Accuracy vs k (absolute values)
    fig, ax = plt.subplots(figsize=(10, 6))
    if firstk_k:
        ax.plot(firstk_k, firstk_acc, marker='o', label='firstk', linewidth=2, markersize=8)
    if lastk_k:
        ax.plot(lastk_k, lastk_acc, marker='s', label='lastk', linewidth=2, markersize=8)
    if results['all'] is not None:
        ax.axhline(y=results['all'], color='green', linestyle='--', linewidth=2, label='all', alpha=0.7)
    
    ax.set_xlabel('k (number of steps)', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Accuracy vs k for Different Strategies', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    plt.tight_layout()
    
    plot_path1 = os.path.join(plots_dir, 'accuracy_vs_k.png')
    plt.savefig(plot_path1, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {plot_path1}")
    
    # Plot 2: Accuracy vs percentage of total steps
    fig, ax = plt.subplots(figsize=(10, 6))
    if firstk_percentages:
        ax.plot(firstk_percentages, firstk_acc, marker='o', label='firstk', linewidth=2, markersize=8)
    if lastk_percentages:
        ax.plot(lastk_percentages, lastk_acc, marker='s', label='lastk', linewidth=2, markersize=8)
    if results['all'] is not None:
        ax.axhline(y=results['all'], color='green', linestyle='--', linewidth=2, label='all', alpha=0.7)
    
    ax.set_xlabel('k as % of total steps', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Accuracy vs k (% of total steps) for Different Strategies', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    ax.set_xlim([0, 105])
    plt.tight_layout()
    
    plot_path2 = os.path.join(plots_dir, 'accuracy_vs_k_percentage.png')
    plt.savefig(plot_path2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {plot_path2}")
    
    # Print summary
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    print(f"Total steps: {total_steps}")
    print(f"Number of k values tested: {len(k_values)}")
    print(f"\nFirstK Results:")
    for k in sorted(results['firstk'].keys()):
        acc = results['firstk'][k]
        pct = 100 * k / total_steps
        print(f"  k={k:4d} ({pct:5.1f}%): {acc:.4f}" if acc is not None else f"  k={k:4d} ({pct:5.1f}%): None")
    print(f"\nLastK Results:")
    for k in sorted(results['lastk'].keys()):
        acc = results['lastk'][k]
        pct = 100 * k / total_steps
        print(f"  k={k:4d} ({pct:5.1f}%): {acc:.4f}" if acc is not None else f"  k={k:4d} ({pct:5.1f}%): None")
    print(f"\nAll Strategy: {results['all']:.4f}" if results['all'] is not None else "\nAll Strategy: None")
    print("="*80)
    print(f"\nAll results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()

