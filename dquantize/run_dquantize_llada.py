import argparse
import os
import yaml
from types import SimpleNamespace
from dquantize.dquantize_llada import DQuantizeModelLLada, DQuantizeConfig
from dquantize.tasks.harness import TaskHarness


def main():
    parser = argparse.ArgumentParser(description='Run DQuantize LLaDA generation on GSM8K')
    
    # Model arguments
    parser.add_argument('--model_name', type=str, default='GSAI-ML/LLaDA-8B-Instruct',
                        help='Model name or path to load from HuggingFace')
    parser.add_argument('--quantized_model_path', type=str, required=True,
                        help='Path to AWQ metadata file (.pt)')
    
    # DQuantize strategy arguments
    parser.add_argument('--strategy', type=str, choices=['all', 'firstk', 'lastk'], default='firstk',
                        help='Strategy for model selection: "all", "firstk", or "lastk"')
    parser.add_argument('--k', type=int, default=64,
                        help='Number of steps for "firstk" and "lastk" strategies')
    
    # Device arguments
    parser.add_argument('--device_precise', type=str, default='cuda:0',
                        help='Device for precise model')
    parser.add_argument('--device_quantized', type=str, default='cuda:1',
                        help='Device for quantized model')
    
    # Quantization arguments
    parser.add_argument('--q_backend', type=str, default='real', choices=['real', 'fake'],
                        help='Quantization backend: "real" for actual quantization, "fake" for calibration only')
    
    # Generation arguments (will be part of DQuantizeConfig)
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
    
    # Dataset arguments (only gsm8k for now)
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
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory to save results and config')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create DQuantizeConfig
    dquantize_config = DQuantizeConfig(
        k=args.k,
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
    
    # Save configuration
    config = SimpleNamespace(
        model_name=args.model_name,
        quantized_model_path=args.quantized_model_path,
        strategy=args.strategy,
        k=args.k,
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
    )
    
    config_path = os.path.join(args.output_dir, 'config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(vars(config), f, default_flow_style=False)
    
    print("="*80)
    print("Initializing DQuantizeModelLLada")
    print("="*80)
    print(f"Model: {args.model_name}")
    print(f"Quantized model path: {args.quantized_model_path}")
    print(f"Strategy: {args.strategy}")
    if args.strategy in ['firstk', 'lastk']:
        print(f"k: {args.k}")
    print(f"Device precise: {args.device_precise}")
    print(f"Device quantized: {args.device_quantized}")
    print("="*80)
    
    # Initialize model
    dq_model = DQuantizeModelLLada(
        model_name=args.model_name,
        quantized_model_path=args.quantized_model_path,
        strategy=args.strategy,
        dquantize_config=dquantize_config,
        device_precise=args.device_precise,
        device_quantized=args.device_quantized,
        q_backend=args.q_backend,
    )
    
    print("\n" + "="*80)
    print("Running Task Harness")
    print("="*80)
    
    # Create task harness
    harness = TaskHarness(
        model=dq_model,
        tokenizer=dq_model.tokenizer,
        output_dir=args.output_dir,
        dataset_name=args.dataset,
        max_examples=args.max_examples,
        max_length=args.max_length,
        min_length=args.min_length,
        split=args.split,
    )
    
    # Run evaluation
    results = harness.run()
    
    print("\n" + "="*80)
    print("Evaluation Complete!")
    print("="*80)
    print(f"Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()

