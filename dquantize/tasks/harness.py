"""
Task harness for evaluating models on datasets.
"""
import os
import json
from typing import List, Dict, Any
from . import load_dataset_examples
from ..utils.postprocess import extract_final_answer_from_output, show_accuracy


class TaskHarness:
    """
    A harness class that accepts a model with a generate method,
    loads dataset examples, generates answers, logs results, and saves to output_dir.
    """
    
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        output_dir: str,
        dataset_name: str = 'gsm8k',
        max_examples: int = 8,
        max_length: int = 0,
        min_length: int = 0,
        **dataset_kwargs
    ):
        """
        Initialize the task harness.
        
        Args:
            model: Model instance with a generate(prompts: List[str]) -> List[str] method
            tokenizer: Tokenizer instance (used for dataset loading)
            output_dir: Directory to save results
            dataset_name: Name of the dataset (default: 'gsm8k')
            max_examples: Maximum number of examples to load
            max_length: Maximum length of each example in tokens (0 for no limit)
            min_length: Minimum length of each example in tokens (0 for no limit)
            **dataset_kwargs: Additional arguments passed to dataset loader
        """
        self.model = model
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        self.dataset_name = dataset_name
        self.max_examples = max_examples
        self.max_length = max_length
        self.min_length = min_length
        self.dataset_kwargs = dataset_kwargs
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def run(self) -> Dict[str, Any]:
        """
        Run the evaluation: load dataset, generate answers, evaluate, and save results.
        
        Returns:
            Dictionary containing results and metrics
        """
        print(f"Loading dataset: {self.dataset_name}")
        data = load_dataset_examples(
            self.dataset_name,
            self.tokenizer,
            self.max_examples,
            self.max_length,
            self.min_length,
            **self.dataset_kwargs
        )
        
        # Extract prompts and targets
        if not data:
            raise ValueError(f"No data loaded from dataset: {self.dataset_name}")
        
        if isinstance(data[0], dict):
            prompts = [d['question'] for d in data]
            targets = [d.get('target', '') for d in data]
        else:
            # Fallback for datasets that return just strings
            prompts = data
            targets = [''] * len(data)
        
        print(f"Loaded {len(prompts)} examples")
        print(f"Generating answers...")
        
        # Generate answers
        outputs = self.model.generate(prompts=prompts)
        
        print(f"Generated {len(outputs)} answers")
        print("Extracting final answers...")
        
        # Extract final answers for evaluation
        final_outputs = [extract_final_answer_from_output(o) for o in outputs]
        
        # Calculate accuracy if targets are available
        accuracy = None
        if targets and any(t for t in targets):
            print("\nEvaluating accuracy...")
            accuracy = show_accuracy(final_outputs, targets)
        
        # Prepare results
        results = {
            'prompts': prompts,
            'outputs': outputs,
            'final_outputs': final_outputs,
            'targets': targets,
            'accuracy': accuracy,
            'num_examples': len(prompts)
        }
        
        # Save results
        self._save_results(results)
        
        # Print summary
        print("\n" + "="*80)
        print("EVALUATION SUMMARY")
        print("="*80)
        print(f"Dataset: {self.dataset_name}")
        print(f"Number of examples: {len(prompts)}")
        if accuracy is not None:
            print(f"Accuracy: {accuracy*100:.2f}%")
        print(f"Results saved to: {self.output_dir}")
        print("="*80)
        
        return results
    
    def _save_results(self, results: Dict[str, Any]):
        """Save results to files in output_dir."""
        # Save full outputs
        outputs_path = os.path.join(self.output_dir, 'outputs.jsonl')
        with open(outputs_path, 'w') as f:
            for i, (prompt, output, final_output, target) in enumerate(zip(
                results['prompts'],
                results['outputs'],
                results['final_outputs'],
                results['targets']
            )):
                f.write(json.dumps({
                    'example_id': i,
                    'prompt': prompt,
                    'output': output,
                    'final_answer': final_output,
                    'target': target,
                    'correct': final_output == target if target else None
                }) + '\n')
        
        # Save summary
        summary_path = os.path.join(self.output_dir, 'summary.json')
        summary = {
            'dataset': self.dataset_name,
            'num_examples': results['num_examples'],
            'accuracy': results['accuracy'],
        }
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save readable output
        readable_path = os.path.join(self.output_dir, 'readable_output.txt')
        with open(readable_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("EVALUATION RESULTS\n")
            f.write("="*80 + "\n\n")
            for i, (prompt, output, final_output, target) in enumerate(zip(
                results['prompts'],
                results['outputs'],
                results['final_outputs'],
                results['targets']
            )):
                f.write(f"\nExample {i + 1}:\n")
                f.write("-" * 80 + "\n")
                f.write(f"Prompt: {prompt}\n\n")
                f.write(f"Generated Output:\n{output}\n\n")
                f.write(f"Extracted Answer: {final_output}\n")
                if target:
                    f.write(f"Target Answer: {target}\n")
                    f.write(f"Correct: {final_output == target}\n")
                f.write("-" * 80 + "\n")
            
            if results['accuracy'] is not None:
                f.write(f"\n\nOverall Accuracy: {results['accuracy']*100:.2f}%\n")
        
        print(f"\nResults saved:")
        print(f"  - Full outputs: {outputs_path}")
        print(f"  - Summary: {summary_path}")
        print(f"  - Readable output: {readable_path}")

