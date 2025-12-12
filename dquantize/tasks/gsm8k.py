from .base import BaseDatasetLoader
from . import register_dataset
from datasets import load_dataset
from typing import List, Dict, Any, Optional, Callable

@register_dataset('gsm8k')
class GSM8kDatasetLoader(BaseDatasetLoader):
    """
    GSM8K (Grade School Math 8K) dataset loader.
    
    GSM8K contains grade school math word problems with step-by-step solutions.
    Each example has:
        - question: The math word problem
        - answer: Step-by-step solution ending with "#### <final_answer>"
    """
        
    def __init__(self, tokenizer, max_examples: int, max_length: int, min_length: int = 0,
                 split: str = 'test', include_answer: bool = False):
        """
        Args:
            tokenizer: Tokenizer to use
            max_examples: Maximum number of examples to load
            max_length: Maximum length of each example (0 for no limit)
            min_length: Minimum length of each example in tokens (0 for no limit)
            split: Dataset split ('train' or 'test')
            include_answer: If True, include the answer in the returned text
        """
        super().__init__(tokenizer, max_examples, max_length, min_length)
        self.split = split
        self.include_answer = include_answer

    def load_raw_data(self) -> List[Dict[str, Any]]:
        print(f"Loading GSM8K dataset ({self.split} split)...")
        dataset = load_dataset('openai/gsm8k', 'main', split=self.split)
        return list(dataset)
    
    def extract_text(self, item: Dict[str, Any]) -> str:
        """Extract question (and optionally answer) from GSM8K item."""
        question = item.get('question', '')
        
        if self.include_answer:
            answer = item.get('answer', '')
            return f"{question}\n\n{answer}"
        
        return question
    
    def load(self) -> List[dict]:
        """
        Load and process examples from the dataset,
        storing both the input text and the correct output.
        """
        raw_data = self.load_raw_data()
        processed = []
        
        for item in raw_data:
            if len(processed) >= self.max_examples:
                break
            
            text = self.extract_text(item)
            if not text or not text.strip():
                continue
            
            filtered_text = self.filter_by_length(text.strip())
            if filtered_text is not None:
                # store both input and target
                processed.append({
                    "question": filtered_text,
                    "target": GSM8kDatasetLoader.extract_final_answer(item.get("answer", "")),
                })
        
        return processed[:self.max_examples]
    

    @staticmethod
    def extract_final_answer(answer_text: str) -> Optional[str]:
        """
        Extract the final numerical answer from GSM8K answer format.
        GSM8K answers end with "#### <number>"
        """
        if '####' in answer_text:
            return answer_text.split('####')[-1].strip()
        return None
