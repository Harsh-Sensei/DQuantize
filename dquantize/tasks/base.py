from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Callable

# Abstract Base Class for Dataset Loading
class BaseDatasetLoader(ABC):
    
    def __init__(self, tokenizer, max_examples: int, max_length: int, min_length: int = 0):
        self.tokenizer = tokenizer
        self.max_examples = max_examples
        self.max_length = max_length
        self.min_length = min_length
    
    @abstractmethod
    def load_raw_data(self) -> List[Dict[str, Any]]:
        """Load raw data from the source. Returns list of dicts with at least 'text' key."""
        pass
    
    def filter_by_length(self, text: str) -> Optional[str]:
        """
        Filter and truncate text by token length.
        Returns None if text doesn't meet min_length requirement.
        """
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        
        # Filter by min_length
        if self.min_length > 0 and len(tokens) < self.min_length:
            return None
        
        # Truncate by max_length
        if self.max_length > 0 and len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
            text = self.tokenizer.decode(tokens, skip_special_tokens=True)
        
        return text
    
    def load(self) -> List[str]:
        """Load and process examples from the dataset."""
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
                processed.append(filtered_text)
        
        return processed[:self.max_examples]
    
    def extract_text(self, item: Dict[str, Any]) -> str:
        """Extract text from a data item. Override for custom extraction."""
        return item.get('text', '')

