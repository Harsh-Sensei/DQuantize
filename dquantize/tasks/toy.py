from .base import BaseDatasetLoader
from . import register_dataset
from typing import List, Dict, Any, Optional, Callable

@register_dataset('toy')
class ToyDatasetLoader(BaseDatasetLoader):
    """Simple toy dataset with math word problems."""
    
    def load_raw_data(self) -> List[Dict[str, Any]]:
        prompts = [
            "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?",
            "Joy can read 8 pages of a book in 20 minutes. How many hours will it take her to read 120 pages?",
            "Randy has 60 mango trees on his farm. He also has 5 less than half as many coconut trees as mango trees. How many trees does Randy have in all on his farm?",
        ]
        return [{'text': p} for p in prompts]



