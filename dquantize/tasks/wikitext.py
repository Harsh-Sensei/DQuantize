from .base import BaseDatasetLoader
from . import register_dataset
from datasets import load_dataset
from typing import List, Dict, Any, Optional, Callable

@register_dataset('wikitext2')
class WikiText2DatasetLoader(BaseDatasetLoader):
    """WikiText-2 dataset loader."""
    
    def load_raw_data(self) -> List[Dict[str, Any]]:
        print("Loading wikitext2 dataset...")
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
        return list(dataset)
    