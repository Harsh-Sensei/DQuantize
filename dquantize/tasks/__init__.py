from .base import BaseDatasetLoader
from typing import List, Dict, Any, Optional, Callable

_DATASET_REGISTRY: Dict[str, "BaseDatasetLoader"] = {}

def register_dataset(name: str):
    # Register a new dataset
    def decorator(cls):
        _DATASET_REGISTRY[name] = cls
        return cls
    return decorator


def get_available_datasets() -> List[str]:
    """Return list of available dataset names."""
    return list(_DATASET_REGISTRY.keys())


def get_dataset(name, **kwargs):
    try:
        return DATASET_REGISTRY[name.lower()](**kwargs)
    except KeyError:
        raise ValueError(f"Unknown dataset '{name}'. Available: {list(DATASET_REGISTRY.keys())}")


def load_dataset_examples(
    dataset_name: str,
    tokenizer,
    max_examples: int,
    max_length: int,
    min_length: int = 0,
    **kwargs
) -> List[str]:
    """
    Load examples from a dataset.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'toy', 'wikitext2', 'gsm8k', 'humaneval')
        tokenizer: Tokenizer to use
        max_examples: Maximum number of examples to load
        max_length: Maximum length of each example (0 for no limit)
        min_length: Minimum length of each example in tokens (0 for no limit)
        **kwargs: Additional arguments passed to the dataset loader
    
    Returns:
        List of text strings
    
    Raises:
        ValueError: If dataset_name is not registered
    """
    if dataset_name not in _DATASET_REGISTRY:
        available = get_available_datasets()
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Available datasets: {available}"
        )
    
    loader_cls = _DATASET_REGISTRY[dataset_name]
    loader = loader_cls(tokenizer, max_examples, max_length, min_length, **kwargs)

    return loader.load()


# import datasets
from . import toy, wikitext, gsm8k