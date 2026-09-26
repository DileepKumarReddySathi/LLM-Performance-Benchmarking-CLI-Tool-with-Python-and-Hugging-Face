import yaml
import json
import os
from dataclasses import dataclass
from typing import List

@dataclass
class BenchmarkConfig:
    models: List[str]
    dataset_path: str
    max_new_tokens: int = 50

def load_config(file_path: str) -> BenchmarkConfig:
    """
    Parses the YAML or JSON file and returns a validated configuration object.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Configuration file not found: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        if file_path.endswith('.json'):
            data = json.load(f)
        elif file_path.endswith('.yaml') or file_path.endswith('.yml'):
            data = yaml.safe_load(f)
        else:
            raise ValueError("Configuration file must be a .yaml or .json file.")

    if 'models' not in data or not isinstance(data['models'], list):
        raise ValueError("Configuration must contain a 'models' list.")
    if len(data['models']) < 3:
        raise ValueError("Configuration must contain a list of at least three Hugging Face model identifiers.")
    if 'dataset_path' not in data or not isinstance(data['dataset_path'], str):
        raise ValueError("Configuration must contain a 'dataset_path' string.")

    max_new_tokens = data.get('max_new_tokens', 50)
    
    return BenchmarkConfig(
        models=data['models'],
        dataset_path=data['dataset_path'],
        max_new_tokens=max_new_tokens
    )
