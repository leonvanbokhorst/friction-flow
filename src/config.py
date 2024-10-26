"""
Configuration module for the lab-politik application.

This module contains global configuration variables and model-specific
configurations for the application.
"""

from pathlib import Path
from typing import Dict, Any

# Application-wide constants
APP_NAME: str = "lab-politik"
IS_DEVELOPMENT: bool = True

# Model configuration dictionary
MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "balanced": {
        "chat": {
            "path": Path(
                "~/.cache/lm-studio/models/lmstudio-community/"
                "Mistral-Nemo-Instruct-2407-GGUF/"
                "Mistral-Nemo-Instruct-2407-Q4_K_M.gguf"
            ).expanduser(),
            "model_name": "mistral-nemo:latest",
        },
        "embedding": {
            "path": Path(
                "~/.cache/lm-studio/models/elliotsayes/"
                "mxbai-embed-large-v1-Q4_K_M-GGUF/"
                "mxbai-embed-large-v1-q4_k_m.gguf"
            ).expanduser(),
            "model_name": "mxbai-embed-large:latest",
        },
        "optimal_config": {
            "n_gpu_layers": -1,
            "n_batch": 512,
            "n_ctx": 16384,
            "metal_device": "mps",
            "main_gpu": 0,
            "use_metal": True,
            "n_threads": 8,
        },
    },
}


def get_model_config(config_name: str = "balanced") -> Dict[str, Any]:
    """
    Retrieve the model configuration for a given configuration name.

    Args:
        config_name (str): The name of the configuration to retrieve.
                           Defaults to "balanced".

    Returns:
        Dict[str, Any]: The model configuration dictionary.

    Raises:
        KeyError: If the specified config_name is not found in MODEL_CONFIGS.
    """
    if config_name not in MODEL_CONFIGS:
        raise KeyError(f"Configuration '{config_name}' not found in MODEL_CONFIGS")
    return MODEL_CONFIGS[config_name]
