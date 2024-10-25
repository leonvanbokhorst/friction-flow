from pathlib import Path

APP_NAME: str = "lab-politik"
IS_DEVELOPMENT: bool = True

MODEL_CONFIGS = {
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
            "n_ctx": 4096,
            "metal_device": "mps",
            "main_gpu": 0,
            "use_metal": True,
            "n_threads": 4,
        },
    },
}
