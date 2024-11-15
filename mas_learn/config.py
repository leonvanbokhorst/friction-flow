from pathlib import Path
from typing import Dict, Any
import yaml


class Config:
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        self.config: Dict[str, Any] = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            return self._create_default_config()
        with open(self.config_path) as f:
            return yaml.safe_load(f)

    def _create_default_config(self) -> Dict[str, Any]:
        config = {
            "logging": {"level": "INFO", "file_path": "logs/mas.log"},
            "agents": {"default_llm": "llama3.2:latest", "timeout": 300},
            "storage": {"results_path": "data/results", "models_path": "data/models"},
        }
        return config
