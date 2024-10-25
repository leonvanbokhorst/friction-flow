from abc import ABC, abstractmethod
from typing import Dict, List
import asyncio
import gc
import logging
from pathlib import Path

import ollama
import torch
from llama_cpp import Llama
from config import MODEL_CONFIGS

class LanguageModel(ABC):
    """Abstract base class for language models."""

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    async def generate(self, prompt: str) -> str:
        """Generate a response for the given prompt."""
        pass

    @abstractmethod
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate an embedding for the given text."""
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up resources used by the model."""
        pass


class OllamaInterface(LanguageModel):
    """Interface for the Ollama language model."""

    def __init__(self, quality_preset: str = "balanced"):
        """Initialize the OllamaInterface."""
        super().__init__()
        try:
            self.chat_model_path = MODEL_CONFIGS[quality_preset]["chat"]["model_name"]
            self.embedding_model_path = MODEL_CONFIGS[quality_preset]["embedding"][
                "model_name"
            ]
            self.embedding_cache: Dict[int, List[float]] = {}
            self.logger.info(
                f"Initialized OllamaInterface with {quality_preset} preset"
            )
        except KeyError as e:
            self.logger.error(f"Invalid quality preset: {quality_preset}")
            raise ValueError(f"Invalid quality preset: {quality_preset}") from e

    async def generate(self, prompt: str) -> str:
        """Generate a response for the given prompt."""
        self.logger.debug(f"Generating response for prompt: {prompt}")
        try:
            response = await asyncio.to_thread(
                ollama.chat,
                model=self.chat_model_path,
                messages=[{"role": "user", "content": prompt}],
            )
            self.logger.debug(f"Response from LLM: {response['message']['content']}")
            return response["message"]["content"]
        except Exception as e:
            self.logger.error(f"Error generating response: {e}", exc_info=True)
            raise e

    async def generate_embedding(self, text: str) -> List[float]:
        """Generate an embedding for the given text."""
        cache_key = hash(text)
        if cache_key in self.embedding_cache:
            self.logger.debug(f"Embedding found in cache for hash: {cache_key}")
            return self.embedding_cache[cache_key]

        self.logger.debug(f"Generating embedding for text: {text[:50]}...")
        try:
            response = await asyncio.to_thread(
                ollama.embeddings,
                model=self.embedding_model_path,
                prompt=text,
            )
            embedding = response["embedding"]
            self.embedding_cache[cache_key] = embedding
            self.logger.debug(f"Embedding generated and cached for hash: {cache_key}")
            return embedding
        except Exception as e:
            self.logger.error(f"Error generating embedding: {e}", exc_info=True)
            raise e

    async def cleanup(self) -> None:
        """Clean up resources used by the model."""
        self.logger.info("Cleaning up OllamaInterface resources")
        self.embedding_cache.clear()
        try:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.logger.debug("OllamaInterface cleanup completed")
        except Exception as e:
            self.logger.error(f"Error cleaning up resources: {e}", exc_info=True)


class LlamaInterface(LanguageModel):
    """Interface for the Llama language model."""

    def __init__(self, quality_preset: str = "balanced"):
        """Initialize the LlamaInterface."""
        super().__init__()
        try:
            self.quality_preset = quality_preset
            self.chat_model_path = MODEL_CONFIGS[quality_preset]["chat"]["path"]
            self.embedding_model_path = MODEL_CONFIGS[quality_preset]["embedding"][
                "path"
            ]
            self.optimal_config = MODEL_CONFIGS[quality_preset]["optimal_config"]
            self.embedding_cache: Dict[int, List[float]] = {}
            self.llm: Llama | None = None
            self.embedding_model: Llama | None = None
            self.setup_models()
            self.logger.info(f"Initialized LlamaInterface with {quality_preset} preset")
        except KeyError as e:
            self.logger.error(f"Invalid quality preset: {quality_preset}")
            raise ValueError(f"Invalid quality preset: {quality_preset}") from e

    def setup_models(self) -> None:
        """Set up the language models."""
        try:
            self.logger.info("Setting up Llama models")
            self.llm = Llama(
                model_path=str(self.chat_model_path),
                verbose=False,
                **self.optimal_config,
            )
            self.embedding_model = Llama(
                model_path=str(self.embedding_model_path),
                embedding=True,
                verbose=False,
                **self.optimal_config,
            )
            self.logger.info("Llama models set up successfully")
        except Exception as e:
            self.logger.error(f"Failed to load models: {e}", exc_info=True)
            raise e

    async def generate(self, prompt: str) -> str:
        """Generate a response for the given prompt."""
        self.logger.debug(f"Generating response for prompt: {prompt}")
        try:
            response = await asyncio.to_thread(
                self.llm.create_chat_completion,
                messages=[{"role": "user", "content": prompt}],
            )
            self.logger.debug(
                f"Response from LLM: {response['choices'][0]['message']['content']}"
            )
            return response["choices"][0]["message"]["content"]
        except Exception as e:
            self.logger.error(f"Error generating response: {e}", exc_info=True)
            raise e

    async def generate_embedding(self, text: str) -> List[float]:
        """Generate an embedding for the given text."""
        self.logger.debug(f"Generating embedding for text: {text[:50]}...")
        try:
            embedding = await asyncio.to_thread(self.embedding_model.embed, text)
            self.logger.debug(f"Embedding generated successfully")
            return embedding
        except Exception as e:
            self.logger.error(f"Error generating embedding: {e}", exc_info=True)
            raise e

    async def cleanup(self) -> None:
        """Clean up resources used by the model."""
        self.logger.info("Cleaning up LlamaInterface resources")
        try:
            if self.llm:
                del self.llm
            if self.embedding_model:
                del self.embedding_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.logger.debug("LlamaInterface cleanup completed")
        except Exception as e:
            self.logger.error(f"Error cleaning up resources: {e}", exc_info=True)
