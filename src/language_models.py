from __future__ import annotations
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
from embedding_cache import EmbeddingCache, InMemoryEmbeddingCache


class ModelInitializationError(Exception):
    """Custom exception for model initialization errors."""

    pass


def async_error_handler(func):
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            logging.error(f"Error in {func.__name__}: {e}", exc_info=True)
            raise

    return wrapper


class LanguageModel(ABC):
    """Abstract base class for language models."""

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.embedding_cache: EmbeddingCache = InMemoryEmbeddingCache()

    @abstractmethod
    async def generate(self, prompt: str) -> str:
        """Generate a response for the given prompt."""
        pass

    @abstractmethod
    async def _generate_embedding(self, text: str) -> List[float]:
        """Internal method to generate an embedding."""
        pass

    @async_error_handler
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate an embedding for the given text."""
        if not text:
            self.logger.warning("Attempted to generate embedding for empty text")
            return []

        cached_embedding = self.embedding_cache.get(text)
        if cached_embedding:
            self.logger.info(f"Embedding found in cache for text: {text[:50]}...")
            return cached_embedding

        self.logger.info(f"Generating embedding for text: {text[:50]}...")
        try:
            embedding = await self._generate_embedding(text)
            self.embedding_cache.set(text, embedding)
            self.logger.info(f"Embedding generated and cached for text: {text[:50]}...")
            return embedding
        except Exception as e:
            self.logger.error(f"Failed to generate embedding: {e}", exc_info=True)
            raise

    async def cleanup(self) -> None:
        """Clean up resources used by the model."""
        self.logger.info(f"Cleaning up {self.__class__.__name__} resources")
        self.embedding_cache.clear()
        try:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.logger.info(f"{self.__class__.__name__} cleanup completed")
        except Exception as e:
            self.logger.error(f"Error cleaning up resources: {e}", exc_info=True)
            # We don't re-raise here as cleanup errors shouldn't stop the program


class OllamaInterface(LanguageModel):
    """Interface for the Ollama language model."""

    def __init__(self, quality_preset: str = "balanced"):
        super().__init__()
        try:
            self.chat_model_path = MODEL_CONFIGS[quality_preset]["chat"]["model_name"]
            self.embedding_model_path = MODEL_CONFIGS[quality_preset]["embedding"][
                "model_name"
            ]
        except KeyError as e:
            raise ModelInitializationError(
                f"Invalid quality preset or missing configuration: {e}"
            )

    @async_error_handler
    async def generate(self, prompt: str) -> str:
        """Generate a response for the given prompt."""
        if not prompt:
            self.logger.warning("Attempted to generate response for empty prompt")
            return ""

        self.logger.info(f"Generating response for prompt: {prompt[:50]}...")
        try:
            response = await asyncio.to_thread(
                ollama.chat,
                model=self.chat_model_path,
                messages=[{"role": "user", "content": prompt}],
            )
            self.logger.info("Response received from LLM")
            self.logger.debug(f"Full response: {response['message']['content']}")
            return response["message"]["content"]
        except Exception as e:
            self.logger.error(f"Failed to generate response: {e}", exc_info=True)
            raise

    async def _generate_embedding(self, text: str) -> List[float]:
        """Internal method to generate an embedding using Ollama."""
        response = await asyncio.to_thread(
            ollama.embeddings,
            model=self.embedding_model_path,
            prompt=text,
        )
        return response["embedding"]


class LlamaInterface(LanguageModel):
    """Interface for the Llama language model."""

    def __init__(self, quality_preset: str = "balanced"):
        super().__init__()
        try:
            self.chat_model_path = MODEL_CONFIGS[quality_preset]["chat"]["path"]
            self.embedding_model_path = MODEL_CONFIGS[quality_preset]["embedding"][
                "path"
            ]
            self.optimal_config = MODEL_CONFIGS[quality_preset]["optimal_config"]
        except KeyError as e:
            raise ModelInitializationError(
                f"Invalid quality preset or missing configuration: {e}"
            )
        self.llm: Llama | None = None
        self.embedding_model: Llama | None = None
        self.setup_models()

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
            raise ModelInitializationError(f"Failed to initialize Llama models: {e}")

    @async_error_handler
    async def generate(self, prompt: str) -> str:
        """Generate a response for the given prompt."""
        if not prompt:
            self.logger.warning("Attempted to generate response for empty prompt")
            return ""

        if not self.llm:
            raise ModelInitializationError("Llama model not initialized")

        self.logger.info(f"Generating response for prompt: {prompt[:50]}...")
        try:
            response = await asyncio.to_thread(
                self.llm.create_chat_completion,
                messages=[{"role": "user", "content": prompt}],
            )
            self.logger.info("Response received from LLM")
            self.logger.debug(
                f"Full response: {response['choices'][0]['message']['content']}"
            )
            return response["choices"][0]["message"]["content"]
        except Exception as e:
            self.logger.error(f"Failed to generate response: {e}", exc_info=True)
            raise

    async def _generate_embedding(self, text: str) -> List[float]:
        """Internal method to generate an embedding using Llama."""
        if not self.embedding_model:
            raise ModelInitializationError("Embedding model not initialized")
        return await asyncio.to_thread(self.embedding_model.embed, text)

    async def cleanup(self) -> None:
        """Clean up resources used by the model."""
        await super().cleanup()
        if self.llm:
            del self.llm
        if self.embedding_model:
            del self.embedding_model
