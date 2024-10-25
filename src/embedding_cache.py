"""Module for caching embeddings with different storage strategies."""

from abc import ABC, abstractmethod
import hashlib
from typing import Dict, List, Optional


class EmbeddingCache(ABC):
    """Abstract base class for embedding caches."""

    @abstractmethod
    def get(self, key: str) -> Optional[List[float]]:
        """Retrieve an embedding from the cache."""

    @abstractmethod
    def set(self, key: str, value: List[float]) -> None:
        """Store an embedding in the cache."""

    @abstractmethod
    def clear(self) -> None:
        """Clear all entries from the cache."""


class InMemoryEmbeddingCache(EmbeddingCache):
    """In-memory implementation of the EmbeddingCache."""

    def __init__(self):
        self._cache: Dict[str, List[float]] = {}

    @staticmethod
    def get_stable_hash(text: str) -> str:
        """Generate a stable hash for the given text."""
        return hashlib.sha256(text.encode()).hexdigest()

    def get(self, key: str) -> Optional[List[float]]:
        """Retrieve an embedding from the cache using a hashed key."""
        hashed_key = self.get_stable_hash(key)
        return self._cache.get(hashed_key)

    def set(self, key: str, value: List[float]) -> None:
        """Store an embedding in the cache using a hashed key."""
        hashed_key = self.get_stable_hash(key)
        self._cache[hashed_key] = value

    def clear(self) -> None:
        """Clear all entries from the cache."""
        self._cache.clear()




