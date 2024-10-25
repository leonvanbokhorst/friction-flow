"""Module for caching embeddings."""

import hashlib
from typing import Dict, List, Optional


class EmbeddingCache:
    """Hash-based cache for storing embeddings."""

    def __init__(self):
        self._cache: Dict[str, List[float]] = {}

    @staticmethod
    def get_stable_hash(text: str) -> str:
        """Generate a stable hash for the given text."""
        return hashlib.sha256(text.encode()).hexdigest()

    def get(self, key: str) -> Optional[List[float]]:
        """Retrieve an embedding from the cache using a hashed key."""
        return self._cache.get(self.get_stable_hash(key))

    def set(self, key: str, value: List[float]) -> None:
        """Store an embedding in the cache using a hashed key."""
        self._cache[self.get_stable_hash(key)] = value

    def clear(self) -> None:
        """Clear all entries from the cache."""
        self._cache.clear()
