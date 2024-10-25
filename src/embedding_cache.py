"""Module for caching embeddings."""

import hashlib
from typing import Dict, List, Optional, Union


class EmbeddingCache:
    """
    Hash-based cache for storing embeddings.

    This class provides a simple key-value store for embeddings, using SHA-256 hashes
    of the input text as keys. This ensures consistent lookup regardless of minor
    text variations.

    Attributes:
        _cache (Dict[str, List[float]]): Internal storage for hashed key-value pairs.
    """

    def __init__(self):
        self._cache: Dict[str, List[float]] = {}

    @staticmethod
    def get_stable_hash(text: str) -> str:
        """
        Generate a stable hash for the given text.

        Args:
            text (str): The input text to hash.

        Returns:
            str: A SHA-256 hash of the input text.
        """
        return hashlib.sha256(text.encode()).hexdigest()

    def get(self, key: str) -> Optional[List[float]]:
        """
        Retrieve an embedding from the cache using a hashed key.

        Args:
            key (str): The text key to look up.

        Returns:
            Optional[List[float]]: The stored embedding if found, None otherwise.
        """
        return self._cache.get(self.get_stable_hash(key))

    def set(self, key: str, value: List[float]) -> None:
        """
        Store an embedding in the cache using a hashed key.

        Args:
            key (str): The text key to associate with the embedding.
            value (List[float]): The embedding to store.
        """
        self._cache[self.get_stable_hash(key)] = value

    def clear(self) -> None:
        """Clear all entries from the cache."""
        self._cache.clear()

    def __len__(self) -> int:
        """
        Get the number of entries in the cache.

        Returns:
            int: The number of cached embeddings.
        """
        return len(self._cache)

    def __contains__(self, key: str) -> bool:
        """
        Check if a key exists in the cache.

        Args:
            key (str): The text key to check.

        Returns:
            bool: True if the key exists in the cache, False otherwise.
        """
        return self.get_stable_hash(key) in self._cache

    def update(self, items: Dict[str, List[float]]) -> None:
        """
        Update the cache with multiple key-value pairs.

        Args:
            items (Dict[str, List[float]]): A dictionary of text keys and their embeddings.
        """
        for key, value in items.items():
            self.set(key, value)
