import pytest
from src.embedding_cache import EmbeddingCache

@pytest.fixture
def cache():
    """Fixture to provide a fresh EmbeddingCache instance for each test."""
    return EmbeddingCache()

def test_get_stable_hash():
    text = "Hello, world!"
    # This hash is pre-computed and should remain stable
    expected_hash = "315f5bdb76d078c43b8ac0064e4a0164612b1fce77c869345bfc94c75894edd3"
    assert EmbeddingCache.get_stable_hash(text) == expected_hash

def test_set_and_get(cache):
    key = "test_key"
    value = [0.1, 0.2, 0.3]
    cache.set(key, value)
    assert cache.get(key) == value

def test_get_nonexistent_key(cache):
    # Should return None for keys that haven't been set
    assert cache.get("nonexistent_key") is None

def test_clear(cache):
    # Set up the cache with some values
    cache.set("key1", [1.0, 2.0])
    cache.set("key2", [3.0, 4.0])
    
    cache.clear()
    
    # Verify that the cache is empty after clearing
    assert cache.get("key1") is None
    assert cache.get("key2") is None

def test_multiple_sets_same_key(cache):
    key = "test_key"
    value1 = [0.1, 0.2, 0.3]
    value2 = [0.4, 0.5, 0.6]
    
    cache.set(key, value1)
    cache.set(key, value2)
    
    # Verify that the most recent value is retrieved
    assert cache.get(key) == value2
