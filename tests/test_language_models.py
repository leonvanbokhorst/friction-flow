import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

# Mock configuration for language models
# This simulates the structure of the actual MODEL_CONFIGS in the config module
mock_MODEL_CONFIGS = {
    "balanced": {  # Quality preset
        "chat": {
            "model_name": "test_chat_model",
            "path": Path("/path/to/chat/model")
        },
        "embedding": {
            "model_name": "test_embedding_model",
            "path": Path("/path/to/embedding/model")
        },
        "optimal_config": {
            "n_ctx": 2048,  # Context window size
            "n_batch": 512  # Batch size for processing
        }
    }
    # Additional quality presets could be added here
}

# Mock the EmbeddingCache class
# This allows us to simulate cache behavior without using a real cache in tests
mock_EmbeddingCache = MagicMock()
# Methods like get() and set() can be mocked on this object as needed in tests

# Patch both config and embedding_cache imports
with patch.dict("sys.modules", {
    "config": MagicMock(),
    "embedding_cache": MagicMock(),
    "torch": MagicMock(),  # Mock torch
    "llama_cpp": MagicMock()  # Mock llama_cpp
}):
    import sys

    sys.modules["config"].MODEL_CONFIGS = mock_MODEL_CONFIGS
    sys.modules["embedding_cache"].EmbeddingCache = mock_EmbeddingCache
    from src.language_models import (
        LanguageModel,
        OllamaInterface,
        LlamaInterface,
        ModelError,
        ModelInitializationError,
        async_error_handler,  # Add this import
    )


@pytest.fixture
def mock_ollama():
    with patch("src.language_models.ollama") as mock:
        yield mock


@pytest.fixture
def mock_llama():
    with patch("src.language_models.Llama") as mock:
        yield mock


@pytest.fixture(autouse=True)
def mock_config():
    with patch("src.language_models.MODEL_CONFIGS", mock_MODEL_CONFIGS):
        yield


class TestLanguageModel:
    @pytest.fixture
    def concrete_language_model(self):
        # Define a concrete implementation of LanguageModel for testing
        class ConcreteLanguageModel(LanguageModel):
            @async_error_handler
            async def generate(self, prompt: str) -> str:
                return f"Generated: {prompt}"
            
            @async_error_handler
            async def _generate_embedding(self, text: str) -> list[float]:
                # Simulate error for empty text
                if not text:
                    raise Exception("Test error")
                return [0.1, 0.2, 0.3]
        
        return ConcreteLanguageModel()

    @pytest.mark.asyncio
    async def test_generate_embedding_cached(self, concrete_language_model):
        # Test when the embedding is already in the cache
        text = "Test text"
        cached_embedding = [0.4, 0.5, 0.6]
        concrete_language_model.embedding_cache.get.return_value = cached_embedding
        
        result = await concrete_language_model.generate_embedding(text)
        assert result == cached_embedding
        # No need to check if set() was called, as it shouldn't be for cached results

    @pytest.mark.asyncio
    async def test_generate_embedding_not_cached(self, concrete_language_model):
        # Test when the embedding is not in the cache
        text = "New test text"
        expected_embedding = [0.1, 0.2, 0.3]
        concrete_language_model.embedding_cache.get.return_value = None
        
        result = await concrete_language_model.generate_embedding(text)
        assert result == expected_embedding
        # Verify that the new embedding was cached
        concrete_language_model.embedding_cache.set.assert_called_once_with(text, expected_embedding)

    @pytest.mark.asyncio
    async def test_generate_embedding_empty_text(self, concrete_language_model, caplog):
        # Test behavior when given empty text
        result = await concrete_language_model.generate_embedding("")
        assert result == []
        # Check that the appropriate warning was logged
        assert "Attempted to generate embedding for empty text" in caplog.text


class TestOllamaInterface:
    @pytest.mark.asyncio
    async def test_init(self, mock_ollama, mock_config):
        # Test initialization of OllamaInterface
        ollama_interface = OllamaInterface()
        # Verify that model names are correctly set from the mock config
        assert ollama_interface.chat_model_name == "test_chat_model"
        assert ollama_interface.embedding_model_name == "test_embedding_model"
        # Ensure that the Ollama process status check is called during initialization
        mock_ollama.ps.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate(self, mock_ollama, mock_config):
        # Test the generate method of OllamaInterface
        ollama_interface = OllamaInterface()
        # Mock the response from Ollama's chat method
        mock_ollama.chat.return_value = {"message": {"content": "Generated response"}}

        # Call the generate method and check the response
        response = await ollama_interface.generate("Test prompt")
        assert response == "Generated response"
        # Verify that Ollama's chat method was called with correct parameters
        mock_ollama.chat.assert_called_once_with(
            model="test_chat_model",
            messages=[{"role": "user", "content": "Test prompt"}],
        )

    @pytest.mark.asyncio
    async def test_generate_embedding(self, mock_ollama, mock_config):
        # Test the generate_embedding method of OllamaInterface
        ollama_interface = OllamaInterface()
        # Mock the response from Ollama's embeddings method
        mock_ollama.embeddings.return_value = {"embedding": [0.1, 0.2, 0.3]}

        # Call the generate_embedding method and check the result
        embedding = await ollama_interface.generate_embedding("Test text")
        assert embedding == [0.1, 0.2, 0.3]
        # Verify that Ollama's embeddings method was called with correct parameters
        mock_ollama.embeddings.assert_called_once_with(
            model="test_embedding_model", prompt="Test text"
        )


class TestLlamaInterface:
    @pytest.mark.asyncio
    async def test_init(self, mock_llama, mock_config):
        # Test initialization of LlamaInterface
        llama_interface = LlamaInterface()
        # Verify that model paths and configurations are correctly set from the mock config
        assert llama_interface.chat_model_path == Path('/path/to/chat/model')
        assert llama_interface.embedding_model_path == Path('/path/to/embedding/model')
        assert llama_interface.optimal_config == {'n_ctx': 2048, 'n_batch': 512}
        # Ensure that the Llama model is initialized during interface creation
        mock_llama.assert_called()

    @pytest.mark.asyncio
    async def test_generate(self, mock_llama, mock_config):
        # Test the generate method of LlamaInterface
        llama_interface = LlamaInterface()
        # Mock the response from Llama's create_chat_completion method
        mock_llama.return_value.create_chat_completion.return_value = {
            'choices': [{'message': {'content': 'Generated response'}}]
        }
        
        # Call the generate method and check the response
        response = await llama_interface.generate("Test prompt")
        assert response == 'Generated response'
        # Verify that Llama's create_chat_completion method was called with correct parameters
        mock_llama.return_value.create_chat_completion.assert_called_once_with(
            messages=[{'role': 'user', 'content': 'Test prompt'}]
        )

    @pytest.mark.asyncio
    async def test_generate_embedding(self, mock_llama, mock_config):
        # Test the generate_embedding method of LlamaInterface
        llama_interface = LlamaInterface()
        # Mock the response from Llama's embed method
        mock_llama.return_value.embed.return_value = [0.1, 0.2, 0.3]
        
        # Call the generate_embedding method and check the result
        embedding = await llama_interface.generate_embedding("Test text")
        assert embedding == [0.1, 0.2, 0.3]
        # Verify that Llama's embed method was called with correct parameters
        mock_llama.return_value.embed.assert_called_once_with("Test text")

    @pytest.mark.asyncio
    async def test_cleanup(self, mock_llama, mock_config):
        # Test the cleanup method of LlamaInterface
        llama_interface = LlamaInterface()
        
        # Call the cleanup method
        await llama_interface.cleanup()
        
        # Check that the models have been deleted
        assert not hasattr(llama_interface, 'llm')
        assert not hasattr(llama_interface, 'embedding_model')


@pytest.mark.asyncio
async def test_model_error():
    class ErrorModel(LanguageModel):
        @async_error_handler
        async def generate(self, prompt: str) -> str:
            raise Exception("Test error")
        
        @async_error_handler
        async def _generate_embedding(self, text: str) -> list[float]:
            raise Exception("Test error")

    error_model = ErrorModel()
    with pytest.raises(ModelError):
        await error_model.generate("Test prompt")
    
    with pytest.raises(ModelError):
        await error_model.generate_embedding("Test text")


@pytest.mark.asyncio
async def test_model_error():
    class ErrorModel(LanguageModel):
        @async_error_handler
        async def generate(self, prompt: str) -> str:
            raise Exception("Test error")
        
        @async_error_handler
        async def _generate_embedding(self, text: str) -> list[float]:
            raise Exception("Test error")

    error_model = ErrorModel()
    with pytest.raises(ModelError):
        await error_model.generate("Test prompt")
    
    with pytest.raises(ModelError):
        await error_model.generate_embedding("Test text")

    error_model = ErrorModel()
    with pytest.raises(ModelError):
        await error_model.generate("Test prompt")
    
    with pytest.raises(ModelError):
        await error_model.generate_embedding("Test text")


def test_model_initialization_error(mock_config):
    with pytest.raises(ModelInitializationError):
        OllamaInterface(quality_preset="invalid_preset")

    with pytest.raises(ModelInitializationError):
        LlamaInterface(quality_preset="invalid_preset")
