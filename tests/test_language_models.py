import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

# Mock the config module
mock_MODEL_CONFIGS = {
    "balanced": {
        "chat": {"model_name": "test_chat_model", "path": Path("/path/to/chat/model")},
        "embedding": {
            "model_name": "test_embedding_model",
            "path": Path("/path/to/embedding/model"),
        },
        "optimal_config": {"n_ctx": 2048, "n_batch": 512},
    }
}

# Mock the EmbeddingCache
mock_EmbeddingCache = MagicMock()

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
        class ConcreteLanguageModel(LanguageModel):
            @async_error_handler
            async def generate(self, prompt: str) -> str:
                return f"Generated: {prompt}"
            
            @async_error_handler
            async def _generate_embedding(self, text: str) -> list[float]:
                if not text:
                    raise Exception("Test error")
                return [0.1, 0.2, 0.3]
        
        return ConcreteLanguageModel()

    @pytest.mark.asyncio
    async def test_generate_embedding_cached(self, concrete_language_model):
        text = "Test text"
        cached_embedding = [0.4, 0.5, 0.6]
        concrete_language_model.embedding_cache.get.return_value = cached_embedding
        
        result = await concrete_language_model.generate_embedding(text)
        assert result == cached_embedding

    @pytest.mark.asyncio
    async def test_generate_embedding_not_cached(self, concrete_language_model):
        text = "New test text"
        expected_embedding = [0.1, 0.2, 0.3]
        concrete_language_model.embedding_cache.get.return_value = None
        
        result = await concrete_language_model.generate_embedding(text)
        assert result == expected_embedding
        concrete_language_model.embedding_cache.set.assert_called_once_with(text, expected_embedding)

    @pytest.mark.asyncio
    async def test_generate_embedding_empty_text(self, concrete_language_model, caplog):
        result = await concrete_language_model.generate_embedding("")
        assert result == []
        assert "Attempted to generate embedding for empty text" in caplog.text


class TestOllamaInterface:
    @pytest.mark.asyncio
    async def test_init(self, mock_ollama, mock_config):
        ollama_interface = OllamaInterface()
        assert ollama_interface.chat_model_name == "test_chat_model"
        assert ollama_interface.embedding_model_name == "test_embedding_model"
        mock_ollama.ps.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate(self, mock_ollama, mock_config):
        ollama_interface = OllamaInterface()
        mock_ollama.chat.return_value = {"message": {"content": "Generated response"}}

        response = await ollama_interface.generate("Test prompt")
        assert response == "Generated response"
        mock_ollama.chat.assert_called_once_with(
            model="test_chat_model",
            messages=[{"role": "user", "content": "Test prompt"}],
        )

    @pytest.mark.asyncio
    async def test_generate_embedding(self, mock_ollama, mock_config):
        ollama_interface = OllamaInterface()
        mock_ollama.embeddings.return_value = {"embedding": [0.1, 0.2, 0.3]}

        embedding = await ollama_interface.generate_embedding("Test text")
        assert embedding == [0.1, 0.2, 0.3]
        mock_ollama.embeddings.assert_called_once_with(
            model="test_embedding_model", prompt="Test text"
        )


class TestLlamaInterface:
    @pytest.mark.asyncio
    async def test_init(self, mock_llama, mock_config):
        llama_interface = LlamaInterface()
        assert llama_interface.chat_model_path == Path('/path/to/chat/model')
        assert llama_interface.embedding_model_path == Path('/path/to/embedding/model')
        assert llama_interface.optimal_config == {'n_ctx': 2048, 'n_batch': 512}
        mock_llama.assert_called()

    @pytest.mark.asyncio
    async def test_generate(self, mock_llama, mock_config):
        llama_interface = LlamaInterface()
        mock_llama.return_value.create_chat_completion.return_value = {
            'choices': [{'message': {'content': 'Generated response'}}]
        }
        
        response = await llama_interface.generate("Test prompt")
        assert response == 'Generated response'
        mock_llama.return_value.create_chat_completion.assert_called_once_with(
            messages=[{'role': 'user', 'content': 'Test prompt'}]
        )

    @pytest.mark.asyncio
    async def test_generate_embedding(self, mock_llama, mock_config):
        llama_interface = LlamaInterface()
        mock_llama.return_value.embed.return_value = [0.1, 0.2, 0.3]
        
        embedding = await llama_interface.generate_embedding("Test text")
        assert embedding == [0.1, 0.2, 0.3]
        mock_llama.return_value.embed.assert_called_once_with("Test text")

    @pytest.mark.asyncio
    async def test_cleanup(self, mock_llama, mock_config):
        llama_interface = LlamaInterface()
        
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
