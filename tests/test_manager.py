"""
Test manager module
"""

import pytest
from unittest.mock import Mock, patch

from llm_interface import (
    LLMConfig, 
    ProviderType, 
    LLMManager,
    ProviderNotFoundError,
    EmbeddingsNotSupportedError,
    ProviderAlreadyExistsError,
    InvalidInputError,
)
from llm_interface.src.core.interfaces import LLMProvider


class MockProvider(LLMProvider):
    """Mock provider for testing"""
    
    def __init__(self, config):
        self.config = config
        self._model = Mock()
        self._chat_model = Mock()
        self._embeddings = Mock()
    
    def get_model(self):
        return self._model
    
    def get_chat_model(self):
        return self._chat_model
    
    def get_embeddings(self):
        return self._embeddings
    
    def validate_config(self):
        return True


class TestLLMManager:
    """Test LLMManager functionality"""
    
    def test_manager_initialization(self):
        """Test that manager initializes correctly"""
        manager = LLMManager()
        
        assert isinstance(manager.providers, dict)
        assert isinstance(manager.models, dict)
        assert isinstance(manager.chat_models, dict)
        assert isinstance(manager.embeddings, dict)
        assert len(manager.providers) == 0
        assert manager.list_providers() == []
    
    @patch('llm_interface.src.manager.LLMProviderFactory.create_provider')
    def test_add_provider(self, mock_create_provider):
        """Test adding a provider to the manager"""
        # Setup
        manager = LLMManager()
        config = LLMConfig(
            provider=ProviderType.OPENAI,
            model_name="gpt-3.5-turbo"
        )
        
        mock_provider = MockProvider(config)
        mock_create_provider.return_value = mock_provider
        
        # Test
        manager.add_provider("test_openai", config)
        
        # Verify
        assert "test_openai" in manager.providers
        assert "test_openai" in manager.models
        assert "test_openai" in manager.chat_models
        assert "test_openai" in manager.embeddings
        assert manager.list_providers() == ["test_openai"]
        mock_create_provider.assert_called_once_with(config)
    
    @patch('llm_interface.src.manager.LLMProviderFactory.create_provider')
    def test_add_provider_with_embeddings_error(self, mock_create_provider):
        """Test adding a provider where embeddings fail"""
        # Setup
        manager = LLMManager()
        config = LLMConfig(
            provider=ProviderType.OLLAMA,
            model_name="llama2"
        )
        
        mock_provider = MockProvider(config)
        mock_provider.get_embeddings = Mock(side_effect=EmbeddingsNotSupportedError("ollama"))
        mock_create_provider.return_value = mock_provider
        
        # Test
        manager.add_provider("test_ollama", config)
        
        # Verify
        assert "test_ollama" in manager.providers
        assert manager.embeddings["test_ollama"] is None
    
    @patch('llm_interface.src.manager.LLMProviderFactory.create_provider')
    def test_get_model(self, mock_create_provider):
        """Test getting a model by provider name"""
        # Setup
        manager = LLMManager()
        config = LLMConfig(
            provider=ProviderType.OPENAI,
            model_name="gpt-4"
        )
        
        mock_provider = MockProvider(config)
        mock_create_provider.return_value = mock_provider
        
        manager.add_provider("test_provider", config)
        
        # Test
        model = manager.get_model("test_provider")
        
        # Verify
        assert model == mock_provider._model
    
    def test_get_model_not_found(self):
        """Test getting a model for non-existent provider"""
        manager = LLMManager()
        
        with pytest.raises(ProviderNotFoundError, match="Provider 'nonexistent' not found"):
            manager.get_model("nonexistent")
    
    @patch('llm_interface.src.manager.LLMProviderFactory.create_provider')
    def test_get_chat_model(self, mock_create_provider):
        """Test getting a chat model by provider name"""
        # Setup
        manager = LLMManager()
        config = LLMConfig(
            provider=ProviderType.ANTHROPIC,
            model_name="claude-3-sonnet-20240229"
        )
        
        mock_provider = MockProvider(config)
        mock_create_provider.return_value = mock_provider
        
        manager.add_provider("test_anthropic", config)
        
        # Test
        chat_model = manager.get_chat_model("test_anthropic")
        
        # Verify
        assert chat_model == mock_provider._chat_model
    
    def test_get_chat_model_not_found(self):
        """Test getting a chat model for non-existent provider"""
        manager = LLMManager()
        
        with pytest.raises(ProviderNotFoundError, match="Provider 'nonexistent' not found"):
            manager.get_chat_model("nonexistent")
    
    @patch('llm_interface.src.manager.LLMProviderFactory.create_provider')
    def test_get_embeddings(self, mock_create_provider):
        """Test getting embeddings by provider name"""
        # Setup
        manager = LLMManager()
        config = LLMConfig(
            provider=ProviderType.OPENAI,
            model_name="text-embedding-ada-002"
        )
        
        mock_provider = MockProvider(config)
        mock_create_provider.return_value = mock_provider
        
        manager.add_provider("test_embeddings", config)
        
        # Test
        embeddings = manager.get_embeddings("test_embeddings")
        
        # Verify
        assert embeddings == mock_provider._embeddings
    
    def test_get_embeddings_not_found(self):
        """Test getting embeddings for non-existent provider"""
        manager = LLMManager()
        
        with pytest.raises(ProviderNotFoundError, match="Provider 'nonexistent' not found"):
            manager.get_embeddings("nonexistent")
    
    @patch('llm_interface.src.manager.LLMProviderFactory.create_provider')
    def test_multiple_providers(self, mock_create_provider):
        """Test managing multiple providers"""
        # Setup
        manager = LLMManager()
        
        openai_config = LLMConfig(
            provider=ProviderType.OPENAI,
            model_name="gpt-4"
        )
        
        azure_config = LLMConfig(
            provider=ProviderType.AZURE,
            model_name="gpt-4",
            azure_endpoint="https://test.openai.azure.com",
            azure_deployment="gpt-4"
        )
        
        mock_create_provider.side_effect = [
            MockProvider(openai_config),
            MockProvider(azure_config)
        ]
        
        # Test
        manager.add_provider("openai", openai_config)
        manager.add_provider("azure", azure_config)
        
        # Verify
        providers = manager.list_providers()
        assert "openai" in providers
        assert "azure" in providers
        assert len(providers) == 2
    
    @patch('llm_interface.src.manager.LLMProviderFactory.create_provider')
    def test_create_chain(self, mock_create_provider):
        """Test creating a chain with a specific provider"""
        # Setup
        manager = LLMManager()
        config = LLMConfig(
            provider=ProviderType.OPENAI,
            model_name="gpt-3.5-turbo"
        )
        
        # Create a mock model
        mock_model = Mock()
        
        # Create a mock chain that will be returned from prompt | model
        mock_chain = Mock(spec=['invoke'])
        
        # Setup the mock provider
        mock_provider = MockProvider(config)
        mock_provider._model = mock_model
        mock_provider._chat_model = mock_model
        mock_create_provider.return_value = mock_provider
        
        manager.add_provider("test_provider", config)
        
        # Test
        with patch('llm_interface.src.manager.ChatPromptTemplate') as mock_prompt_template:
            mock_prompt = Mock()
            # Make the mock prompt support the | operator
            mock_prompt.__or__ = Mock(return_value=mock_chain)
            mock_prompt_template.from_template.return_value = mock_prompt
            
            chain = manager.create_chain("test_provider", "Test template: {input}")
            
            # Verify
            mock_prompt_template.from_template.assert_called_once_with("Test template: {input}")
            # Verify the | operator was called on the prompt with the model
            mock_prompt.__or__.assert_called_once_with(mock_model)
            assert chain == mock_chain