"""
Test factory module
"""

import pytest
from unittest.mock import Mock, patch

from llm_interface import (
    LLMConfig, 
    ProviderType, 
    LLMProviderFactory,
    UnsupportedProviderError,
    ProviderValidationError,
    ConfigurationError,
    APIKeyError,
)


class TestLLMProviderFactory:
    """Test LLMProviderFactory functionality"""
    
    def test_factory_has_default_providers(self):
        """Test that factory has default providers registered"""
        factory = LLMProviderFactory()
        
        # Check that default providers are registered
        assert ProviderType.OPENAI in factory._providers
        assert ProviderType.AZURE in factory._providers
        assert ProviderType.OLLAMA in factory._providers
        assert ProviderType.ANTHROPIC in factory._providers
    
    def test_register_custom_provider(self):
        """Test registering a custom provider"""
        # Create a mock provider class that inherits from LLMProvider
        from llm_interface.src.core.interfaces import LLMProvider
        
        class CustomProvider(LLMProvider):
            def __init__(self, config):
                self.config = config
            
            def validate_config(self):
                return True
            
            def get_model(self):
                return Mock()
            
            def get_chat_model(self):
                return Mock()
            
            def get_embeddings(self):
                return Mock()
        
        # Save original providers
        original_providers = LLMProviderFactory._providers.copy()
        
        try:
            # We can't dynamically add to ProviderType enum, so we'll
            # test by re-registering an existing provider type
            # Test registering (OLLAMA is already registered, so this should override with warning)
            LLMProviderFactory.register_provider(ProviderType.OLLAMA, CustomProvider)
            
            # Verify it's registered
            assert ProviderType.OLLAMA in LLMProviderFactory._providers
            assert LLMProviderFactory._providers[ProviderType.OLLAMA] == CustomProvider
        finally:
            # Restore original providers
            LLMProviderFactory._providers = original_providers
    
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    def test_create_openai_provider(self):
        """Test creating an OpenAI provider"""
        # Setup
        config = LLMConfig(
            provider=ProviderType.OPENAI,
            model_name="gpt-4",
            api_key="test-key"
        )
        
        # Test - no mocking, actually create the provider
        provider = LLMProviderFactory.create_provider(config)
        
        # Verify
        from llm_interface.src.providers.openai import OpenAIProvider
        assert isinstance(provider, OpenAIProvider)
        assert provider.config == config
    
    @patch.dict('os.environ', {'AZURE_OPENAI_API_KEY': 'test-key'})
    def test_create_azure_provider(self):
        """Test creating an Azure provider"""
        # Setup
        config = LLMConfig(
            provider=ProviderType.AZURE,
            model_name="gpt-4",
            azure_endpoint="https://test.openai.azure.com",
            azure_deployment="gpt-4"
        )
        
        # Test
        provider = LLMProviderFactory.create_provider(config)
        
        # Verify
        from llm_interface.src.providers.azure import AzureProvider
        assert isinstance(provider, AzureProvider)
        assert provider.config == config
    
    def test_create_ollama_provider(self):
        """Test creating an Ollama provider"""
        # Setup
        config = LLMConfig(
            provider=ProviderType.OLLAMA,
            model_name="llama2",
            base_url="http://localhost:11434"
        )
        
        # Test
        provider = LLMProviderFactory.create_provider(config)
        
        # Verify
        from llm_interface.src.providers.ollama import OllamaProvider
        assert isinstance(provider, OllamaProvider)
        assert provider.config == config
    
    @patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test-key'})
    def test_create_anthropic_provider(self):
        """Test creating an Anthropic provider"""
        # Setup
        config = LLMConfig(
            provider=ProviderType.ANTHROPIC,
            model_name="claude-3-sonnet-20240229",
            api_key="test-key"
        )
        
        # Test
        provider = LLMProviderFactory.create_provider(config)
        
        # Verify
        from llm_interface.src.providers.anthropic import AnthropicProvider
        assert isinstance(provider, AnthropicProvider)
        assert provider.config == config
    
    def test_unsupported_provider(self):
        """Test creating provider with unsupported provider type"""
        # Create a fake provider type not in the factory
        from enum import Enum
        
        class FakeProviderType(Enum):
            FAKE = "fake_provider"
        
        # We need to create a config, but ProviderType enum won't accept our fake type
        # So we'll test by removing a real provider temporarily
        # Save original providers
        original_providers = LLMProviderFactory._providers.copy()
        
        try:
            # Remove OLLAMA temporarily to test unsupported provider
            if ProviderType.OLLAMA in LLMProviderFactory._providers:
                del LLMProviderFactory._providers[ProviderType.OLLAMA]
            
            config = LLMConfig(
                provider=ProviderType.OLLAMA,  # Now unsupported
                model_name="llama2"
            )
            
            # Should raise UnsupportedProviderError for unsupported provider
            with pytest.raises(UnsupportedProviderError, match="Unsupported provider"):
                LLMProviderFactory.create_provider(config)
        finally:
            # Restore original providers
            LLMProviderFactory._providers = original_providers
    
    def test_invalid_provider_config(self):
        """Test creating provider with invalid configuration (missing API key)"""
        # Setup - config without API key and no environment variable
        config = LLMConfig(
            provider=ProviderType.OPENAI,
            model_name="gpt-4"
        )
        
        # Test - should raise APIKeyError (not wrapped in ConfigurationError now)
        with pytest.raises(APIKeyError, match="API key is required"):
            LLMProviderFactory.create_provider(config)
    
    def test_factory_is_singleton_like(self):
        """Test that factory methods work as class methods"""
        # The factory should work without instantiation
        config = LLMConfig(
            provider=ProviderType.OPENAI,
            model_name="gpt-3.5-turbo"
        )
        
        # These should work without creating an instance
        assert hasattr(LLMProviderFactory, 'create_provider')
        assert hasattr(LLMProviderFactory, 'register_provider')
        assert hasattr(LLMProviderFactory, '_providers')
        
        # Should be able to access the providers dict
        assert isinstance(LLMProviderFactory._providers, dict)