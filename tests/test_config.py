"""
Test configuration module
"""

import pytest

from llm_interface import LLMConfig, ProviderType


class TestProviderType:
    """Test ProviderType enum"""
    
    def test_provider_types_exist(self):
        """Test that all expected provider types exist"""
        assert ProviderType.OPENAI.value == "openai"
        assert ProviderType.AZURE.value == "azure"
        assert ProviderType.OLLAMA.value == "ollama"
        assert ProviderType.ANTHROPIC.value == "anthropic"
        # GEMINI removed - not yet implemented
    
    def test_provider_types_are_strings(self):
        """Test that provider type values are strings"""
        for provider in ProviderType:
            assert isinstance(provider.value, str)


class TestLLMConfig:
    """Test LLMConfig dataclass"""
    
    def test_basic_config_creation(self):
        """Test creating a basic configuration"""
        config = LLMConfig(
            provider=ProviderType.OPENAI,
            model_name="gpt-3.5-turbo"
        )
        
        assert config.provider == ProviderType.OPENAI
        assert config.model_name == "gpt-3.5-turbo"
        assert config.api_key is None
        assert config.base_url is None
        assert config.temperature == 0.7  # Default value
        assert config.max_tokens == 1000  # Default value
    
    def test_config_with_all_fields(self):
        """Test creating a configuration with all fields"""
        config = LLMConfig(
            provider=ProviderType.AZURE,
            model_name="gpt-4",
            api_key="test-key",
            base_url="https://test.openai.azure.com",
            temperature=0.5,
            max_tokens=2000,
            azure_endpoint="https://test.openai.azure.com",
            azure_deployment="gpt-4-deployment",
            azure_api_version="2023-12-01-preview"
        )
        
        assert config.provider == ProviderType.AZURE
        assert config.model_name == "gpt-4"
        assert config.api_key == "test-key"
        assert config.base_url == "https://test.openai.azure.com"
        assert config.temperature == 0.5
        assert config.max_tokens == 2000
        assert config.azure_endpoint == "https://test.openai.azure.com"
        assert config.azure_deployment == "gpt-4-deployment"
        assert config.azure_api_version == "2023-12-01-preview"
    
    def test_config_defaults(self):
        """Test that default values are set correctly"""
        config = LLMConfig(
            provider=ProviderType.OLLAMA,
            model_name="llama2"
        )
        
        assert config.temperature == 0.7
        assert config.max_tokens == 1000
        assert config.api_key is None
        assert config.base_url is None
        assert config.azure_endpoint is None
        assert config.azure_deployment is None
        assert config.azure_api_version is None
    
    def test_config_immutability(self):
        """Test that config is mutable with Pydantic validation"""
        config = LLMConfig(
            provider=ProviderType.OPENAI,
            model_name="gpt-3.5-turbo"
        )
        
        # Should be able to modify (Pydantic models are mutable with validation)
        config.temperature = 0.8
        assert config.temperature == 0.8
        
        # Should have proper string representation (Pydantic format)
        config_str = str(config)
        # Pydantic BaseModel uses different repr format than dataclass
        assert "provider=" in config_str or "model_name=" in config_str
        assert "gpt-3.5-turbo" in config_str