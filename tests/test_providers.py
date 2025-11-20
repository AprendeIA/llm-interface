"""
Test base provider functionality
"""

import pytest
import os
from unittest.mock import Mock, patch
from pydantic import ValidationError

from llm_interface import LLMConfig, ProviderType, APIKeyError, ConfigurationError
from llm_interface.src.providers.base import BaseProvider, ProviderUtils
from llm_interface.src.core.interfaces import LLMProvider


class ConcreteProvider(BaseProvider):
    """Concrete implementation of BaseProvider for testing"""
    
    def get_model(self):
        return Mock()
    
    def get_chat_model(self):
        return Mock()
    
    def get_embeddings(self):
        return Mock()
    
    def validate_config(self):
        return True


class TestBaseProvider:
    """Test BaseProvider functionality"""
    
    def test_base_provider_initialization(self):
        """Test that base provider initializes correctly"""
        config = LLMConfig(
            provider=ProviderType.OPENAI,
            model_name="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=1000
        )
        
        provider = ConcreteProvider(config)
        
        assert provider.config == config
    
    def test_validate_common_config_valid(self):
        """Test common config validation with valid config"""
        config = LLMConfig(
            provider=ProviderType.OPENAI,
            model_name="gpt-4",
            temperature=0.5,
            max_tokens=2000
        )
        
        # Should not raise an exception
        provider = ConcreteProvider(config)
        assert provider.config.model_name == "gpt-4"
    
    def test_validate_common_config_missing_model_name(self):
        """Test common config validation with missing model name"""
        # With Pydantic, validation happens at config creation time
        with pytest.raises(ValidationError, match="model_name"):
            config = LLMConfig(
                provider=ProviderType.OPENAI,
                model_name="",  # Empty model name
                temperature=0.7,
                max_tokens=1000
            )
    
    def test_validate_common_config_invalid_temperature_low(self):
        """Test common config validation with temperature too low"""
        # With Pydantic, validation happens at config creation time
        with pytest.raises(ValidationError, match="greater than or equal"):
            config = LLMConfig(
                provider=ProviderType.OPENAI,
                model_name="gpt-4",
                temperature=-0.1,  # Invalid temperature
                max_tokens=1000
            )
    
    def test_validate_common_config_invalid_temperature_high(self):
        """Test common config validation with temperature too high"""
        # With Pydantic, validation happens at config creation time
        with pytest.raises(ValidationError, match="less than or equal"):
            config = LLMConfig(
                provider=ProviderType.OPENAI,
                model_name="gpt-4",
                temperature=2.1,  # Invalid temperature
                max_tokens=1000
            )
    
    def test_validate_common_config_invalid_max_tokens(self):
        """Test common config validation with invalid max tokens"""
        # With Pydantic, validation happens at config creation time
        with pytest.raises(ValidationError, match="greater than"):
            config = LLMConfig(
                provider=ProviderType.OPENAI,
                model_name="gpt-4",
                temperature=0.7,
                max_tokens=0  # Invalid max tokens
            )
    
    def test_get_api_key_from_config(self):
        """Test getting API key from config"""
        config = LLMConfig(
            provider=ProviderType.OPENAI,
            model_name="gpt-4",
            api_key="test-key-from-config"
        )
        
        provider = ConcreteProvider(config)
        api_key = provider._get_api_key("OPENAI_API_KEY")
        
        assert api_key == "test-key-from-config"
    
    @patch.dict(os.environ, {"TEST_API_KEY": "test-key-from-env"})
    def test_get_api_key_from_env(self):
        """Test getting API key from environment variable"""
        config = LLMConfig(
            provider=ProviderType.OPENAI,
            model_name="gpt-4"
            # No api_key in config
        )
        
        provider = ConcreteProvider(config)
        api_key = provider._get_api_key("TEST_API_KEY")
        
        assert api_key == "test-key-from-env"
    
    def test_get_api_key_none(self):
        """Test getting API key when none is available"""
        config = LLMConfig(
            provider=ProviderType.OPENAI,
            model_name="gpt-4"
            # No api_key in config
        )
        
        provider = ConcreteProvider(config)
        api_key = provider._get_api_key("NONEXISTENT_API_KEY")
        
        assert api_key == ""
    
    @patch.dict(os.environ, {"TEST_API_KEY": "test-key-from-env"})
    def test_validate_api_key_exists(self):
        """Test API key validation when key exists"""
        config = LLMConfig(
            provider=ProviderType.OPENAI,
            model_name="gpt-4"
        )
        
        provider = ConcreteProvider(config)
        result = provider._validate_api_key("TEST_API_KEY")
        
        assert result is True
    
    def test_validate_api_key_missing_required(self):
        """Test API key validation when required key is missing"""
        config = LLMConfig(
            provider=ProviderType.OPENAI,
            model_name="gpt-4"
        )
        
        provider = ConcreteProvider(config)
        
        with pytest.raises(APIKeyError):
            provider._validate_api_key("NONEXISTENT_API_KEY", required=True)
    
    def test_validate_api_key_missing_not_required(self):
        """Test API key validation when key is missing but not required"""
        config = LLMConfig(
            provider=ProviderType.OPENAI,
            model_name="gpt-4"
        )
        
        provider = ConcreteProvider(config)
        result = provider._validate_api_key("NONEXISTENT_API_KEY", required=False)
        
        assert result is False


class TestProviderUtils:
    """Test ProviderUtils functionality"""
    
    def test_resolve_base_url_with_config(self):
        """Test resolving base URL when provided in config"""
        result = ProviderUtils.resolve_base_url("https://custom.example.com", "https://default.example.com")
        assert result == "https://custom.example.com"
    
    def test_resolve_base_url_with_default(self):
        """Test resolving base URL when using default"""
        result = ProviderUtils.resolve_base_url(None, "https://default.example.com")
        assert result == "https://default.example.com"
    
    def test_resolve_base_url_fallback(self):
        """Test resolving base URL with fallback"""
        result = ProviderUtils.resolve_base_url(None, None)
        assert result == "http://localhost:11434"
    
    def test_validate_endpoint_exists(self):
        """Test validating endpoint when it exists"""
        result = ProviderUtils.validate_endpoint("https://api.example.com", "API Endpoint")
        assert result == "https://api.example.com"
    
    def test_validate_endpoint_missing(self):
        """Test validating endpoint when it's missing"""
        with pytest.raises(ConfigurationError, match="API Endpoint is required"):
            ProviderUtils.validate_endpoint(None, "API Endpoint")
    
    def test_get_model_kwargs(self):
        """Test getting model kwargs from config"""
        config = LLMConfig(
            provider=ProviderType.OPENAI,
            model_name="gpt-4",
            temperature=0.8,
            max_tokens=1500
        )
        
        kwargs = ProviderUtils.get_model_kwargs(config)
        
        assert kwargs["temperature"] == 0.8
        assert kwargs["max_tokens"] == 1500
        assert len(kwargs) == 2
    
    def test_get_model_kwargs_removes_none(self):
        """Test that get_model_kwargs removes None values"""
        config = LLMConfig(
            provider=ProviderType.OPENAI,
            model_name="gpt-4",
            temperature=0.7
            # max_tokens will use default 1000
        )
        
        kwargs = ProviderUtils.get_model_kwargs(config)
        
        # Should not contain None values
        for value in kwargs.values():
            assert value is not None
        
        assert "temperature" in kwargs
        assert "max_tokens" in kwargs