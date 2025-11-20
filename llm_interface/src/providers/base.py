import os
from abc import ABC, abstractmethod
from typing import Optional

from ..core.interfaces import LLMProvider
from ..core.config import LLMConfig
from ..core.exceptions import APIKeyError, ConfigurationError

class BaseProvider(LLMProvider, ABC):
    """Base provider class with common functionality"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        # No need for _validate_common_config - Pydantic handles it at config creation time
    
    def _get_api_key(self, env_var: str) -> str:
        """Get API key from config or environment variable"""
        if self.config.api_key:
            return self.config.api_key
        return os.getenv(env_var, "")
    
    def _validate_api_key(self, env_var: str, required: bool = True) -> bool:
        """Validate API key exists"""
        api_key = self._get_api_key(env_var)
        if required and not api_key:
            raise APIKeyError("provider", env_var)
        return bool(api_key)
    
    @abstractmethod
    def get_model(self):
        """Get the language model instance"""
        pass
    
    @abstractmethod
    def get_chat_model(self):
        """Get the chat model instance"""
        pass
    
    @abstractmethod
    def get_embeddings(self):
        """Get the embeddings model instance"""
        pass
    
    @abstractmethod
    def validate_config(self) -> bool:
        """Validate provider-specific configuration"""
        pass

# Common utility functions for providers
class ProviderUtils:
    @staticmethod
    def resolve_base_url(config_base_url: Optional[str], default_url: str) -> str:
        """Resolve base URL with fallback"""
        return config_base_url or default_url or "http://localhost:11434"
    
    @staticmethod
    def validate_endpoint(endpoint: Optional[str], name: str) -> str:
        """Validate and return endpoint"""
        if not endpoint:
            raise ConfigurationError(f"{name} is required")
        return endpoint
    
    @staticmethod
    def get_model_kwargs(config: LLMConfig) -> dict:
        """Get common model parameters"""
        kwargs = {
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
        }
        
        # Remove None values
        return {k: v for k, v in kwargs.items() if v is not None}
