import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from .base import BaseProvider, ProviderUtils
from ..core.config import LLMConfig
from ..core.exceptions import APIKeyError

class OpenAIProvider(BaseProvider):
    """OpenAI provider implementation"""
    
    def __init__(self, config: LLMConfig):
        # Call parent constructor - this handles common validation
        super().__init__(config)
        
        # OpenAI-specific initialization
        self.api_key = self._get_api_key("OPENAI_API_KEY")
        self._validate_openai_config()
    
    def _validate_openai_config(self) -> None:
        """Validate OpenAI-specific configuration"""
        if not self.api_key:
            raise APIKeyError("openai", "OPENAI_API_KEY")
        
    def get_model(self) -> ChatOpenAI:
        """Get OpenAI language model instance"""
        # Get common model parameters from base class utility
        model_kwargs = ProviderUtils.get_model_kwargs(self.config)
        
        # Create OpenAI model with configuration
        return ChatOpenAI(
            model=self.config.model_name,
            api_key=self.api_key,
            temperature=model_kwargs.get("temperature", 0.7),
            max_tokens=model_kwargs.get("max_tokens", None),  # OpenAI handles None
            **self._get_additional_kwargs()
        )
    
    def get_chat_model(self) -> ChatOpenAI:
        """Get OpenAI chat model instance"""
        # For OpenAI, chat model is the same as regular model
        return self.get_model()
    
    def get_embeddings(self) -> OpenAIEmbeddings:
        """Get OpenAI embeddings model instance"""
        return OpenAIEmbeddings(
            model=self.config.model_name.replace("gpt", "text-embedding"),
            api_key=self.api_key
        )
    
    def validate_config(self) -> bool:
        """Validate OpenAI provider configuration"""
        return bool(self.api_key and self.config.model_name)
    
    def _get_additional_kwargs(self) -> dict:
        """Get additional OpenAI-specific parameters"""
        kwargs = {}
        
        # Add any OpenAI-specific parameters from config
        if hasattr(self.config, 'openai_organization'):
            kwargs['organization'] = self.config.openai_organization
        
        if hasattr(self.config, 'openai_api_base'):
            kwargs['base_url'] = self.config.openai_api_base
        
        return kwargs

# Convenience functions for common use cases
class OpenAIProviderUtils:
    """Utility functions for OpenAI provider"""
    
    @staticmethod
    def create_gpt4_config(api_key: str = None, 
                          temperature: float = 0.7) -> LLMConfig:
        """Create configuration for GPT-4"""
        return LLMConfig(
            provider="openai",
            model_name="gpt-4",
            api_key=api_key,
            temperature=temperature
        )
    
    @staticmethod
    def create_gpt35_config(api_key: str = None,
                           temperature: float = 0.7) -> LLMConfig:
        """Create configuration for GPT-3.5"""
        return LLMConfig(
            provider="openai",
            model_name="gpt-3.5-turbo",
            api_key=api_key,
            temperature=temperature
        )
    
    @staticmethod
    def from_environment() -> 'OpenAIProvider':
        """Create OpenAI provider from environment variables"""
        config = LLMConfig(
            provider="openai",
            model_name=os.getenv("OPENAI_MODEL_NAME", "gpt-3.5-turbo"),
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "1000"))
        )
        return OpenAIProvider(config)
