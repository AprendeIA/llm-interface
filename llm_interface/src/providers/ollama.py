try:
    # Try the new langchain-ollama package first (recommended)
    from langchain_ollama import ChatOllama, OllamaEmbeddings
except ImportError:
    # Fallback to langchain_community for backwards compatibility
    from langchain_community.chat_models import ChatOllama
    from langchain_community.embeddings import OllamaEmbeddings

from .base import BaseProvider
from ..core.config import LLMConfig
from ..core.exceptions import InvalidInputError, ConfigurationError

class OllamaProvider(BaseProvider):
    """Ollama provider implementation for local LLM models"""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self._validate_ollama_config()
    
    def _validate_ollama_config(self) -> None:
        """Validate Ollama-specific configuration"""
        if not self.config.model_name:
            raise InvalidInputError("Model name is required for Ollama provider")
        
        # Validate base_url format if provided
        if self.config.base_url:
            base_url = self.config.base_url.lower()
            if not (base_url.startswith('http://') or base_url.startswith('https://')):
                raise ConfigurationError(
                    f"Invalid base_url format: {self.config.base_url}. "
                    "Must start with http:// or https://"
                )
    
    def get_model(self) -> ChatOllama:
        """Get Ollama chat model instance"""
        return ChatOllama(
            model=self.config.model_name,
            base_url=self.config.base_url or "http://localhost:11434",
            temperature=self.config.temperature,
            num_predict=self.config.max_tokens
        )
    
    def get_chat_model(self) -> ChatOllama:
        """Get Ollama chat model instance (same as get_model for Ollama)"""
        return self.get_model()
    
    def get_embeddings(self):
        """Get Ollama embeddings model instance"""
        return OllamaEmbeddings(
            model=self.config.model_name,
            base_url=self.config.base_url or "http://localhost:11434"
        )
    
    def validate_config(self) -> bool:
        """Validate Ollama provider configuration"""
        return bool(self.config.model_name)
