from langchain_anthropic import ChatAnthropic
from .base import BaseProvider, ProviderUtils
from ..core.config import LLMConfig
from ..core.exceptions import APIKeyError, EmbeddingsNotSupportedError

class AnthropicProvider(BaseProvider):
    """Anthropic provider implementation"""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.api_key = self._get_api_key("ANTHROPIC_API_KEY")
        self._validate_anthropic_config()
    
    def _validate_anthropic_config(self) -> None:
        """Validate Anthropic-specific configuration"""
        if not self.api_key:
            raise APIKeyError("anthropic", "ANTHROPIC_API_KEY")
    
    def get_model(self) -> ChatAnthropic:
        """Get Anthropic language model instance"""
        model_kwargs = ProviderUtils.get_model_kwargs(self.config)
        
        return ChatAnthropic(
            model=self.config.model_name,
            api_key=self.api_key,
            temperature=model_kwargs.get("temperature", 0.7),
            max_tokens=model_kwargs.get("max_tokens", 1000)
        )
    
    def get_chat_model(self) -> ChatAnthropic:
        """Get Anthropic chat model instance"""
        return self.get_model()
    
    def get_embeddings(self):
        """Get Anthropic embeddings model instance"""
        raise EmbeddingsNotSupportedError("anthropic")
    
    def validate_config(self) -> bool:
        """Validate Anthropic provider configuration"""
        return bool(self.api_key and self.config.model_name)