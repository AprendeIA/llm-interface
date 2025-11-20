from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from .base import BaseProvider, ProviderUtils
from ..core.config import LLMConfig
from ..core.exceptions import APIKeyError, ConfigurationError

class AzureProvider(BaseProvider):
    """Azure OpenAI provider implementation"""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.api_key = self._get_api_key("AZURE_OPENAI_API_KEY")
        self._validate_azure_config()
    
    def _validate_azure_config(self) -> None:
        """Validate Azure-specific configuration"""
        if not self.api_key:
            raise APIKeyError("azure", "AZURE_OPENAI_API_KEY")
        
        if not self.config.azure_endpoint:
            raise ConfigurationError("Azure endpoint is required for Azure provider")
        
        if not self.config.azure_deployment:
            raise ConfigurationError("Azure deployment name is required for Azure provider")
    
    def get_model(self) -> AzureChatOpenAI:
        """Get Azure OpenAI language model instance"""
        model_kwargs = ProviderUtils.get_model_kwargs(self.config)
        
        return AzureChatOpenAI(
            deployment_name=self.config.azure_deployment,
            api_key=self.api_key,
            azure_endpoint=self.config.azure_endpoint,
            api_version=self.config.azure_api_version or "2023-12-01-preview",
            temperature=model_kwargs.get("temperature", 0.7),
            max_tokens=model_kwargs.get("max_tokens", 1000)
        )
    
    def get_chat_model(self) -> AzureChatOpenAI:
        """Get Azure OpenAI chat model instance"""
        return self.get_model()
    
    def get_embeddings(self) -> AzureOpenAIEmbeddings:
        """Get Azure OpenAI embeddings model instance"""
        return AzureOpenAIEmbeddings(
            deployment=self.config.azure_deployment,
            api_key=self.api_key,
            azure_endpoint=self.config.azure_endpoint,
            api_version=self.config.azure_api_version or "2023-12-01-preview"
        )
    
    def validate_config(self) -> bool:
        """Validate Azure provider configuration"""
        return bool(
            self.api_key and 
            self.config.azure_endpoint and 
            self.config.azure_deployment
        )