import logging
import threading
from typing import Dict, List, Type, Any
from .providers.openai import OpenAIProvider
from .providers.azure import AzureProvider
from .providers.ollama import OllamaProvider
from .providers.anthropic import AnthropicProvider
from .core.config import LLMConfig, ProviderType
from .core.interfaces import LLMProvider
from .core.exceptions import (
    UnsupportedProviderError,
    ProviderValidationError,
    ConfigurationError,
    APIKeyError,
)
    

logger = logging.getLogger(__name__)

class LLMProviderFactory:
    """Enhanced factory class to create LLM providers with improved error handling and validation"""
    
    _providers: Dict[ProviderType, Type[LLMProvider]] = {
        ProviderType.OPENAI: OpenAIProvider,
        ProviderType.AZURE: AzureProvider,
        ProviderType.OLLAMA: OllamaProvider,
        ProviderType.ANTHROPIC: AnthropicProvider,
    }
    _lock = threading.RLock()
    
    @classmethod
    def register_provider(cls, provider_type: ProviderType, provider_class: Type[LLMProvider]) -> None:
        """Register a new provider type with validation"""
        if provider_type is None:
            raise ConfigurationError("Provider type cannot be None")
        
        if not isinstance(provider_type, ProviderType):
            raise TypeError(f"Expected ProviderType, got {type(provider_type).__name__}")
        
        if provider_class is None:
            raise ConfigurationError("Provider class cannot be None")
        
        if not hasattr(provider_class, '__call__'):
            raise TypeError("Provider class must be callable")
        
        # Check if provider_class implements the LLMProvider interface
        if hasattr(provider_class, '__bases__') and not issubclass(provider_class, LLMProvider):
            raise TypeError(f"Provider class must inherit from LLMProvider")
        
        with cls._lock:
            if provider_type in cls._providers:
                existing_class = cls._providers[provider_type]
                logger.warning(f"Overriding existing provider {provider_type.value}: {existing_class.__name__} -> {provider_class.__name__}")
            
            cls._providers[provider_type] = provider_class
            logger.info(f"Registered provider: {provider_type.value} -> {provider_class.__name__}")
    
    @classmethod
    def create_provider(cls, config: LLMConfig) -> LLMProvider:
        """Create a provider instance based on configuration with enhanced error handling"""
        if config is None:
            raise ConfigurationError("Configuration cannot be None")
        
        if not isinstance(config, LLMConfig):
            raise TypeError(f"Expected LLMConfig, got {type(config).__name__}")
        
        logger.info(f"Creating provider of type: {config.provider.value}")
        
        if config.provider not in cls._providers:
            available_providers = cls.list_supported_providers()
            available_list = [p.value for p in available_providers]
            raise UnsupportedProviderError(config.provider.value, available_list)
        
        try:
            with cls._lock:
                provider_class = cls._providers[config.provider]
            
            logger.debug(f"Instantiating {provider_class.__name__}")
            provider = provider_class(config)
            
            logger.debug(f"Validating configuration for {config.provider.value}")
            if not provider.validate_config():
                raise ProviderValidationError(
                    config.provider.value,
                    "Configuration validation failed. Please check your API keys and settings."
                )
            
            logger.info(f"Provider {config.provider.value} created and validated successfully")
            return provider
            
        except (ConfigurationError, UnsupportedProviderError, ProviderValidationError, APIKeyError):
            # Re-raise our custom exceptions as-is
            raise
        except Exception as e:
            logger.error(f"Failed to create provider {config.provider.value}: {e}")
            raise ConfigurationError(f"Failed to create provider {config.provider.value}: {e}") from e
    
    @classmethod
    def list_supported_providers(cls) -> List[ProviderType]:
        """Get list of all supported provider types"""
        with cls._lock:
            return list(cls._providers.keys())
    
    @classmethod
    def is_provider_supported(cls, provider_type: ProviderType) -> bool:
        """Check if a provider type is supported"""
        if not isinstance(provider_type, ProviderType):
            return False
        
        with cls._lock:
            return provider_type in cls._providers
    
    @classmethod
    def get_provider_class(cls, provider_type: ProviderType) -> Type[LLMProvider]:
        """Get the class for a specific provider type"""
        if not isinstance(provider_type, ProviderType):
            raise TypeError(f"Expected ProviderType, got {type(provider_type).__name__}")
        
        with cls._lock:
            if provider_type not in cls._providers:
                available_providers = cls.list_supported_providers()
                available_list = [p.value for p in available_providers]
                raise UnsupportedProviderError(provider_type.value, available_list)
            
            return cls._providers[provider_type]
    
    @classmethod
    def unregister_provider(cls, provider_type: ProviderType) -> None:
        """Unregister a provider type"""
        if not isinstance(provider_type, ProviderType):
            raise TypeError(f"Expected ProviderType, got {type(provider_type).__name__}")
        
        with cls._lock:
            if provider_type not in cls._providers:
                available_providers = cls.list_supported_providers()
                available_list = [p.value for p in available_providers]
                raise UnsupportedProviderError(provider_type.value, available_list)
            
            del cls._providers[provider_type]
            logger.info(f"Unregistered provider: {provider_type.value}")
    
    @classmethod
    def get_factory_info(cls) -> Dict[str, Any]:
        """Get information about the factory and registered providers"""
        with cls._lock:
            return {
                "total_providers": len(cls._providers),
                "supported_providers": [p.value for p in cls._providers.keys()],
                "provider_classes": {
                    p.value: cls_type.__name__ 
                    for p, cls_type in cls._providers.items()
                }
            }

