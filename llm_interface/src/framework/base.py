"""Base framework adapter class.

This module provides the abstract base class that all framework adapters
must implement. It defines the common interface for working with different
AI frameworks while using the unified LLM provider interface.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from ..manager import LLMManager
from .exceptions import FrameworkConfigurationError


class FrameworkAdapter(ABC):
    """Abstract base class for framework adapters.
    
    All framework adapters must inherit from this class and implement
    the abstract methods to provide a unified interface to different
    AI frameworks while using LLM providers managed by LLMManager.
    
    Attributes:
        llm_manager: The LLMManager instance managing providers
        config: Framework-specific configuration dictionary
        name: Human-readable framework name
    """
    
    def __init__(self, llm_manager: LLMManager, config: Dict[str, Any] = None):
        """Initialize framework adapter.
        
        Args:
            llm_manager: LLMManager instance with configured providers
            config: Optional framework-specific configuration
            
        Raises:
            FrameworkConfigurationError: If configuration is invalid
        """
        if llm_manager is None:
            raise FrameworkConfigurationError(
                "llm_manager cannot be None"
            )
        
        self.llm_manager = llm_manager
        self.config = config or {}
        
        # Validate configuration
        if not self.validate_config(self.config):
            raise FrameworkConfigurationError(
                f"Invalid configuration for {self.framework_name}"
            )
    
    @property
    @abstractmethod
    def framework_name(self) -> str:
        """Return framework identifier.
        
        Returns:
            str: Unique framework name (e.g., 'crewai', 'autogen')
        """
        pass
    
    @property
    @abstractmethod
    def framework_version(self) -> str:
        """Return minimum supported framework version.
        
        Returns:
            str: Version string (e.g., '0.1.0')
        """
        pass
    
    @abstractmethod
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate framework-specific configuration.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            bool: True if configuration is valid, False otherwise
        """
        pass
    
    @abstractmethod
    def create_model(self, provider_name: str, **kwargs):
        """Create framework-specific model instance.
        
        Args:
            provider_name: Name of the provider to use
            **kwargs: Additional framework-specific arguments
            
        Returns:
            Framework-specific model instance
            
        Raises:
            FrameworkError: If model creation fails
        """
        pass
    
    # Utility methods (framework-agnostic)
    
    def list_providers(self) -> List[str]:
        """List available providers from manager.
        
        Returns:
            List[str]: Names of registered providers
        """
        return self.llm_manager.list_providers()
    
    def get_provider_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about available providers.
        
        Returns:
            Dict containing provider type and model information
        """
        info = {}
        for name, provider in self.llm_manager.providers.items():
            info[name] = {
                'type': provider.config.provider.value,
                'model': provider.config.model_name,
                'supports_embeddings': provider.config.provider.value != 'anthropic'
            }
        return info
    
    def has_provider(self, provider_name: str) -> bool:
        """Check if provider is registered.
        
        Args:
            provider_name: Name to check
            
        Returns:
            bool: True if provider exists
        """
        return provider_name in self.llm_manager.list_providers()
    
    def get_chat_model(self, provider_name: str):
        """Get chat model from provider.
        
        Args:
            provider_name: Name of provider
            
        Returns:
            Chat model instance
        """
        return self.llm_manager.get_chat_model(provider_name)
    
    def get_embeddings(self, provider_name: str):
        """Get embeddings model from provider.
        
        Args:
            provider_name: Name of provider
            
        Returns:
            Embeddings model instance
        """
        return self.llm_manager.get_embeddings(provider_name)
    
    def get_default_provider(self) -> Optional[str]:
        """Get default provider (first available).
        
        Returns:
            str: Name of default provider, or None if none available
        """
        providers = self.list_providers()
        return providers[0] if providers else None
    
    def __repr__(self) -> str:
        """String representation.
        
        Returns:
            str: Representation string
        """
        return (
            f"{self.__class__.__name__}("
            f"framework={self.framework_name}, "
            f"providers={len(self.llm_manager.list_providers())}"
            f")"
        )
