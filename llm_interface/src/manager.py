import threading
from typing import Dict, List, Any, TYPE_CHECKING
from langchain_core.runnables import Runnable
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseLanguageModel

from .core.interfaces import LLMProvider
from .core.config import LLMConfig
from .core.exceptions import (
    ProviderNotFoundError,
    ProviderAlreadyExistsError,
    InvalidInputError,
    EmbeddingsNotSupportedError
)
from .factory import LLMProviderFactory

if TYPE_CHECKING:
    from .framework.base import FrameworkAdapter

class LLMManager:
    """Manager class to handle multiple LLM providers with thread safety"""
    
    def __init__(self):
        self.providers: Dict[str, LLMProvider] = {}
        self.models: Dict[str, BaseLanguageModel] = {}
        self.chat_models: Dict[str, BaseLanguageModel] = {}
        self.embeddings: Dict[str, Any] = {}
        self.frameworks: Dict[str, 'FrameworkAdapter'] = {}
        self._lock = threading.RLock()  # Reentrant lock for thread safety
    
    def add_provider(self, name: str, config: LLMConfig) -> None:
        """Add a new provider to the manager (thread-safe)"""
        if not name or not name.strip():
            raise InvalidInputError("name", "Provider name cannot be empty or whitespace")
        
        name = name.strip()  # Clean up whitespace
        
        with self._lock:
            if name in self.providers:
                raise ProviderAlreadyExistsError(name)
            
            provider = LLMProviderFactory.create_provider(config)
            self.providers[name] = provider
            # Pre-initialize models
            self.models[name] = provider.get_model()
            self.chat_models[name] = provider.get_chat_model()
            
            try:
                self.embeddings[name] = provider.get_embeddings()
            except (NotImplementedError, EmbeddingsNotSupportedError):
                self.embeddings[name] = None
    
    def get_model(self, provider_name: str) -> BaseLanguageModel:
        """Get a model by provider name (thread-safe)"""
        if not provider_name or not provider_name.strip():
            raise InvalidInputError("provider_name", "Provider name cannot be empty or whitespace")
        
        provider_name = provider_name.strip()
        
        with self._lock:
            if provider_name not in self.models:
                available_providers = self.list_providers()
                raise ProviderNotFoundError(provider_name, available_providers)
            
            return self.models[provider_name]
    
    def get_chat_model(self, provider_name: str) -> BaseLanguageModel:
        """Get a chat model by provider name (thread-safe)"""
        if not provider_name or not provider_name.strip():
            raise InvalidInputError("provider_name", "Provider name cannot be empty or whitespace")
        
        provider_name = provider_name.strip()
        
        with self._lock:
            if provider_name not in self.chat_models:
                available_providers = self.list_providers()
                raise ProviderNotFoundError(provider_name, available_providers)
            
            return self.chat_models[provider_name]
    
    def get_embeddings(self, provider_name: str):
        """Get embeddings by provider name (thread-safe)"""
        if not provider_name or not provider_name.strip():
            raise InvalidInputError("provider_name", "Provider name cannot be empty or whitespace")
        
        provider_name = provider_name.strip()
        
        with self._lock:
            if provider_name not in self.embeddings:
                available_providers = self.list_providers()
                raise ProviderNotFoundError(provider_name, available_providers)
            
            if self.embeddings[provider_name] is None:
                raise EmbeddingsNotSupportedError(provider_name)
            
            return self.embeddings[provider_name]
    
    def list_providers(self) -> List[str]:
        """List all available providers (thread-safe)"""
        with self._lock:
            return list(self.providers.keys())
    
    def create_chain(self, provider_name: str, prompt_template: str) -> Runnable:
        """Create a chain with a specific provider"""
        if not prompt_template or not prompt_template.strip():
            raise InvalidInputError("prompt_template", "Prompt template cannot be empty or whitespace")
        
        prompt = ChatPromptTemplate.from_template(prompt_template)
        model = self.get_chat_model(provider_name)  # This will handle provider validation
        
        return prompt | model
    
    # Framework adapter support
    
    def register_framework(self, adapter: 'FrameworkAdapter') -> None:
        """Register a framework adapter (thread-safe).
        
        Args:
            adapter: Framework adapter instance
            
        Raises:
            ValueError: If adapter is invalid or already registered
        """
        if adapter is None:
            raise InvalidInputError("adapter", "Framework adapter cannot be None")
        
        framework_name = adapter.framework_name
        
        with self._lock:
            if framework_name in self.frameworks:
                raise ValueError(
                    f"Framework '{framework_name}' is already registered. "
                    f"Use a different adapter or unregister the existing one first."
                )
            
            self.frameworks[framework_name] = adapter
    
    def get_framework(self, framework_name: str) -> 'FrameworkAdapter':
        """Get a registered framework adapter (thread-safe).
        
        Args:
            framework_name: Name of the framework
            
        Returns:
            Framework adapter instance
            
        Raises:
            ValueError: If framework not found
        """
        if not framework_name or not framework_name.strip():
            raise InvalidInputError("framework_name", "Framework name cannot be empty")
        
        framework_name = framework_name.strip()
        
        with self._lock:
            if framework_name not in self.frameworks:
                available = self.list_frameworks()
                if available:
                    avail_str = "', '".join(available)
                    raise ValueError(
                        f"Framework '{framework_name}' not registered. "
                        f"Available frameworks: ['{avail_str}']"
                    )
                else:
                    raise ValueError(
                        f"Framework '{framework_name}' not registered. "
                        f"No frameworks have been registered yet."
                    )
            
            return self.frameworks[framework_name]
    
    def list_frameworks(self) -> List[str]:
        """List all registered framework adapters (thread-safe).
        
        Returns:
            List[str]: Names of registered frameworks
        """
        with self._lock:
            return list(self.frameworks.keys())
    
    def unregister_framework(self, framework_name: str) -> None:
        """Unregister a framework adapter (thread-safe).
        
        Args:
            framework_name: Name of framework to unregister
            
        Raises:
            ValueError: If framework not found
        """
        with self._lock:
            if framework_name not in self.frameworks:
                raise ValueError(
                    f"Framework '{framework_name}' not registered"
                )
            
            del self.frameworks[framework_name]
