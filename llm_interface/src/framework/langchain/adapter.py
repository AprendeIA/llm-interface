"""LangChain adapter implementation.

Provides FrameworkAdapter for LangChain integration with the unified LLM Interface.
Wraps existing LangGraph workflows and adds LangChain-specific utilities.
"""

from typing import Any, Optional, Dict, List, Tuple
from langchain_core.language_models import BaseLanguageModel
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable

from llm_interface.src.framework.base import FrameworkAdapter
from llm_interface.src.framework.interfaces import FrameworkModel, FrameworkWorkflow
from llm_interface.src.framework.exceptions import (
    FrameworkModelCreationError,
    FrameworkConfigurationError,
    FrameworkNotAvailableError,
)
from llm_interface.src.manager import LLMManager


class LangChainAdapter(FrameworkAdapter):
    """LangChain framework adapter.
    
    Provides LangChain-specific functionality built on top of the unified LLM Interface.
    Supports creating LangChain models, chains, and workflows from configured providers.
    
    Attributes:
        manager: LLMManager instance for provider access
        _chains: Cache of created chains
        _models: Cache of created models
    
    Example:
        >>> manager = LLMManager()
        >>> adapter = LangChainAdapter(manager)
        >>> model = adapter.get_model("default")
        >>> chain = adapter.create_chain("default", "Summarize: {text}")
        >>> result = chain.invoke({"text": "Long document..."})
    """
    
    def __init__(self, llm_manager: LLMManager, config: Dict[str, Any] = None):
        """Initialize LangChain adapter.
        
        Args:
            llm_manager: LLMManager instance for accessing providers
            config: Optional framework-specific configuration
            
        Raises:
            FrameworkConfigurationError: If manager is not configured or has no providers
        """
        super().__init__(llm_manager=llm_manager, config=config)
        self._chains: Dict[str, Runnable] = {}
        self._models: Dict[str, BaseLanguageModel] = {}
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate LangChain adapter configuration.
        
        Checks that:
        - Manager is configured
        - At least one provider is available
        - Available providers have required LangChain models
        
        Args:
            config: Configuration dictionary to validate
        
        Returns:
            bool: True if configuration is valid
        """
        if not self.llm_manager:
            return False
        
        providers = self.list_providers()
        if not providers:
            return False
        
        for provider_name in providers:
            try:
                if not self.has_provider(provider_name):
                    return False
            except Exception:
                return False
        
        return True
    
    def create_model(
        self,
        provider_name: str,
        **kwargs: Any
    ) -> BaseLanguageModel:
        """Create a LangChain language model from a provider.
        
        Args:
            provider_name: Name of the configured provider
            **kwargs: Additional arguments passed to model creation
            
        Returns:
            BaseLanguageModel instance from LangChain
            
        Raises:
            FrameworkNotAvailableError: If provider not available
            FrameworkModelCreationError: If model creation fails
            
        Example:
            >>> model = adapter.create_model("openai")
            >>> response = model.invoke("What is Python?")
        """
        if not self.has_provider(provider_name):
            raise FrameworkNotAvailableError(
                f"Provider '{provider_name}' not available",
                framework="LangChain"
            )
        
        try:
            model = self.get_chat_model(provider_name, **kwargs)
            if not model:
                raise FrameworkModelCreationError(
                    f"Failed to create model from provider '{provider_name}'",
                    framework="LangChain"
                )
            self._models[provider_name] = model
            return model
        except Exception as e:
            raise FrameworkModelCreationError(
                f"Error creating LangChain model from '{provider_name}': {str(e)}",
                framework="LangChain"
            ) from e
    
    def get_model(
        self,
        provider_name: str,
        use_cache: bool = True,
        **kwargs: Any
    ) -> BaseLanguageModel:
        """Get or create a LangChain model.
        
        Args:
            provider_name: Name of the provider
            use_cache: Whether to use cached model if available
            **kwargs: Additional arguments for model creation
            
        Returns:
            BaseLanguageModel instance
            
        Raises:
            FrameworkNotAvailableError: If provider not available
            FrameworkModelCreationError: If model creation fails
        """
        if use_cache and provider_name in self._models:
            return self._models[provider_name]
        
        return self.create_model(provider_name, **kwargs)
    
    def create_chain(
        self,
        provider_name: str,
        prompt_template: str,
        output_key: str = "text",
        **kwargs: Any
    ) -> Runnable:
        """Create a LangChain chain with a prompt template.
        
        Args:
            provider_name: Name of the provider
            prompt_template: String template for the prompt
            output_key: Key for output variable in chain (deprecated, kept for compatibility)
            **kwargs: Additional arguments
            
        Returns:
            Runnable chain instance
            
        Raises:
            FrameworkNotAvailableError: If provider not available
            FrameworkModelCreationError: If chain creation fails
            
        Example:
            >>> chain = adapter.create_chain(
            ...     "openai",
            ...     "Summarize this text: {text}"
            ... )
            >>> result = chain.invoke({"text": "Long document..."})
        """
        chain_key = f"{provider_name}_{prompt_template[:20]}"
        
        if chain_key in self._chains:
            return self._chains[chain_key]
        
        try:
            model = self.get_model(provider_name, **kwargs)
            prompt = PromptTemplate(
                input_variables=self._extract_variables(prompt_template),
                template=prompt_template
            )
            # Modern LangChain uses the pipe operator to create chains
            chain = prompt | model
            self._chains[chain_key] = chain
            return chain
        except Exception as e:
            raise FrameworkModelCreationError(
                f"Error creating LangChain chain: {str(e)}",
                framework="LangChain"
            ) from e
    
    def create_graph(
        self,
        graph_name: str,
        provider_name: str,
        **kwargs: Any
    ) -> Any:
        """Create a LangGraph workflow using LLM provider.
        
        Args:
            graph_name: Name for the graph/workflow
            provider_name: Name of the provider
            **kwargs: Additional configuration
            
        Returns:
            LangGraph workflow
            
        Raises:
            FrameworkNotAvailableError: If provider not available
            FrameworkModelCreationError: If graph creation fails
        """
        if not self.has_provider(provider_name):
            raise FrameworkNotAvailableError(
                f"Provider '{provider_name}' not available for graph",
                framework="LangChain"
            )
        
        try:
            model = self.get_model(provider_name)
            
            # Import here to avoid hard dependency
            from .graph import LLMGraph
            
            graph = LLMGraph(
                name=graph_name,
                model=model,
                **kwargs
            )
            return graph
        except ImportError:
            raise FrameworkModelCreationError(
                "LangGraph not available. Install with: pip install langgraph",
                framework="LangChain"
            )
        except Exception as e:
            raise FrameworkModelCreationError(
                f"Error creating LangGraph workflow: {str(e)}",
                framework="LangChain"
            ) from e
    
    def create_embeddings(
        self,
        provider_name: str,
        **kwargs: Any
    ) -> Optional[Embeddings]:
        """Create a LangChain embeddings model.
        
        Args:
            provider_name: Name of the provider
            **kwargs: Additional arguments
            
        Returns:
            Embeddings instance or None if not available
            
        Raises:
            FrameworkNotAvailableError: If provider not available
            FrameworkModelCreationError: If creation fails
        """
        if not self.has_provider(provider_name):
            raise FrameworkNotAvailableError(
                f"Provider '{provider_name}' not available",
                framework="LangChain"
            )
        
        try:
            embeddings = self.get_embeddings(provider_name, **kwargs)
            return embeddings
        except Exception as e:
            raise FrameworkModelCreationError(
                f"Error creating embeddings: {str(e)}",
                framework="LangChain"
            ) from e
    
    def get_available_models(self) -> Dict[str, List[str]]:
        """Get available models for each provider.
        
        Returns:
            Dictionary mapping provider names to available model names
        """
        models: Dict[str, List[str]] = {}
        for provider_name in self.list_providers():
            info = self.get_provider_info(provider_name)
            if info:
                models[provider_name] = info.get("available_models", [])
        return models
    
    def clear_cache(self) -> None:
        """Clear all cached models and chains."""
        self._models.clear()
        self._chains.clear()
    
    @staticmethod
    def _extract_variables(template: str) -> List[str]:
        """Extract variable names from prompt template.
        
        Args:
            template: Prompt template string
            
        Returns:
            List of variable names found in template
        """
        import re
        # Find all {variable} patterns
        pattern = r'\{(\w+)\}'
        return re.findall(pattern, template)
    
    @property
    def framework_name(self) -> str:
        """Return framework name."""
        return "LangChain"
    
    @property
    def framework_version(self) -> str:
        """Return LangChain version."""
        try:
            import langchain
            return getattr(langchain, "__version__", "unknown")
        except (ImportError, AttributeError):
            return "unknown"
