"""
Semantic Kernel framework adapter.

Provides integration between llm_interface providers and Semantic Kernel.
"""

from typing import Dict, Any, List, Optional, Callable, Union
from ...manager import LLMManager
from ..base import FrameworkAdapter
from ..exceptions import FrameworkConfigurationError, FrameworkExecutionError

try:
    from semantic_kernel import Kernel
    from semantic_kernel.functions import KernelFunction, KernelPlugin
    from semantic_kernel.connectors.ai.open_ai import (
        OpenAIChatCompletion,
        AzureChatCompletion,
    )
    from semantic_kernel.prompt_template import PromptTemplateConfig
    SK_AVAILABLE = True
except ImportError:
    SK_AVAILABLE = False
    # Create placeholder types for when semantic_kernel is not installed
    class Kernel:  # type: ignore
        """Placeholder for semantic_kernel.Kernel"""
        pass
    
    class KernelFunction:  # type: ignore
        """Placeholder for semantic_kernel.functions.KernelFunction"""
        pass
    
    class KernelPlugin:  # type: ignore
        """Placeholder for semantic_kernel.functions.KernelPlugin"""
        pass
    
    class OpenAIChatCompletion:  # type: ignore
        """Placeholder for semantic_kernel OpenAIChatCompletion"""
        pass
    
    class AzureChatCompletion:  # type: ignore
        """Placeholder for semantic_kernel AzureChatCompletion"""
        pass
    
    class PromptTemplateConfig:  # type: ignore
        """Placeholder for semantic_kernel PromptTemplateConfig"""
        pass


class SemanticKernelAdapter(FrameworkAdapter):
    """
    Adapter for Microsoft Semantic Kernel framework.
    
    Enables creation of SK kernels using unified provider interface,
    supporting plugins, functions, and AI orchestration.
    
    Example:
        >>> manager = LLMManager()
        >>> manager.add_provider("openai", openai_config)
        >>> adapter = SemanticKernelAdapter(manager)
        >>> kernel = adapter.create_kernel("openai")
        >>> function = adapter.create_semantic_function(
        ...     kernel=kernel,
        ...     prompt="Translate to {{$language}}: {{$text}}",
        ...     function_name="translate"
        ... )
    """
    
    def __init__(self, llm_manager: LLMManager):
        """
        Initialize Semantic Kernel adapter.
        
        Args:
            llm_manager: LLMManager instance with configured providers
            
        Raises:
            FrameworkConfigurationError: If Semantic Kernel is not installed
        """
        if not SK_AVAILABLE:
            raise FrameworkConfigurationError(
                "Semantic Kernel is not installed. Install with: pip install semantic-kernel"
            )
        super().__init__(llm_manager)
        self.kernels: Dict[str, Kernel] = {}
        self.plugins: Dict[str, KernelPlugin] = {}
        self.functions: Dict[str, KernelFunction] = {}
    
    @property
    def framework_name(self) -> str:
        """Return framework identifier."""
        return "semantic_kernel"
    
    @property
    def framework_version(self) -> str:
        """Return minimum supported framework version."""
        return "1.0.0"
    
    def create_kernel(
        self,
        provider_name: str,
        kernel_id: Optional[str] = None,
        **kwargs
    ) -> Kernel:
        """
        Create a Semantic Kernel instance with specified provider.
        
        Args:
            provider_name: Name of provider from LLMManager
            kernel_id: Optional identifier for the kernel
            **kwargs: Additional kernel configuration
            
        Returns:
            Configured Kernel instance
            
        Raises:
            FrameworkConfigurationError: If provider not found or config invalid
            
        Example:
            >>> kernel = adapter.create_kernel("openai", kernel_id="main")
        """
        try:
            # Get provider config
            if provider_name not in self.llm_manager.providers:
                raise FrameworkConfigurationError(
                    f"Provider '{provider_name}' not found in LLMManager"
                )
            
            provider = self.llm_manager.providers[provider_name]
            config = provider.config
            
            # Create kernel
            kernel = Kernel()
            
            # Add AI service based on provider type
            provider_type = config.provider.value.lower() if hasattr(config.provider, 'value') else str(config.provider).lower()
            
            if provider_type == "openai":
                service = OpenAIChatCompletion(
                    ai_model_id=config.model_name or "gpt-4",
                    api_key=config.api_key,
                    service_id=provider_name
                )
                kernel.add_service(service)
                
            elif provider_type == "azure":
                if not config.azure_endpoint or not config.azure_deployment:
                    raise FrameworkConfigurationError(
                        "Azure provider requires azure_endpoint and azure_deployment"
                    )
                
                service = AzureChatCompletion(
                    deployment_name=config.azure_deployment,
                    endpoint=config.azure_endpoint,
                    api_key=config.api_key,
                    api_version=config.azure_api_version or "2024-02-15-preview",
                    service_id=provider_name
                )
                kernel.add_service(service)
                
            else:
                raise FrameworkConfigurationError(
                    f"Provider type '{provider_type}' not supported by Semantic Kernel. "
                    f"Supported: openai, azure"
                )
            
            # Store kernel reference
            kid = kernel_id or provider_name
            self.kernels[kid] = kernel
            
            return kernel
            
        except Exception as e:
            if isinstance(e, FrameworkConfigurationError):
                raise
            raise FrameworkConfigurationError(
                f"Failed to create Semantic Kernel with '{provider_name}': {str(e)}"
            ) from e
    
    def create_semantic_function(
        self,
        kernel: Kernel,
        prompt: str,
        function_name: str,
        plugin_name: Optional[str] = None,
        description: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> KernelFunction:
        """
        Create a semantic function (AI-powered function).
        
        Args:
            kernel: Kernel instance to add function to
            prompt: Prompt template with {{$variables}}
            function_name: Name of the function
            plugin_name: Optional plugin name
            description: Function description
            max_tokens: Maximum tokens for completion
            temperature: Temperature for generation
            **kwargs: Additional function parameters
            
        Returns:
            KernelFunction instance
            
        Example:
            >>> function = adapter.create_semantic_function(
            ...     kernel=kernel,
            ...     prompt="Summarize: {{$text}}",
            ...     function_name="summarize",
            ...     description="Summarizes text"
            ... )
        """
        try:
            # Create prompt template config
            template_config = PromptTemplateConfig(
                template=prompt,
                name=function_name,
                description=description or f"Semantic function: {function_name}",
            )
            
            # Set execution settings if provided
            if max_tokens is not None or temperature is not None:
                execution_settings = {}
                if max_tokens:
                    execution_settings["max_tokens"] = max_tokens
                if temperature is not None:
                    execution_settings["temperature"] = temperature
                template_config.execution_settings = execution_settings
            
            # Create the function
            function = kernel.add_function(
                plugin_name=plugin_name or "custom",
                function_name=function_name,
                prompt_template_config=template_config,
                **kwargs
            )
            
            # Store function reference
            full_name = f"{plugin_name}.{function_name}" if plugin_name else function_name
            self.functions[full_name] = function
            
            return function
            
        except Exception as e:
            raise FrameworkConfigurationError(
                f"Failed to create semantic function '{function_name}': {str(e)}"
            ) from e
    
    def create_native_function(
        self,
        kernel: Kernel,
        function: Callable,
        function_name: str,
        plugin_name: Optional[str] = None,
        description: Optional[str] = None,
        **kwargs
    ) -> KernelFunction:
        """
        Create a native function (Python function).
        
        Args:
            kernel: Kernel instance to add function to
            function: Python callable
            function_name: Name of the function
            plugin_name: Optional plugin name
            description: Function description
            **kwargs: Additional function parameters
            
        Returns:
            KernelFunction instance
            
        Example:
            >>> def get_weather(city: str) -> str:
            ...     return f"Weather in {city}: Sunny"
            >>> 
            >>> func = adapter.create_native_function(
            ...     kernel=kernel,
            ...     function=get_weather,
            ...     function_name="get_weather",
            ...     description="Gets weather for a city"
            ... )
        """
        try:
            # Create kernel function from callable
            kernel_func = kernel.add_function(
                plugin_name=plugin_name or "native",
                function_name=function_name,
                function=function,
                description=description or f"Native function: {function_name}",
                **kwargs
            )
            
            # Store function reference
            full_name = f"{plugin_name}.{function_name}" if plugin_name else function_name
            self.functions[full_name] = kernel_func
            
            return kernel_func
            
        except Exception as e:
            raise FrameworkConfigurationError(
                f"Failed to create native function '{function_name}': {str(e)}"
            ) from e
    
    def create_plugin(
        self,
        kernel: Kernel,
        plugin_name: str,
        functions: Optional[Dict[str, Callable]] = None,
        **kwargs
    ) -> KernelPlugin:
        """
        Create a plugin (collection of functions).
        
        Args:
            kernel: Kernel instance
            plugin_name: Name of the plugin
            functions: Dictionary of function_name -> callable
            **kwargs: Additional plugin parameters
            
        Returns:
            KernelPlugin instance
            
        Example:
            >>> plugin = adapter.create_plugin(
            ...     kernel=kernel,
            ...     plugin_name="weather",
            ...     functions={
            ...         "get_current": get_current_weather,
            ...         "get_forecast": get_forecast
            ...     }
            ... )
        """
        try:
            # Add functions to kernel under plugin
            if functions:
                for func_name, func in functions.items():
                    kernel.add_function(
                        plugin_name=plugin_name,
                        function_name=func_name,
                        function=func,
                        **kwargs
                    )
            
            # Get the plugin from kernel
            plugin = kernel.plugins.get(plugin_name)
            
            if plugin:
                self.plugins[plugin_name] = plugin
            
            return plugin
            
        except Exception as e:
            raise FrameworkConfigurationError(
                f"Failed to create plugin '{plugin_name}': {str(e)}"
            ) from e
    
    async def invoke_function(
        self,
        kernel: Kernel,
        function: Union[KernelFunction, str],
        arguments: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Any:
        """
        Invoke a kernel function.
        
        Args:
            kernel: Kernel instance
            function: KernelFunction or function name
            arguments: Function arguments
            **kwargs: Additional invocation parameters
            
        Returns:
            Function result
            
        Example:
            >>> result = await adapter.invoke_function(
            ...     kernel=kernel,
            ...     function="summarize",
            ...     arguments={"text": "Long text here..."}
            ... )
        """
        try:
            # Get function if string name provided
            if isinstance(function, str):
                if function not in self.functions:
                    raise FrameworkExecutionError(
                        f"Function '{function}' not found"
                    )
                function = self.functions[function]
            
            # Invoke function
            result = await kernel.invoke(
                function=function,
                arguments=arguments or {},
                **kwargs
            )
            
            return result
            
        except Exception as e:
            if isinstance(e, FrameworkExecutionError):
                raise
            raise FrameworkExecutionError(
                f"Failed to invoke function: {str(e)}"
            ) from e
    
    def invoke_function_sync(
        self,
        kernel: Kernel,
        function: Union[KernelFunction, str],
        arguments: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Any:
        """
        Synchronous wrapper for invoke_function.
        
        Args:
            kernel: Kernel instance
            function: KernelFunction or function name
            arguments: Function arguments
            **kwargs: Additional invocation parameters
            
        Returns:
            Function result
        """
        import asyncio
        
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(
            self.invoke_function(kernel, function, arguments, **kwargs)
        )
    
    def create_model(self, provider_name: str, **kwargs):
        """
        Create a kernel instance (SK uses kernels, not standalone models).
        
        Args:
            provider_name: Provider name
            **kwargs: Additional parameters
            
        Returns:
            Kernel instance
        """
        return self.create_kernel(provider_name, **kwargs)
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate Semantic Kernel-specific configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            True if valid
            
        Raises:
            FrameworkConfigurationError: If configuration is invalid
        """
        if not isinstance(config, dict):
            raise FrameworkConfigurationError("Config must be a dictionary")
        
        # Check for kernel configuration
        if "kernels" in config:
            kernels_config = config["kernels"]
            if not isinstance(kernels_config, (list, dict)):
                raise FrameworkConfigurationError(
                    "kernels config must be list or dict"
                )
        
        # Check for plugins configuration
        if "plugins" in config:
            plugins_config = config["plugins"]
            if not isinstance(plugins_config, (list, dict)):
                raise FrameworkConfigurationError(
                    "plugins config must be list or dict"
                )
        
        return True
    
    def get_kernel(self, kernel_id: str) -> Kernel:
        """
        Get a previously created kernel by ID.
        
        Args:
            kernel_id: Kernel identifier
            
        Returns:
            Kernel instance
            
        Raises:
            KeyError: If kernel not found
        """
        if kernel_id not in self.kernels:
            raise KeyError(f"Kernel '{kernel_id}' not found")
        return self.kernels[kernel_id]
    
    def list_kernels(self) -> List[str]:
        """
        List all created kernel IDs.
        
        Returns:
            List of kernel IDs
        """
        return list(self.kernels.keys())
    
    def get_plugin(self, plugin_name: str) -> KernelPlugin:
        """
        Get a previously created plugin by name.
        
        Args:
            plugin_name: Plugin name
            
        Returns:
            KernelPlugin instance
            
        Raises:
            KeyError: If plugin not found
        """
        if plugin_name not in self.plugins:
            raise KeyError(f"Plugin '{plugin_name}' not found")
        return self.plugins[plugin_name]
    
    def list_plugins(self) -> List[str]:
        """
        List all created plugin names.
        
        Returns:
            List of plugin names
        """
        return list(self.plugins.keys())
    
    def get_function(self, function_name: str) -> KernelFunction:
        """
        Get a previously created function by name.
        
        Args:
            function_name: Function name (can include plugin: "plugin.function")
            
        Returns:
            KernelFunction instance
            
        Raises:
            KeyError: If function not found
        """
        if function_name not in self.functions:
            raise KeyError(f"Function '{function_name}' not found")
        return self.functions[function_name]
    
    def list_functions(self) -> List[str]:
        """
        List all created function names.
        
        Returns:
            List of function names
        """
        return list(self.functions.keys())
