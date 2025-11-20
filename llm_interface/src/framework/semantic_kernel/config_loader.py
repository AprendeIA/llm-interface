"""
Configuration loader for Semantic Kernel framework.

Handles loading and validation of Semantic Kernel-specific configurations.
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
import yaml
from ...config_loader import ConfigLoader
from ..exceptions import FrameworkConfigurationError


class SemanticKernelConfigLoader:
    """
    Load and validate Semantic Kernel-specific configurations.
    
    Supports loading kernel configurations, plugin definitions,
    and function parameters from YAML or dictionaries.
    
    Example:
        >>> loader = SemanticKernelConfigLoader()
        >>> config = loader.from_yaml("sk_config.yaml")
        >>> kernels_config = config["kernels"]
    """
    
    def __init__(self, base_config_loader: Optional[ConfigLoader] = None):
        """
        Initialize Semantic Kernel config loader.
        
        Args:
            base_config_loader: Optional base ConfigLoader for provider configs
        """
        self.base_loader = base_config_loader or ConfigLoader()
    
    @staticmethod
    def from_yaml(file_path: str) -> Dict[str, Any]:
        """
        Load Semantic Kernel configuration from YAML file.
        
        Args:
            file_path: Path to YAML configuration file
            
        Returns:
            Configuration dictionary
            
        Raises:
            FrameworkConfigurationError: If file not found or invalid YAML
            
        Example:
            >>> config = SemanticKernelConfigLoader.from_yaml("config.yaml")
        """
        try:
            path = Path(file_path)
            if not path.exists():
                raise FrameworkConfigurationError(
                    f"Configuration file not found: {file_path}"
                )
            
            with open(path, 'r') as f:
                config = yaml.safe_load(f)
            
            if not isinstance(config, dict):
                raise FrameworkConfigurationError(
                    "Configuration must be a dictionary"
                )
            
            return config
            
        except yaml.YAMLError as e:
            raise FrameworkConfigurationError(
                f"Invalid YAML in {file_path}: {str(e)}"
            ) from e
        except Exception as e:
            if isinstance(e, FrameworkConfigurationError):
                raise
            raise FrameworkConfigurationError(
                f"Failed to load configuration: {str(e)}"
            ) from e
    
    @staticmethod
    def from_dict(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and return Semantic Kernel configuration from dictionary.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Validated configuration
            
        Raises:
            FrameworkConfigurationError: If configuration is invalid
        """
        if not isinstance(config, dict):
            raise FrameworkConfigurationError(
                "Configuration must be a dictionary"
            )
        
        return config
    
    @staticmethod
    def validate_kernel_config(kernel_config: Dict[str, Any]) -> bool:
        """
        Validate individual kernel configuration.
        
        Args:
            kernel_config: Kernel configuration dictionary
            
        Returns:
            True if valid
            
        Raises:
            FrameworkConfigurationError: If configuration is invalid
        """
        required_fields = ["provider_name"]
        
        for field in required_fields:
            if field not in kernel_config:
                raise FrameworkConfigurationError(
                    f"Kernel config missing required field: {field}"
                )
        
        return True
    
    @staticmethod
    def validate_function_config(function_config: Dict[str, Any]) -> bool:
        """
        Validate function configuration.
        
        Args:
            function_config: Function configuration dictionary
            
        Returns:
            True if valid
            
        Raises:
            FrameworkConfigurationError: If configuration is invalid
        """
        required_fields = ["name"]
        
        for field in required_fields:
            if field not in function_config:
                raise FrameworkConfigurationError(
                    f"Function config missing required field: {field}"
                )
        
        # Validate function type
        if "type" in function_config:
            valid_types = ["semantic", "native"]
            if function_config["type"] not in valid_types:
                raise FrameworkConfigurationError(
                    f"Invalid function type. Must be one of: {valid_types}"
                )
        
        # Semantic functions must have prompt
        if function_config.get("type") == "semantic":
            if "prompt" not in function_config:
                raise FrameworkConfigurationError(
                    "Semantic function must have 'prompt' field"
                )
        
        return True
    
    @staticmethod
    def validate_plugin_config(plugin_config: Dict[str, Any]) -> bool:
        """
        Validate plugin configuration.
        
        Args:
            plugin_config: Plugin configuration dictionary
            
        Returns:
            True if valid
            
        Raises:
            FrameworkConfigurationError: If configuration is invalid
        """
        required_fields = ["name"]
        
        for field in required_fields:
            if field not in plugin_config:
                raise FrameworkConfigurationError(
                    f"Plugin config missing required field: {field}"
                )
        
        # Validate functions if present
        if "functions" in plugin_config:
            functions = plugin_config["functions"]
            if not isinstance(functions, (list, dict)):
                raise FrameworkConfigurationError(
                    "Plugin 'functions' must be a list or dict"
                )
        
        return True
    
    def load_full_config(self, file_path: str) -> Dict[str, Any]:
        """
        Load complete configuration including providers and SK settings.
        
        Args:
            file_path: Path to configuration file
            
        Returns:
            Complete configuration dictionary with:
                - providers: Provider configurations
                - semantic_kernel: SK-specific settings
                
        Example:
            >>> loader = SemanticKernelConfigLoader()
            >>> config = loader.load_full_config("full_config.yaml")
            >>> provider_configs = config["providers"]
            >>> sk_config = config["semantic_kernel"]
        """
        config = self.from_yaml(file_path)
        
        # Validate structure
        if "providers" not in config and "semantic_kernel" not in config:
            raise FrameworkConfigurationError(
                "Configuration must contain 'providers' or 'semantic_kernel' section"
            )
        
        # Validate kernels if present
        if "semantic_kernel" in config and "kernels" in config["semantic_kernel"]:
            kernels = config["semantic_kernel"]["kernels"]
            if isinstance(kernels, list):
                for kernel_config in kernels:
                    self.validate_kernel_config(kernel_config)
            elif isinstance(kernels, dict):
                for kernel_id, kernel_config in kernels.items():
                    if "id" not in kernel_config:
                        kernel_config["id"] = kernel_id
                    self.validate_kernel_config(kernel_config)
        
        # Validate plugins if present
        if "semantic_kernel" in config and "plugins" in config["semantic_kernel"]:
            plugins = config["semantic_kernel"]["plugins"]
            if isinstance(plugins, list):
                for plugin_config in plugins:
                    self.validate_plugin_config(plugin_config)
            elif isinstance(plugins, dict):
                for plugin_name, plugin_config in plugins.items():
                    if "name" not in plugin_config:
                        plugin_config["name"] = plugin_name
                    self.validate_plugin_config(plugin_config)
        
        # Validate functions if present
        if "semantic_kernel" in config and "functions" in config["semantic_kernel"]:
            functions = config["semantic_kernel"]["functions"]
            if isinstance(functions, list):
                for func_config in functions:
                    self.validate_function_config(func_config)
            elif isinstance(functions, dict):
                for func_name, func_config in functions.items():
                    if "name" not in func_config:
                        func_config["name"] = func_name
                    self.validate_function_config(func_config)
        
        return config
    
    @staticmethod
    def create_kernel_config(
        provider_name: str,
        kernel_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create kernel configuration dictionary.
        
        Args:
            provider_name: Provider to use
            kernel_id: Optional kernel identifier
            **kwargs: Additional parameters
            
        Returns:
            Kernel configuration dictionary
        """
        config = {
            "provider_name": provider_name,
        }
        
        if kernel_id:
            config["id"] = kernel_id
        
        config.update(kwargs)
        return config
    
    @staticmethod
    def create_semantic_function_config(
        name: str,
        prompt: str,
        plugin_name: Optional[str] = None,
        description: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create semantic function configuration dictionary.
        
        Args:
            name: Function name
            prompt: Prompt template
            plugin_name: Optional plugin name
            description: Function description
            max_tokens: Maximum tokens
            temperature: Temperature
            **kwargs: Additional parameters
            
        Returns:
            Function configuration dictionary
        """
        config = {
            "name": name,
            "type": "semantic",
            "prompt": prompt,
        }
        
        if plugin_name:
            config["plugin_name"] = plugin_name
        
        if description:
            config["description"] = description
        
        if max_tokens is not None:
            config["max_tokens"] = max_tokens
        
        if temperature is not None:
            config["temperature"] = temperature
        
        config.update(kwargs)
        return config
    
    @staticmethod
    def create_native_function_config(
        name: str,
        plugin_name: Optional[str] = None,
        description: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create native function configuration dictionary.
        
        Args:
            name: Function name
            plugin_name: Optional plugin name
            description: Function description
            **kwargs: Additional parameters
            
        Returns:
            Function configuration dictionary
        """
        config = {
            "name": name,
            "type": "native",
        }
        
        if plugin_name:
            config["plugin_name"] = plugin_name
        
        if description:
            config["description"] = description
        
        config.update(kwargs)
        return config
    
    @staticmethod
    def create_plugin_config(
        name: str,
        functions: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create plugin configuration dictionary.
        
        Args:
            name: Plugin name
            functions: List of function names in plugin
            **kwargs: Additional parameters
            
        Returns:
            Plugin configuration dictionary
        """
        config = {
            "name": name,
        }
        
        if functions:
            config["functions"] = functions
        
        config.update(kwargs)
        return config
