"""Custom exceptions for LLM Interface library.

This module provides a hierarchy of custom exceptions for better error handling
and more descriptive error messages throughout the library.
"""

from typing import List, Optional


class LLMInterfaceError(Exception):
    """Base exception for all LLM Interface errors.
    
    All custom exceptions in this library inherit from this base class,
    making it easy to catch all library-specific errors.
    """
    pass


class ConfigurationError(LLMInterfaceError):
    """Raised when there's an error in configuration.
    
    This includes invalid configuration parameters, missing required fields,
    or incompatible configuration combinations.
    """
    pass


class ProviderNotFoundError(LLMInterfaceError):
    """Raised when a requested provider is not found in the manager.
    
    Attributes:
        provider_name: The name of the provider that was not found
        available_providers: List of available provider names
    """
    
    def __init__(self, provider_name: str, available_providers: Optional[List[str]] = None):
        self.provider_name = provider_name
        self.available_providers = available_providers or []
        
        if self.available_providers:
            available_str = "', '".join(self.available_providers)
            message = (
                f"Provider '{provider_name}' not found. "
                f"Available providers: ['{available_str}']"
            )
        else:
            message = (
                f"Provider '{provider_name}' not found. "
                f"No providers have been added yet. Use add_provider() first."
            )
        
        super().__init__(message)


class ProviderAlreadyExistsError(LLMInterfaceError):
    """Raised when attempting to add a provider that already exists.
    
    Attributes:
        provider_name: The name of the provider that already exists
    """
    
    def __init__(self, provider_name: str):
        self.provider_name = provider_name
        message = (
            f"Provider '{provider_name}' already exists. "
            f"Use a different name or remove the existing provider first."
        )
        super().__init__(message)


class UnsupportedProviderError(LLMInterfaceError):
    """Raised when attempting to use an unsupported provider type.
    
    Attributes:
        provider_type: The unsupported provider type
        supported_providers: List of supported provider types
    """
    
    def __init__(self, provider_type: str, supported_providers: Optional[List[str]] = None):
        self.provider_type = provider_type
        self.supported_providers = supported_providers or []
        
        if self.supported_providers:
            supported_str = ", ".join(self.supported_providers)
            message = (
                f"Unsupported provider: '{provider_type}'. "
                f"Available providers: [{supported_str}]"
            )
        else:
            message = f"Unsupported provider: '{provider_type}'"
        
        super().__init__(message)


class ProviderValidationError(LLMInterfaceError):
    """Raised when provider configuration validation fails.
    
    This is raised when the provider-specific validation logic determines
    that the configuration is invalid.
    
    Attributes:
        provider_type: The type of provider that failed validation
        reason: Description of why validation failed
    """
    
    def __init__(self, provider_type: str, reason: str):
        self.provider_type = provider_type
        self.reason = reason
        message = f"Invalid configuration for '{provider_type}': {reason}"
        super().__init__(message)


class APIKeyError(LLMInterfaceError):
    """Raised when an API key is missing or invalid.
    
    Attributes:
        provider_type: The provider type requiring the API key
        env_var: The environment variable name for the API key
    """
    
    def __init__(self, provider_type: str, env_var: Optional[str] = None):
        self.provider_type = provider_type
        self.env_var = env_var
        
        if env_var:
            message = (
                f"API key is required for '{provider_type}'. "
                f"Set {env_var} environment variable or provide api_key in config."
            )
        else:
            message = f"API key is required for '{provider_type}'."
        
        super().__init__(message)


class ModelNotFoundError(LLMInterfaceError):
    """Raised when a requested model is not available.
    
    Attributes:
        model_name: The name of the model that was not found
        provider_name: The provider where the model was requested
    """
    
    def __init__(self, model_name: str, provider_name: Optional[str] = None):
        self.model_name = model_name
        self.provider_name = provider_name
        
        if provider_name:
            message = f"Model '{model_name}' not found for provider '{provider_name}'"
        else:
            message = f"Model '{model_name}' not found"
        
        super().__init__(message)


class EmbeddingsNotSupportedError(LLMInterfaceError):
    """Raised when embeddings are requested from a provider that doesn't support them.
    
    Attributes:
        provider_name: The provider that doesn't support embeddings
    """
    
    def __init__(self, provider_name: str):
        self.provider_name = provider_name
        message = f"Provider '{provider_name}' does not support embeddings"
        super().__init__(message)


class InvalidInputError(LLMInterfaceError):
    """Raised when invalid input is provided to a function.
    
    Attributes:
        parameter_name: The name of the invalid parameter
        reason: Description of why the input is invalid
    """
    
    def __init__(self, parameter_name: str, reason: str):
        self.parameter_name = parameter_name
        self.reason = reason
        message = f"Invalid input for '{parameter_name}': {reason}"
        super().__init__(message)


class GraphExecutionError(LLMInterfaceError):
    """Raised when there's an error executing a LangGraph workflow.
    
    Attributes:
        node_name: The name of the node where the error occurred
        original_error: The original exception that was raised
    """
    
    def __init__(self, node_name: Optional[str] = None, original_error: Optional[Exception] = None):
        self.node_name = node_name
        self.original_error = original_error
        
        if node_name and original_error:
            message = f"Error in graph node '{node_name}': {str(original_error)}"
        elif node_name:
            message = f"Error in graph node '{node_name}'"
        elif original_error:
            message = f"Graph execution error: {str(original_error)}"
        else:
            message = "Graph execution error"
        
        super().__init__(message)
