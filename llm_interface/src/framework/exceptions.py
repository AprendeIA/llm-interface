"""Framework-specific exceptions.

This module provides custom exceptions for framework adapter operations
and framework-specific errors.
"""

from typing import Optional, List
from ..core.exceptions import LLMInterfaceError


class FrameworkError(LLMInterfaceError):
    """Base exception for framework adapter errors.
    
    All framework-specific exceptions inherit from this class.
    """
    pass


class FrameworkNotAvailableError(FrameworkError):
    """Raised when a requested framework is not installed.
    
    Attributes:
        framework_name: Name of the framework that's not available
        installed_version: Installed version if available
    """
    
    def __init__(self, framework_name: str, installed_version: Optional[str] = None):
        self.framework_name = framework_name
        self.installed_version = installed_version
        
        if installed_version:
            message = (
                f"Framework '{framework_name}' is not compatible. "
                f"Installed version: {installed_version}. "
                f"Please update or install the required version."
            )
        else:
            message = (
                f"Framework '{framework_name}' is not installed. "
                f"Install with: pip install llm-interface[{framework_name}]"
            )
        
        super().__init__(message)


class FrameworkConfigurationError(FrameworkError):
    """Raised when framework configuration is invalid.
    
    Attributes:
        framework_name: Name of the framework
        reason: Description of the configuration error
    """
    
    def __init__(self, reason: str, framework_name: Optional[str] = None):
        self.framework_name = framework_name
        self.reason = reason
        
        if framework_name:
            message = f"Invalid configuration for '{framework_name}': {reason}"
        else:
            message = f"Invalid framework configuration: {reason}"
        
        super().__init__(message)


class FrameworkExecutionError(FrameworkError):
    """Raised when framework execution fails.
    
    Attributes:
        framework_name: Name of the framework
        operation: Description of the operation that failed
        original_error: The underlying exception
    """
    
    def __init__(self, 
                 operation: str,
                 framework_name: Optional[str] = None,
                 original_error: Optional[Exception] = None):
        self.framework_name = framework_name
        self.operation = operation
        self.original_error = original_error
        
        if framework_name and original_error:
            message = (
                f"Execution error in '{framework_name}' during {operation}: "
                f"{str(original_error)}"
            )
        elif framework_name:
            message = f"Execution error in '{framework_name}' during {operation}"
        elif original_error:
            message = f"Framework execution error in {operation}: {str(original_error)}"
        else:
            message = f"Framework execution error in {operation}"
        
        super().__init__(message)


class FrameworkModelCreationError(FrameworkError):
    """Raised when model creation fails.
    
    Attributes:
        provider_name: Name of the provider
        framework_name: Name of the framework
        reason: Description of the error
    """
    
    def __init__(self, 
                 provider_name: str,
                 framework_name: str,
                 reason: str):
        self.provider_name = provider_name
        self.framework_name = framework_name
        self.reason = reason
        
        message = (
            f"Failed to create model in '{framework_name}' with provider "
            f"'{provider_name}': {reason}"
        )
        super().__init__(message)


class FrameworkAgentExecutionError(FrameworkError):
    """Raised when agent execution fails.
    
    Attributes:
        agent_name: Name of the agent
        task_description: Description of the task
        original_error: The underlying exception
    """
    
    def __init__(self,
                 agent_name: str,
                 task_description: str,
                 original_error: Optional[Exception] = None):
        self.agent_name = agent_name
        self.task_description = task_description
        self.original_error = original_error
        
        if original_error:
            message = (
                f"Agent '{agent_name}' failed on task '{task_description}': "
                f"{str(original_error)}"
            )
        else:
            message = (
                f"Agent '{agent_name}' failed on task '{task_description}'"
            )
        
        super().__init__(message)


class FrameworkProviderMismatchError(FrameworkError):
    """Raised when provider is incompatible with framework.
    
    Attributes:
        provider_type: Type of provider
        framework_name: Name of the framework
        supported_providers: List of supported provider types
    """
    
    def __init__(self,
                 provider_type: str,
                 framework_name: str,
                 supported_providers: Optional[List[str]] = None):
        self.provider_type = provider_type
        self.framework_name = framework_name
        self.supported_providers = supported_providers or []
        
        if self.supported_providers:
            supported_str = ", ".join(self.supported_providers)
            message = (
                f"Provider type '{provider_type}' is not compatible with "
                f"'{framework_name}'. Supported providers: [{supported_str}]"
            )
        else:
            message = (
                f"Provider type '{provider_type}' is not compatible with "
                f"'{framework_name}'"
            )
        
        super().__init__(message)
