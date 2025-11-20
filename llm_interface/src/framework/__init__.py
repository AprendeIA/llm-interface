"""Framework adapters for multi-framework support.

This module provides adapters that allow the LLM Interface to work with
multiple AI frameworks (LangChain, CrewAI, AutoGen, Semantic Kernel) while
maintaining a unified provider interface.

Available Adapters:
- LangChain: Graph-based workflow orchestration
- CrewAI: Multi-agent teams with role-based agents
- AutoGen: Agent conversation framework
- Semantic Kernel: Skill-based plugin system

Example:
    from llm_interface import LLMManager, LLMConfig
    from llm_interface.framework.crewai import CrewAIAdapter
    
    manager = LLMManager()
    # ... add providers ...
    
    # Register framework adapter
    crewai = CrewAIAdapter(manager)
    manager.register_framework(crewai)
"""

from .base import FrameworkAdapter
from .interfaces import (
    FrameworkModel,
    FrameworkStreamableModel,
    FrameworkAsyncModel,
    FrameworkWorkflow,
    FrameworkAgent,
    FrameworkTask,
    FrameworkConversation,
    FrameworkOrchestrator,
)
from .exceptions import (
    FrameworkError,
    FrameworkNotAvailableError,
    FrameworkConfigurationError,
    FrameworkExecutionError,
    FrameworkModelCreationError,
    FrameworkAgentExecutionError,
    FrameworkProviderMismatchError,
)

__all__ = [
    "FrameworkAdapter",
    "FrameworkModel",
    "FrameworkStreamableModel",
    "FrameworkAsyncModel",
    "FrameworkWorkflow",
    "FrameworkAgent",
    "FrameworkTask",
    "FrameworkConversation",
    "FrameworkOrchestrator",
    "FrameworkError",
    "FrameworkNotAvailableError",
    "FrameworkConfigurationError",
    "FrameworkExecutionError",
    "FrameworkModelCreationError",
    "FrameworkAgentExecutionError",
    "FrameworkProviderMismatchError",
]
