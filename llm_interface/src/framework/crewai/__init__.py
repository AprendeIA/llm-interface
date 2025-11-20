"""CrewAI Framework Adapter Module.

Integrates CrewAI multi-agent workflows with the unified LLM Interface.
"""

from .adapter import CrewAIAdapter
from .config_loader import (
    AgentConfig,
    TaskConfig,
    CrewConfig,
    CrewAIConfigBuilder,
    CrewAIConfigLoader,
)

__all__ = [
    "CrewAIAdapter",
    "AgentConfig",
    "TaskConfig",
    "CrewConfig",
    "CrewAIConfigBuilder",
    "CrewAIConfigLoader",
]
