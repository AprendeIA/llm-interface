"""LangChain framework adapter.

This module provides the LangChain adapter that allows using the unified
LLM Interface with LangChain workflows and components.

Example:
    from llm_interface import LLMManager, LLMConfig, ProviderType
    from llm_interface.src.framework.langchain import LangChainAdapter
    
    # Setup manager
    manager = LLMManager()
    config = LLMConfig(
        provider=ProviderType.OPENAI,
        model_name="gpt-4",
        api_key="sk-..."
    )
    manager.add_provider("primary", config)
    
    # Create and register adapter
    langchain = LangChainAdapter(manager)
    manager.register_framework(langchain)
    
    # Use with LangChain components
    model = langchain.get_model("primary")
    chain = langchain.create_chain("primary", "Tell me about {topic}")
    
    # Use with workflows
    result = chain.invoke({"topic": "AI"})
"""

from .adapter import LangChainAdapter
from .workflows import (
    WorkflowBuilder,
    ChainWorkflow,
    ConditionalWorkflow,
    ParallelWorkflow,
)
from .config_loader import LangChainConfigLoader
from .graph import LLMGraph, GraphState, MultiProviderRouter, GraphBuilder

__all__ = [
    "LangChainAdapter",
    "WorkflowBuilder",
    "ChainWorkflow",
    "ConditionalWorkflow",
    "ParallelWorkflow",
    "LangChainConfigLoader",
    "LLMGraph",
    "GraphState",
    "MultiProviderRouter",
    "GraphBuilder",
]
