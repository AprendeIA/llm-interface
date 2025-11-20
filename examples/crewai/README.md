# CrewAI Examples

This directory contains examples demonstrating the CrewAI framework adapter.

## Files

- **crewai_adapter_examples.py** - Comprehensive examples of using CrewAI with the LLM interface
- **crewai_workflow.yaml** - YAML configuration example for CrewAI workflows

## Quick Start

```python
from llm_interface import LLMManager, LLMConfig, ProviderType
from llm_interface.framework.crewai import CrewAIAdapter

# Setup
manager = LLMManager()
manager.add_provider("openai", LLMConfig(
    provider=ProviderType.OPENAI,
    model_name="gpt-4",
    api_key="your-key"
))

# Create CrewAI adapter
adapter = CrewAIAdapter(manager)

# Create agents and crews
agent = adapter.create_agent(
    role="Researcher",
    goal="Find information",
    provider_name="openai"
)
```

## See Also

- Main documentation: `../../README.md`
- Configuration examples: `../configs/`
- Cross-framework examples: `../cross_framework/`
