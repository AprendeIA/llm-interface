# LangChain Examples

This directory contains examples demonstrating the LangChain framework adapter.

## Files

- **langchain_adapter_examples.py** - Comprehensive examples of using LangChain with the LLM interface
- **langchain_workflow.yaml** - YAML configuration example for LangGraph workflows

## Quick Start

```python
from llm_interface import LLMManager, LLMConfig, ProviderType
from llm_interface.framework.langchain import LangChainAdapter

# Setup
manager = LLMManager()
manager.add_provider("openai", LLMConfig(
    provider=ProviderType.OPENAI,
    model_name="gpt-4",
    api_key="your-key"
))

# Create LangChain adapter
adapter = LangChainAdapter(manager)

# Use with LangChain
model = adapter.get_model("openai")
response = model.invoke("Hello!")
```

## See Also

- Main documentation: `../../README.md`
- Configuration examples: `../configs/`
- Graph examples: See langchain_adapter_examples.py for LangGraph usage
