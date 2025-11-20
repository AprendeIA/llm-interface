# Core Examples

This directory contains basic examples demonstrating core functionality of the LLM interface library.

## Files

- **example_usage.py** - Basic usage examples showing provider setup and simple interactions
- **logging_example.py** - Examples of logging configuration and usage

## Quick Start

### Basic Usage

```python
from llm_interface import LLMManager, LLMConfig, ProviderType

# Create manager
manager = LLMManager()

# Add a provider
manager.add_provider("openai", LLMConfig(
    provider=ProviderType.OPENAI,
    model_name="gpt-4",
    api_key="your-api-key"
))

# Get and use model
model = manager.get_chat_model("openai")
response = model.invoke("Hello, world!")
print(response.content)
```

### With Logging

```python
import logging
from llm_interface import LLMManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Use the library (all operations will be logged)
manager = LLMManager()
# ...
```

## See Also

- Main documentation: `../../README.md`
- Configuration examples: `../configs/`
- Framework-specific examples: `../autogen/`, `../crewai/`, `../langchain/`, `../semantic_kernel/`
