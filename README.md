# LLM Interface

A flexible, production-ready Python library for managing multiple LLM providers with a unified interface. Easily switch between OpenAI, Azure OpenAI, Anthropic, and Ollama. Features advanced LangGraph integration, cross-framework support (LangChain, CrewAI, AutoGen, Semantic Kernel), and comprehensive tool utilities.

## Features

- **Multi-Provider Support**: OpenAI, Azure OpenAI, Anthropic, Ollama with extensible architecture
- **Unified Interface**: Consistent API across all providers with thread-safe operations
- **Multiple Framework Support**: LangChain, CrewAI, AutoGen, Semantic Kernel adapters
- **Advanced LangGraph Integration**: Build complex multi-provider workflows with state management
- **Framework Utilities**: Comparison tools, dynamic switching, benchmarking
- **Flexible Configuration**: YAML, environment variables, or programmatic setup
- **Factory Pattern**: Type-safe, extensible provider instantiation
- **Pydantic Validation**: Comprehensive configuration validation with runtime checks
- **Type Safe**: Full type annotations throughout the codebase
- **Error Handling**: Robust validation, custom exceptions, and detailed error messages
- **Thread Safety**: Reentrant locks for concurrent provider operations
- **Production Ready**: Tested, documented, and suitable for enterprise use

## Installation

### Option 1: Virtual Environment (Recommended)

```bash
# Clone or download the repository
cd llm_interface

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows (PowerShell):.ª
.\venv\Scripts\Activate.ps1
# On Windows (Command Prompt):
venv\Scripts\activate.bat
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install the package in editable mode
pip install -e .
```

### Option 2: Direct Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Dependencies

Core packages (automatically installed):
```
langgraph>=0.1.0
pydantic>=2.0.0
langchain-community>=0.0.1
langchain-ollama>=0.0.1
langchain-core>=0.1.0
langchain-openai>=0.1.0
langchain-anthropic>=0.1.0
PyYAML>=6.0
setuptools>=40.0.0
```

Optional framework packages:
```
crewai>=0.1.0           # For CrewAI integration
pyautogen>=0.2.0        # For AutoGen integration
semantic-kernel>=0.4.0  # For Semantic Kernel integration
```

## Quick Start

### Basic Usage

```python
from llm_interface import LLMManager, LLMConfig, ProviderType

# Create manager
manager = LLMManager()

# Add OpenAI provider
openai_config = LLMConfig(
    provider=ProviderType.OPENAI,
    model_name="gpt-4",
    api_key="your-api-key",  # Or use OPENAI_API_KEY environment variable
    temperature=0.7
)
manager.add_provider("openai", openai_config)

# Use the model
model = manager.get_chat_model("openai")
response = model.invoke("Hello, how are you?")
print(response.content)
```

### Configuration File

Create a `config.yaml`:

```yaml
providers:
  openai:
    provider: "openai"
    model_name: "gpt-4"
    api_key: "${OPENAI_API_KEY}"  # References environment variable
    temperature: 0.7
    max_tokens: 1000
  
  azure:
    provider: "azure"
    model_name: "gpt-4"
    api_key: "${AZURE_OPENAI_API_KEY}"
    azure_endpoint: "${AZURE_OPENAI_ENDPOINT}"
    azure_deployment: "gpt-4-deployment"
    azure_api_version: "2023-12-01-preview"
    temperature: 0.7
  
  ollama:
    provider: "ollama"
    model_name: "llama2"
    base_url: "http://localhost:11434"
    temperature: 0.7
  
  anthropic:
    provider: "anthropic"
    model_name: "claude-3-sonnet-20240229"
    api_key: "${ANTHROPIC_API_KEY}"
    temperature: 0.7
```

Load configuration:

```python
from llm_interface import ConfigLoader, LLMManager

# Load from YAML
configs = ConfigLoader.from_yaml("config.yaml")

# Create manager and add all providers
manager = LLMManager()
for name, config in configs.items():
    manager.add_provider(name, config)

# Use any provider
response = manager.get_chat_model("openai").invoke("Hello!")
```

### Environment Variables

Set environment variables:

```bash
# OpenAI
export OPENAI_API_KEY="your-openai-key"

# Azure OpenAI
export AZURE_OPENAI_API_KEY="your-azure-key"
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com"

# Anthropic
export ANTHROPIC_API_KEY="your-anthropic-key"
```

Load from environment:

```python
from llm_interface import ConfigLoader, LLMManager

# Load from environment variables (with custom prefix)
configs = ConfigLoader.from_env("LLM_")

manager = LLMManager()
for name, config in configs.items():
    manager.add_provider(name, config)
```

## Testing Your Installation

After installation, you can test the library with this simple script:

```python
# test_installation.py
from llm_interface import LLMManager, LLMConfig, ProviderType

def test_installation():
    print("Testing LLM Interface Library...")
    
    # Test basic imports
    manager = LLMManager()
    print("LLMManager created successfully")
    
    # Test provider factory
    from llm_interface import LLMProviderFactory
    factory = LLMProviderFactory()
    supported = factory.list_supported_providers()
    print(f"Supported providers: {[p.value for p in supported]}")
    
    # Test adding a provider (no real API key needed for this test)
    test_config = LLMConfig(
        provider=ProviderType.OPENAI,
        model_name="gpt-3.5-turbo",
        api_key="test-key"
    )
    manager.add_provider("test", test_config)
    print(f"Provider added. Current providers: {manager.list_providers()}")
    
    print("Installation test completed successfully!")

if __name__ == "__main__":
    test_installation()
```

Run the test:
```bash
python test_installation.py
```

## Framework Integration

### LangGraph Workflows

Build complex workflows with LangGraph:

```python
from llm_interface import LLMManager, LLMGraph
from langchain_core.messages import HumanMessage

# Setup manager with providers
manager = LLMManager()
# ... add providers ...

# Create LangGraph wrapper
graph = LLMGraph(manager)

# Simple chat with routing
messages = [HumanMessage(content="Explain quantum computing")]
result = graph.run_simple_chat(messages, "openai")
print(result["response"])

# Custom multi-provider workflow
workflow_definition = {
    "nodes": {
        "analyze": {"provider": "openai"},
        "summarize": {"provider": "anthropic"},
        "enhance": {"provider": "azure"}
    },
    "edges": {
        "analyze": "summarize",
        "summarize": "enhance"
    },
    "entry_point": "analyze"
}

custom_graph = graph.build_custom_graph(workflow_definition)
compiled_graph = graph.compile_graph(custom_graph)
```

### LangChain Integration

Seamless LangChain adapter:

```python
from llm_interface.framework.langchain import LangChainAdapter

adapter = LangChainAdapter(manager)

# Use LangChain chains with any provider
chain = adapter.create_chain(
    "openai",
    "You are a helpful assistant.\nUser: {input}"
)

response = chain.invoke({"input": "What is machine learning?"})
```

### CrewAI Integration

Run CrewAI agents with LLM Interface providers:

```python
from llm_interface.framework.crewai import CrewAIAdapter

adapter = CrewAIAdapter(manager)

# Create CrewAI agents using LLM Interface providers
agent = adapter.create_agent(
    provider="openai",
    role="Data Scientist",
    goal="Analyze trends",
    backstory="Expert in data analysis"
)
```

### AutoGen Integration

Use AutoGen with LLM Interface:

```python
from llm_interface.framework.autogen import AutoGenAdapter

adapter = AutoGenAdapter(manager)

# Create AutoGen conversations with any provider
config = adapter.create_llm_config("openai")
```

### Semantic Kernel Integration

Integrate with Semantic Kernel:

```python
from llm_interface.framework.semantic_kernel import SemanticKernelAdapter

adapter = SemanticKernelAdapter(manager)

# Create Semantic Kernel with LLM Interface providers
kernel = adapter.create_kernel("openai")
```

## Cross-Framework Tools

### Framework Comparison

Compare the same task across different frameworks:

```python
from llm_interface.cross_framework import FrameworkComparison

comparison = FrameworkComparison(manager)

# Compare frameworks on a specific task
task = "Generate a Python function for fibonacci"
report = comparison.compare_frameworks(
    task=task,
    frameworks=["langchain", "crewai", "autogen"],
    providers=["openai", "anthropic"]
)

print(f"Fastest: {report.fastest_framework}")
print(f"Most reliable: {report.most_reliable_framework}")
```

### Framework Switching

Dynamically switch between frameworks:

```python
from llm_interface.cross_framework import FrameworkSwitcher, SwitchStrategy

switcher = FrameworkSwitcher(manager)

# Set up fallback chain
switcher.set_fallback_chain(
    ["langchain", "crewai", "autogen"],
    strategy=SwitchStrategy.FALLBACK
)

# Automatically falls back to next framework on error
result = switcher.execute_task(
    task="Process data",
    current_framework="langchain"
)
```

### Performance Benchmarking

Benchmark provider performance:

```python
from llm_interface.cross_framework import FrameworkBenchmark

benchmark = FrameworkBenchmark(manager)

# Run benchmarks
report = benchmark.run_benchmark(
    providers=["openai", "anthropic", "ollama"],
    tasks=[
        "Generate a short story",
        "Explain a concept",
        "Write code"
    ],
    iterations=3
)

# Get performance insights
print(report.performance_summary())
```

## Provider-Specific Features

### OpenAI

```python
from llm_interface import LLMConfig, ProviderType

config = LLMConfig(
    provider=ProviderType.OPENAI,
    model_name="gpt-4",  # or "gpt-3.5-turbo", "gpt-4-turbo", etc.
    temperature=0.7,
    max_tokens=2000,
    api_key="${OPENAI_API_KEY}"  # Or pass directly
)
```

**Supported Models**: gpt-4, gpt-4-turbo, gpt-3.5-turbo, gpt-4o, and more

### Azure OpenAI

```python
from llm_interface import LLMConfig, ProviderType

config = LLMConfig(
    provider=ProviderType.AZURE,
    model_name="gpt-4",
    azure_endpoint="https://your-resource.openai.azure.com",
    azure_deployment="your-deployment-name",
    azure_api_version="2023-12-01-preview",
    api_key="${AZURE_OPENAI_API_KEY}"
)
```

**Required Fields**: `azure_endpoint`, `azure_deployment`
**Optional**: `azure_api_version` (defaults to latest)

### Ollama (Local Models)

```python
from llm_interface import LLMConfig, ProviderType

config = LLMConfig(
    provider=ProviderType.OLLAMA,
    model_name="llama2",  # or "mistral", "neural-chat", etc.
    base_url="http://localhost:11434",  # Default Ollama endpoint
    temperature=0.7
)
```

**Setup**:
```bash
# Install Ollama from https://ollama.ai
# Pull a model
ollama pull llama2

# Run Ollama server (usually auto-starts)
ollama serve
```

### Anthropic

```python
from llm_interface import LLMConfig, ProviderType

config = LLMConfig(
    provider=ProviderType.ANTHROPIC,
    model_name="claude-3-sonnet-20240229",  # or "claude-3-opus", "claude-3-haiku"
    api_key="${ANTHROPIC_API_KEY}",
    temperature=0.7
)
```

**Supported Models**: claude-3-haiku, claude-3-sonnet, claude-3-opus

## Advanced Usage

### Custom Provider Implementation

Extend the library with custom providers:

```python
from llm_interface.core.interfaces import LLMProvider
from llm_interface.providers.base import BaseProvider
from llm_interface import LLMProviderFactory, ProviderType

class CustomProvider(BaseProvider):
    """Custom LLM provider implementation."""
    
    def get_model(self):
        """Return the model instance."""
        # Implement your custom model
        pass
    
    def get_chat_model(self):
        """Return the chat model instance."""
        # Implement your custom chat model
        pass
    
    def get_embeddings(self):
        """Return embeddings model."""
        # Implement embeddings or raise NotImplementedError
        pass
    
    def validate_config(self) -> bool:
        """Validate configuration."""
        return True

# Register and use custom provider
class CustomProviderType(Enum):
    CUSTOM = "custom"

LLMProviderFactory.register_provider(
    CustomProviderType.CUSTOM, 
    CustomProvider
)
```

### Embeddings Support

Get embeddings from any provider:

```python
# Get embeddings
embeddings = manager.get_embeddings("openai")

# Embed documents
vectors = embeddings.embed_documents([
    "Hello world",
    "How are you?",
    "Machine learning is great"
])

# Embed query
query_vector = embeddings.embed_query("What is AI?")

print(f"Document vectors shape: {len(vectors)}")
print(f"Query vector dimensions: {len(query_vector)}")
```

### Chain Creation

Create chains with specific providers:

```python
# Create a simple chain
chain = manager.create_chain(
    "openai",
    "You are a helpful assistant. Answer the following question:\n{input}"
)

result = chain.invoke({"input": "What is machine learning?"})
print(result)

# Create a more complex chain with templates
from langchain_core.prompts import ChatPromptTemplate

template = ChatPromptTemplate.from_messages([
    ("system", "You are a {role}"),
    ("human", "{query}")
])

chain = template | manager.get_chat_model("anthropic")
response = chain.invoke({"role": "data scientist", "query": "Explain regression"})
```

### Thread-Safe Operations

The manager is fully thread-safe:

```python
import threading
from llm_interface import LLMManager, LLMConfig, ProviderType

manager = LLMManager()

# Add providers from main thread
config = LLMConfig(provider=ProviderType.OPENAI, model_name="gpt-4")
manager.add_provider("openai", config)

# Use from multiple threads safely
def worker(provider_name, query):
    model = manager.get_chat_model(provider_name)
    response = model.invoke(query)
    print(f"Response: {response.content}")

threads = [
    threading.Thread(target=worker, args=("openai", f"Query {i}"))
    for i in range(10)
]

for t in threads:
    t.start()

for t in threads:
    t.join()
```

## Configuration Options

### LLMConfig Parameters

| Parameter | Type | Description | Required | Default |
|-----------|------|-------------|----------|---------|
| `provider` | `ProviderType` | Provider type (OPENAI, AZURE, ANTHROPIC, OLLAMA) | ✅ | — |
| `model_name` | `str` | Model name (gpt-4, claude-3-sonnet, llama2, etc.) | ✅ | — |
| `api_key` | `str` | API key (or use environment variables) | Conditional | None |
| `base_url` | `str` | Custom API base URL | ❌ | None |
| `temperature` | `float` | Sampling temperature (0.0-2.0) | ❌ | 0.7 |
| `max_tokens` | `int` | Maximum tokens in response | ❌ | 1000 |
| `azure_endpoint` | `str` | Azure OpenAI endpoint URL | Azure only | None |
| `azure_deployment` | `str` | Azure OpenAI deployment name | Azure only | None |
| `azure_api_version` | `str` | Azure OpenAI API version | ❌ | Latest |

**Validation Rules**:
- `model_name`: Non-empty string, automatically trimmed
- `temperature`: Must be between 0.0 and 2.0
- `max_tokens`: Must be positive integer
- `base_url`: Must start with http:// or https://
- `azure_endpoint`: Required and validated if provider is AZURE
- `azure_deployment`: Required if provider is AZURE

### Environment Variables

| Variable | Description | Provider |
|----------|-------------|----------|
| `OPENAI_API_KEY` | OpenAI API key | OpenAI |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI API key | Azure |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI endpoint | Azure |
| `ANTHROPIC_API_KEY` | Anthropic API key | Anthropic |
| `LLM_*` | Custom environment prefix | All |

Configuration files support environment variable interpolation using `${VAR_NAME}` syntax.

## Error Handling

The library provides comprehensive error handling with custom exceptions:

```python
from llm_interface import LLMManager, LLMConfig, ProviderType
from llm_interface.core.exceptions import (
    ProviderNotFoundError,
    ProviderAlreadyExistsError,
    InvalidInputError,
    ConfigurationError
)

try:
    # Invalid temperature
    config = LLMConfig(
        provider=ProviderType.OPENAI,
        model_name="gpt-4",
        temperature=3.0  # Must be 0.0-2.0
    )
except ValueError as e:
    print(f"Configuration error: {e}")

try:
    # Provider not found
    manager = LLMManager()
    model = manager.get_chat_model("nonexistent")
except ProviderNotFoundError as e:
    print(f"Provider error: {e}")

try:
    # Duplicate provider
    manager.add_provider("openai", config1)
    manager.add_provider("openai", config2)  # Raises error
except ProviderAlreadyExistsError as e:
    print(f"Duplicate provider: {e}")
```

**Available Exceptions**:
- `ProviderNotFoundError`: Provider doesn't exist in manager
- `ProviderAlreadyExistsError`: Attempting to add duplicate provider
- `InvalidInputError`: Invalid input parameters
- `ConfigurationError`: Configuration validation failed
- `EmbeddingsNotSupportedError`: Provider doesn't support embeddings

## Examples

The project includes comprehensive examples in the `examples/` directory:

### Basic Examples

**Core Examples** (`examples/core/`):
- `example_usage.py` - Basic usage patterns
- `logging_example.py` - Configuration logging

**Provider Examples**:
- `example_usage.py` - Multi-provider setup
- `config_*.yaml` - Configuration templates for different scenarios

### Framework Examples

**LangChain** (`examples/langchain/`):
- `langchain_adapter_examples.py` - LangChain integration
- `langchain_workflow.yaml` - Workflow configuration

**CrewAI** (`examples/crewai/`):
- `crewai_adapter_examples.py` - CrewAI agents
- `crewai_workflow.yaml` - CrewAI workflows

**AutoGen** (`examples/autogen/`):
- `simple_chat.py` - Simple AutoGen conversations
- `group_conversation.py` - Multi-agent conversations

**Semantic Kernel** (`examples/semantic_kernel/`):
- `simple_functions.py` - Basic functions
- `plugins_example.py` - Plugin system

### Cross-Framework Examples

**Utilities** (`examples/cross_framework/`):
- `comparison_example.py` - Framework comparison
- `switching_example.py` - Dynamic framework switching
- `benchmark_example.py` - Performance benchmarking
- `hybrid_workflow.py` - Multi-framework workflows

### Running Examples

```bash
# Basic example
python examples/example_usage.py

# Framework specific
python examples/langchain/langchain_adapter_examples.py
python examples/crewai/crewai_adapter_examples.py
python examples/autogen/simple_chat.py

# Cross-framework tools
python examples/cross_framework/comparison_example.py
python examples/cross_framework/benchmark_example.py
```

### Configuration Examples

```bash
# Different configuration scenarios are in examples/configs/
# - config_local_development.yaml: Local Ollama setup
# - config_multi_provider.yaml: Multiple providers with fallbacks
# - config_specialized.yaml: Specialized providers for different tasks
# - config_production.yaml: Enterprise production setup
# - config_openai_only.yaml: Single provider setup
```

## Testing

The project includes comprehensive tests for all components:

### Test Structure

```
tests/
├── test_config.py              # Configuration validation tests
├── test_config_loader.py       # Config loader tests
├── test_factory.py             # Provider factory tests
├── test_manager.py             # Manager functionality tests
├── test_providers.py           # Provider implementation tests
├── test_integration.py         # Integration tests
├── test_thread_safety.py       # Thread safety tests
└── framework/                  # Framework-specific tests
    ├── autogen/test_autogen_adapter.py
    ├── crewai/test_crewai_adapter.py
    ├── langchain/test_langchain_adapter.py
    └── semantic_kernel/test_sk_adapter.py
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=llm_interface

# Run specific test file
pytest tests/test_manager.py

# Run specific test
pytest tests/test_manager.py::test_add_provider

# Run tests in verbose mode
pytest -v

# Run framework tests only
pytest tests/framework/
```

### Test Development Setup

```bash
pip install -r requirements-dev.txt

# Code formatting
black .

# Linting
flake8 .

# Type checking
mypy .

# Pre-commit hooks
pre-commit install
pre-commit run --all-files
```

### Key Test Areas

- ✅ Configuration validation with Pydantic
- ✅ Provider creation and management
- ✅ Thread safety with concurrent operations
- ✅ Configuration loading (YAML, environment)
- ✅ Framework adapters (LangChain, CrewAI, AutoGen, Semantic Kernel)
- ✅ Cross-framework utilities (comparison, switching, benchmarking)
- ✅ Error handling and custom exceptions
- ✅ Integration with external APIs

## Development Setup

### Setting up the Development Environment

1. **Clone the repository and set up virtual environment:**
   ```bash
   git clone <your-repo-url>
   cd llm_interface
   python -m venv venv
   
   # Activate (Windows PowerShell)
   .\venv\Scripts\Activate.ps1
   
   # Or use convenience script
   .\activate-env.ps1
   ```

2. **Install in development mode:**
   ```bash
   pip install -e .
   pip install -r requirements-dev.txt
   ```

3. **Run tests:**
   ```bash
   pytest
   pytest --cov=llm_interface  # With coverage
   ```

4. **Code formatting and linting:**
   ```bash
   black .                     # Format code
   flake8 .                    # Lint code
   mypy .                      # Type checking
   ```

### Project Structure

```
llm_interface/
├── venv/                   # Virtual environment (created after setup)
├── llm_interface/          # Main package directory
│   ├── __init__.py        # Package entry point
│   └── src/               # Source code
│       ├── __init__.py    # Main module exports
│       ├── core/          # Core interfaces and config
│       │   ├── config.py  # Configuration classes
│       │   └── interfaces.py # Provider interfaces
│       ├── providers/     # Provider implementations
│       │   ├── base.py    # Base provider class
│       │   ├── openai.py  # OpenAI provider
│       │   ├── azure.py   # Azure OpenAI provider
│       │   ├── ollama.py  # Ollama provider
│       │   └── anthropic.py # Anthropic provider
│       ├── manager.py     # LLM Manager (main interface)
│       ├── factory.py     # Provider Factory
│       ├── graph.py       # LangGraph integration
│       └── config_loader.py # Configuration loading
├── requirements.txt       # Package dependencies
├── setup.py              # Package setup configuration
├── test_library.py       # Installation test script
├── test_imports.py       # Import verification script
├── .gitignore           # Git ignore rules
└── README.md            # This documentation
```

## Troubleshooting

### Common Issues

**Import Errors**

```python
# If you get "No module named 'llm_interface'"
# Make sure you've installed the package in editable mode:
pip install -e .

# Test the installation:
python -c "from llm_interface import LLMManager; print('Import successful')"
```

**Virtual Environment Issues**

```bash
# Reactivate virtual environment
source venv/bin/activate  # macOS/Linux
# or
.\venv\Scripts\Activate.ps1  # Windows PowerShell

# Verify you're in the right environment
python -c "import sys; print(sys.prefix)"
```

**Provider Configuration Issues**

- Ensure API keys are set correctly in environment variables
- Check model names are valid for each provider
- Verify endpoints for Azure and Ollama configurations
- Use `ConfigLoader.from_yaml()` with proper YAML syntax
- Validate YAML with a YAML linter before loading

**Connection Issues**

```python
# Test provider connectivity
from llm_interface import LLMManager, LLMConfig, ProviderType

try:
    config = LLMConfig(
        provider=ProviderType.OPENAI,
        model_name="gpt-4",
        api_key="your-key"
    )
    manager = LLMManager()
    manager.add_provider("test", config)
    print("Provider configured successfully")
except Exception as e:
    print(f"Configuration failed: {e}")
```

**Ollama Connection Issues**

```bash
# Verify Ollama is running
curl http://localhost:11434/api/tags

# If not running, start Ollama:
ollama serve

# Check available models:
ollama list

# Pull a model if needed:
ollama pull llama2
```

**Rate Limiting**

```python
# Implement retry logic for rate-limited APIs
from langchain.callbacks import Callbacks
import time

def invoke_with_retry(model, prompt, max_retries=3):
    for attempt in range(max_retries):
        try:
            return model.invoke(prompt)
        except Exception as e:
            if "rate limit" in str(e).lower():
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"Rate limited. Waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise
    raise Exception("Max retries exceeded")
```

### Debug Mode

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)
logging.getLogger('llm_interface').setLevel(logging.DEBUG)

# This will show detailed information about what's happening
manager = LLMManager()
```

### Getting Help

1. Check the documentation in this README
2. Review example files in the `examples/` directory
3. Check test files for usage patterns
4. Enable debug logging to see detailed information
5. Verify configuration with `python -c "from llm_interface import *; print('Import successful')"`

## Security Best Practices

### ⚠️ IMPORTANT: Protect Your API Keys

**Never commit API keys to version control!**

```bash
# Add to .gitignore
echo "*.yaml" >> .gitignore
echo ".env" >> .gitignore
echo "config*.yaml" >> .gitignore
echo "*.key" >> .gitignore
```

### Use Environment Variables

```python
# GOOD - Use environment variables
import os
api_key = os.getenv("OPENAI_API_KEY")

# BAD - Never hardcode API keys
api_key = "sk-1234567890..."  # DON'T DO THIS!
```

### Use .env Files

```bash
# Create .env file (add to .gitignore!)
cat > .env << EOF
OPENAI_API_KEY=sk-your-key-here
AZURE_OPENAI_API_KEY=your-azure-key
ANTHROPIC_API_KEY=your-anthropic-key
EOF

# Load in Python
from dotenv import load_dotenv
load_dotenv()  # Loads .env variables
```

### Safe YAML Configuration

```yaml
# config.yaml - Safe to commit!
providers:
  openai:
    provider: "openai"
    model_name: "gpt-4"
    api_key: "${OPENAI_API_KEY}"  # References environment variable
    temperature: 0.7
```

### Vault Integration

For production environments, use secure vaults:

```python
# Example with AWS Secrets Manager
import boto3

def get_secret(secret_name):
    client = boto3.client('secretsmanager')
    response = client.get_secret_value(SecretId=secret_name)
    return response['SecretString']

# Use in configuration
from llm_interface import LLMConfig, ProviderType

openai_key = get_secret('openai/api-key')
config = LLMConfig(
    provider=ProviderType.OPENAI,
    model_name="gpt-4",
    api_key=openai_key
)
```

## Logging Configuration

### Basic Logging Setup

```python
import logging
from llm_interface import LLMManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('llm_interface.log'),
        logging.StreamHandler()
    ]
)

# Enable debug logging for troubleshooting
logging.getLogger('llm_interface').setLevel(logging.DEBUG)

# Create manager - now with logging
manager = LLMManager()
```

### Advanced Logging with Filtering

Protect sensitive information in logs:

```python
import logging
import re

class APIKeyFilter(logging.Filter):
    """Filter to mask API keys in log messages."""
    
    def filter(self, record):
        # Mask common API key patterns
        if hasattr(record, 'msg'):
            record.msg = re.sub(
                r'(sk-[a-zA-Z0-9]{20,})',
                'sk-***REDACTED***',
                str(record.msg)
            )
            record.msg = re.sub(
                r'(api[_-]?key["\']?\s*[:=]\s*["\']?)([^"\'\\s]+)',
                r'\1***REDACTED***',
                str(record.msg),
                flags=re.IGNORECASE
            )
        return True

# Add filter to handlers
logger = logging.getLogger('llm_interface')
for handler in logger.handlers:
    handler.addFilter(APIKeyFilter())
```

### Structured Logging

```python
import logging.config

LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'detailed': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        },
        'simple': {
            'format': '%(levelname)s - %(message)s'
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'INFO',
            'formatter': 'simple',
            'stream': 'ext://sys.stdout'
        },
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': 'DEBUG',
            'formatter': 'detailed',
            'filename': 'llm_interface.log',
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5
        }
    },
    'loggers': {
        'llm_interface': {
            'level': 'DEBUG',
            'handlers': ['console', 'file'],
            'propagate': False
        }
    }
}

logging.config.dictConfig(LOGGING_CONFIG)
```

### Logging Best Practices

- Use DEBUG level during development
- Use INFO level in production
- Never log API keys (use the filtering approach above)
- Use rotating file handlers for long-running processes
- Monitor log files for errors and warnings
- Include context information in debug logs

**Provider Configuration:**
- Ensure API keys are set correctly in environment variables
- Check model names are valid for each provider
- Verify endpoints for Azure and Ollama configurations

## Contributing

We welcome contributions! Here's how to get started:

### Development Setup

1. **Fork and clone the repository:**
   ```bash
   git clone https://github.com/your-username/llm-interface.git
   cd llm-interface
   ```

2. **Set up development environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # or use appropriate activation script
   pip install -e ".[dev]"
   ```

3. **Install pre-commit hooks:**
   ```bash
   pre-commit install
   ```

### Code Quality

```bash
# Format code with Black
black .

# Lint with Flake8
flake8 .

# Type checking with mypy
mypy llm_interface/

# Run tests
pytest

# Run tests with coverage
pytest --cov=llm_interface
```

### Submitting Changes

1. Create a feature branch: `git checkout -b feature/your-feature`
2. Make your changes and add tests
3. Ensure all tests pass and code is formatted
4. Commit: `git commit -am 'Add new feature'`
5. Push: `git push origin feature/your-feature`
6. Submit a pull request

### Testing Requirements

- All new features must have unit tests
- Tests should be in the appropriate `tests/` subdirectory
- Aim for >80% code coverage
- Use pytest for testing framework
- Mock external API calls in tests

### Documentation

- Update README.md for user-facing changes
- Add docstrings to all public classes and methods
- Include type hints throughout code
- Update examples if behavior changes

## License

MIT License - see LICENSE file for details

## Support

For issues and questions:
- 📧 Create an issue on GitHub
- 📚 Check the documentation in this README
- 💡 Review the examples in the `examples/` directory
- 🔍 Search existing issues for similar problems

## Changelog

### v1.0.0 - Current Release

**Core Features**:
- Multi-provider support (OpenAI, Azure, Anthropic, Ollama)
- Unified interface across all providers
- Thread-safe operations with reentrant locks
- Comprehensive Pydantic validation
- Full type annotations

**Framework Support**:
- LangChain adapter with graph integration
- CrewAI adapter
- AutoGen adapter
- Semantic Kernel adapter

**Utilities**:
- Framework comparison tools
- Dynamic framework switching
- Performance benchmarking
- Configuration management (YAML, environment variables)

**Quality**:
- Comprehensive test suite
- Production-ready error handling
- Detailed logging with security filters
- Security best practices documented