# LLM Interface

A flexible Python library for handling multiple LLM providers with LangGraph integration. Easily switch between OpenAI, Azure OpenAI, Anthropic, and Ollama with a unified interface.

## Features

- **Multi-Provider Support**: OpenAI, Azure OpenAI, Anthropic, Ollama
- **Unified Interface**: Same API across all providers
- **LangGraph Integration**: Build complex AI workflows
- **Flexible Configuration**: YAML, environment variables, or programmatic
- **Factory Pattern**: Easy provider instantiation
- **Extensible**: Add custom providers easily
- **Type Safe**: Full type annotations
- **Error Handling**: Robust validation and error management

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

Required packages (automatically installed):
```
langgraph>=0.1.0
pydantic>=1.10
langchain-community>=0.0.1
langchain-ollama>=0.0.1
langchain-core>=0.1.0
langchain-openai>=0.1.0
langchain-anthropic>=0.1.0
PyYAML>=6.0
setuptools>=40.0.0
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
    model_name="gpt-4",  # Note: use model_name, not model
    api_key="your-api-key",
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
    api_key: "${OPENAI_API_KEY}"
    temperature: 0.7
    max_tokens: 1000
  
  azure:
    provider: "azure"
    model_name: "gpt-4"
    api_key: "${AZURE_OPENAI_API_KEY}"
    azure_endpoint: "${AZURE_OPENAI_ENDPOINT}"
    azure_deployment: "gpt-4-deployment"
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

# Load from environment variables
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
    print("✅ LLMManager created successfully")
    
    # Test provider factory
    from llm_interface import LLMProviderFactory
    factory = LLMProviderFactory()
    supported = factory.list_supported_providers()
    print(f"✅ Supported providers: {[p.value for p in supported]}")
    
    # Test adding a provider (no real API key needed for this test)
    test_config = LLMConfig(
        provider=ProviderType.OPENAI,
        model_name="gpt-3.5-turbo",
        api_key="test-key"
    )
    manager.add_provider("test", test_config)
    print(f"✅ Provider added. Current providers: {manager.list_providers()}")
    
    print("🎉 Installation test completed successfully!")

if __name__ == "__main__":
    test_installation()
```

Run the test:
```bash
python test_installation.py
```

## LangGraph Integration

Build complex workflows with LangGraph:

```python
from llm_interface import LLMManager, LLMGraph
from langchain_core.messages import HumanMessage

# Setup
manager = LLMManager()
# ... add providers ...

# Create LangGraph
graph = LLMGraph(manager)

# Simple chat
messages = [HumanMessage(content="Explain quantum computing")]
result = graph.run_simple_chat(messages, "openai")
print(result["response"])

# Custom workflow
workflow_definition = {
    "nodes": {
        "analyze": {"provider": "openai"},
        "summarize": {"provider": "anthropic"},
        "finalize": {"function": custom_function}
    },
    "edges": {
        "analyze": "summarize",
        "summarize": "finalize"
    },
    "entry_point": "analyze"
}

custom_graph = graph.build_custom_graph(workflow_definition)
compiled_graph = graph.compile_graph(custom_graph)
```

### Multi-Provider Workflows

```python
from llm_interface.graph import MultiProviderRouter, GraphBuilder

# Router with fallback
router = MultiProviderRouter(manager)
router.set_fallback_chain(["openai", "anthropic", "ollama"])

# Build complex graph
builder = GraphBuilder(manager)
workflow = (builder
    .add_llm_node("initial_analysis", "openai")
    .add_llm_node("review", "anthropic")
    .add_custom_node("validate", validation_function)
    .set_entry_point("initial_analysis")
    .add_edge("initial_analysis", "review")
    .add_edge("review", "validate")
    .build())

result = workflow.invoke({"messages": [HumanMessage(content="Analyze this")]})
```

## Provider-Specific Features

### OpenAI

```python
from llm_interface.providers.openai import OpenAIProviderUtils

# Quick configs
config = OpenAIProviderUtils.create_gpt4_config(temperature=0.5)
provider = OpenAIProviderUtils.from_environment()
```

### Azure OpenAI

```python
azure_config = LLMConfig(
    provider=ProviderType.AZURE,
    model_name="gpt-4",
    azure_endpoint="https://your-resource.openai.azure.com",
    azure_deployment="gpt-4-deployment",
    api_key="your-key"
)
```

### Ollama (Local Models)

```python
ollama_config = LLMConfig(
    provider=ProviderType.OLLAMA,
    model_name="llama2",
    base_url="http://localhost:11434"  # Default Ollama endpoint
)
```

### Anthropic

```python
anthropic_config = LLMConfig(
    provider=ProviderType.ANTHROPIC,
    model_name="claude-3-sonnet-20240229",
    api_key="your-anthropic-key"
)
```

## Advanced Usage

### Custom Provider

```python
from llm_interface.core.interfaces import LLMProvider
from llm_interface.providers.base import BaseProvider

class CustomProvider(BaseProvider):
    def get_model(self):
        # Implement your custom model
        pass
    
    def get_chat_model(self):
        # Implement your custom chat model
        pass
    
    def get_embeddings(self):
        # Implement your custom embeddings
        pass
    
    def validate_config(self) -> bool:
        # Validate your custom config
        return True

# Register custom provider
from llm_interface import LLMProviderFactory, ProviderType

LLMProviderFactory.register_provider(
    ProviderType.CUSTOM, 
    CustomProvider
)
```

### Embeddings

```python
# Get embeddings from any provider
embeddings = manager.get_embeddings("openai")
vectors = embeddings.embed_documents(["Hello world", "How are you?"])
```

### Chain Creation

```python
# Create chains with specific providers
chain = manager.create_chain(
    "openai", 
    "You are a helpful assistant. User: {input}"
)

result = chain.invoke({"input": "What is machine learning?"})
```

## Configuration Options

### LLMConfig Parameters

| Parameter | Type | Description | Required |
|-----------|------|-------------|----------|
| `provider` | `ProviderType` | Provider type (OPENAI, AZURE, etc.) | ✅ |
| `model_name` | `str` | Model name | ✅ |
| `api_key` | `str` | API key (or from env) | Conditional |
| `base_url` | `str` | Base URL for API | ❌ |
| `temperature` | `float` | Temperature (0.0-2.0) | ❌ (0.7) |
| `max_tokens` | `int` | Maximum tokens | ❌ (1000) |
| `azure_endpoint` | `str` | Azure endpoint URL | Azure only |
| `azure_deployment` | `str` | Azure deployment name | Azure only |

### Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI API key |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI endpoint |
| `ANTHROPIC_API_KEY` | Anthropic API key |
| `LLM_*` | Custom environment prefix |

## Error Handling

```python
from llm_interface import LLMManager, LLMConfig, ProviderType

try:
    config = LLMConfig(
        provider=ProviderType.OPENAI,
        model_name="gpt-4",
        temperature=3.0  # Invalid temperature
    )
except ValueError as e:
    print(f"Configuration error: {e}")

try:
    manager = LLMManager()
    manager.add_provider("openai", config)
    model = manager.get_chat_model("nonexistent")
except ValueError as e:
    print(f"Provider error: {e}")
```

## Examples

### Chat Application

```python
from llm_interface import LLMManager, ConfigLoader

def chat_app():
    # Load configuration
    configs = ConfigLoader.from_yaml("config.yaml")
    manager = LLMManager()
    
    for name, config in configs.items():
        manager.add_provider(name, config)
    
    # Interactive chat
    provider = "openai"  # or any available provider
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break
        
        try:
            model = manager.get_chat_model(provider)
            response = model.invoke(user_input)
            print(f"AI: {response.content}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    chat_app()
```

### Batch Processing

```python
from llm_interface import LLMManager, LLMGraph
from langchain_core.messages import HumanMessage

def batch_process(texts, provider="openai"):
    manager = LLMManager()
    # ... setup providers ...
    
    graph = LLMGraph(manager)
    results = []
    
    for text in texts:
        messages = [HumanMessage(content=f"Summarize: {text}")]
        result = graph.run_simple_chat(messages, provider)
        results.append(result["response"])
    
    return results

# Usage
texts = ["Long text 1...", "Long text 2...", "Long text 3..."]
summaries = batch_process(texts)
```

## Testing

Run the built-in test script:

```bash
python test_library.py
```

This will test:
- All core imports
- Manager functionality  
- Provider factory
- Configuration creation
- Basic provider operations

### Test Structure

The project includes comprehensive tests for all components:

```
tests/
├── test_config.py        # Configuration tests
├── test_providers/       # Provider-specific tests  
│   ├── test_openai.py
│   ├── test_azure.py
│   ├── test_anthropic.py
│   └── test_ollama.py
├── test_manager.py       # Manager tests
├── test_factory.py       # Factory tests
└── test_graph.py         # LangGraph integration tests
```

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

**Import Errors:**
```python
# If you get "No module named 'llm_interface'"
# Make sure you've installed the package:
pip install -e .

# Test the installation:
python -c "from llm_interface import LLMManager; print('✅ Import successful')"
```

**Virtual Environment Issues:**
```bash
# Reactivate virtual environment
.\venv\Scripts\Activate.ps1  # Windows PowerShell
# or
venv\Scripts\activate.bat    # Windows Command Prompt

# Verify you're in the right environment
python -c "import sys; print(sys.prefix)"
```

**Provider Configuration:**
- Ensure API keys are set correctly in environment variables
- Check model names are valid for each provider
- Verify endpoints for Azure and Ollama configurations

### Security Best Practices

**⚠️ IMPORTANT: Never commit API keys to version control!**

**Protect Your API Keys:**
```bash
# Add to .gitignore
echo "*.yaml" >> .gitignore
echo ".env" >> .gitignore
echo "config*.yaml" >> .gitignore
```

**Use Environment Variables:**
```python
# ✅ GOOD - Use environment variables
import os
api_key = os.getenv("OPENAI_API_KEY")

# ❌ BAD - Never hardcode API keys
api_key = "sk-1234567890..."  # DON'T DO THIS!
```

**Use .env Files (excluded from git):**
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

**YAML Configuration with Environment Variables:**
```yaml
# config.yaml - Safe to commit!
providers:
  openai:
    provider: "openai"
    model_name: "gpt-4"
    api_key: "${OPENAI_API_KEY}"  # References environment variable
    temperature: 0.7
```

**Key Security Checklist:**
- ✅ Store keys in environment variables or secure vaults
- ✅ Add config files with keys to .gitignore
- ✅ Use different keys for dev/staging/production
- ✅ Rotate keys regularly
- ✅ Never log or print API keys
- ✅ Use read-only keys when possible
- ❌ Never commit keys to git
- ❌ Never share keys in chat/email
- ❌ Never hardcode keys in source code

### Logging Configuration

**Basic Logging Setup:**
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

**Advanced Logging with Filtering (Protect API Keys):**
```python
import logging
import re

class APIKeyFilter(logging.Filter):
    \"\"\"Filter to mask API keys in log messages.\"\"\"
    
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

**Structured Logging Example:**
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

**Provider Configuration:**
- Ensure API keys are set correctly in environment variables
- Check model names are valid for each provider
- Verify endpoints for Azure and Ollama configurations

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for your changes
4. Run the test suite
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Changelog

### v1.0.0 - Current Release
- ✅ **Core functionality implemented and tested**
- ✅ **Multi-provider support** (OpenAI, Azure, Anthropic, Ollama)
- ✅ **Import system fixed** - All imports working correctly
- ✅ **Package structure optimized** - Proper setuptools configuration
- ✅ **LangGraph integration** - Advanced workflow capabilities
- ✅ **Configuration management** - YAML and environment variable support
- ✅ **Type safety** - Full type annotations throughout
- ✅ **Error handling** - Comprehensive validation and error messages
- ✅ **Factory pattern** - Thread-safe provider instantiation
- ✅ **Virtual environment support** - Complete dependency management

## Support

For issues and questions:
- Create an issue on GitHub
- Check the documentation
- Review the examples

## Roadmap

- [ ] Google Gemini provider
- [ ] Async support
- [ ] Caching layer
- [ ] Monitoring/metrics
- [ ] Plugin system
- [ ] CLI interface