# Semantic Kernel Examples

This directory contains examples demonstrating the Semantic Kernel framework adapter for llm_interface.

## Examples

### 1. `simple_functions.py`
Basic semantic and native functions showing:
- Creating kernels with different providers
- Semantic functions (AI-powered)
- Native functions (Python code)
- Function invocation

### 2. `plugins_example.py`
Plugin system demonstration showing:
- Creating plugins (collections of functions)
- Organizing related functions
- Plugin registration and usage

### 3. `config_sk.yaml`
YAML configuration example showing:
- Provider definitions
- Kernel configurations
- Function and plugin definitions

## Running Examples

### Prerequisites
```bash
# Install Semantic Kernel support
pip install semantic-kernel

# Or install with extras
pip install -e ".[semantic-kernel]"
```

### Run Simple Functions
```bash
python examples/semantic_kernel/simple_functions.py
```

### Run Plugins Example
```bash
python examples/semantic_kernel/plugins_example.py
```

### Using Configuration Files
```bash
python examples/semantic_kernel/simple_functions.py --config examples/semantic_kernel/config_sk.yaml
```

## Configuration

Set required API keys:
```bash
# For OpenAI
export OPENAI_API_KEY="your-key-here"

# For Azure
export AZURE_OPENAI_KEY="your-key-here"
export AZURE_OPENAI_ENDPOINT="your-endpoint-here"
```

## Notes

- Semantic Kernel requires Python 3.8+
- Async/await patterns are used for function invocation
- Semantic functions use prompt templates with {{$variables}}
- Native functions can be any Python callable
- Plugins organize related functions together
