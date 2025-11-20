# Configuration Examples

This directory contains YAML configuration examples for various deployment scenarios.

## Available Configurations

### config_openai_only.yaml
Simple configuration using only OpenAI provider.
```yaml
providers:
  openai:
    provider: openai
    model_name: gpt-4
    api_key: ${OPENAI_API_KEY}
```

### config_multi_provider.yaml
Configuration with multiple providers (OpenAI, Azure, Anthropic, Ollama).
Use this for applications that need flexibility in choosing between different LLM providers.

### config_local_development.yaml
Configuration optimized for local development using Ollama.
Great for testing without incurring cloud API costs.

### config_production.yaml
Production-ready configuration with environment variable substitution and proper defaults.

### config_specialized.yaml
Configuration showing specialized use cases like different temperatures for different tasks.

## Usage

### Loading from YAML

```python
from llm_interface import ConfigLoader, LLMManager

# Load configuration
configs = ConfigLoader.from_yaml('configs/config_multi_provider.yaml')

# Create manager and add providers
manager = LLMManager()
for name, config in configs.items():
    manager.add_provider(name, config)
```

### Environment Variables

All configurations support environment variable substitution using `${VAR_NAME}` syntax:

```yaml
api_key: ${OPENAI_API_KEY}  # Will be replaced with environment variable value
```

## Best Practices

1. **Never commit API keys** - Always use environment variables
2. **Use descriptive provider names** - Makes code more readable
3. **Set appropriate defaults** - Temperature, max_tokens, etc.
4. **Separate configs by environment** - dev, staging, production

## See Also

- Main documentation: `../../README.md`
- Core examples: `../core/`
