# AutoGen Examples

This directory contains examples demonstrating the AutoGen framework adapter for llm_interface.

## Examples

### 1. `simple_chat.py`
Basic two-agent conversation showing:
- Creating agents with different providers
- Initiating conversations
- User proxy patterns

### 2. `group_conversation.py`
Multi-agent group chat demonstrating:
- Creating multiple specialized agents
- Group chat orchestration
- Speaker selection methods

### 3. `config_autogen.yaml`
YAML configuration example showing:
- Provider definitions
- Agent configurations
- Group chat settings

## Running Examples

### Prerequisites
```bash
# Install AutoGen support
pip install pyautogen

# Or install with extras
pip install -e ".[autogen]"
```

### Run Simple Chat
```bash
python examples/autogen/simple_chat.py
```

### Run Group Conversation
```bash
python examples/autogen/group_conversation.py
```

### Using Configuration Files
```bash
python examples/autogen/simple_chat.py --config examples/autogen/config_autogen.yaml
```

## Configuration

Set required API keys:
```bash
# For OpenAI
export OPENAI_API_KEY="your-key-here"

# For Anthropic
export ANTHROPIC_API_KEY="your-key-here"

# For Azure
export AZURE_OPENAI_KEY="your-key-here"
export AZURE_OPENAI_ENDPOINT="your-endpoint-here"
```

## Notes

- AutoGen requires Python 3.8+
- Some examples may require additional dependencies
- Set `human_input_mode="ALWAYS"` for interactive conversations
- Use `human_input_mode="NEVER"` for automated workflows
