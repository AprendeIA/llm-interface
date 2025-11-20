# LLM Interface Examples

This directory contains organized examples demonstrating all features of the LLM Interface library.

## 📁 Directory Structure

```
examples/
├── configs/              # Configuration file examples
│   ├── config_openai_only.yaml
│   ├── config_multi_provider.yaml
│   ├── config_local_development.yaml
│   ├── config_production.yaml
│   ├── config_specialized.yaml
│   └── README.md
├── core/                 # Core functionality examples
│   ├── example_usage.py
│   ├── logging_example.py
│   └── README.md
├── autogen/             # Microsoft AutoGen framework
│   ├── simple_chat.py
│   ├── group_conversation.py
│   ├── config_autogen.yaml
│   └── README.md
├── crewai/              # CrewAI framework
│   ├── crewai_adapter_examples.py
│   ├── crewai_workflow.yaml
│   └── README.md
├── langchain/           # LangChain & LangGraph
│   ├── langchain_adapter_examples.py
│   ├── langchain_workflow.yaml
│   └── README.md
├── semantic_kernel/     # Microsoft Semantic Kernel
│   ├── simple_functions.py
│   ├── plugins_example.py
│   ├── config_sk.yaml
│   └── README.md
├── cross_framework/     # Cross-framework tools
│   ├── comparison_example.py
│   ├── switching_example.py
│   ├── benchmark_example.py
│   ├── hybrid_workflow.py
│   └── README.md
└── README.md           # This file
```

## 🚀 Quick Start

### 1. Set Up Environment Variables

**Windows (PowerShell):**
```powershell
$env:OPENAI_API_KEY = "your-openai-api-key"
$env:AZURE_OPENAI_API_KEY = "your-azure-api-key"
$env:AZURE_OPENAI_ENDPOINT = "https://your-resource.openai.azure.com/"
$env:ANTHROPIC_API_KEY = "your-anthropic-api-key"
```

**macOS/Linux:**
```bash
export OPENAI_API_KEY="your-openai-api-key"
export AZURE_OPENAI_API_KEY="your-azure-api-key"
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
export ANTHROPIC_API_KEY="your-anthropic-api-key"
```

### 2. Choose Your Starting Point

| Goal | Directory | Example File |
|------|-----------|--------------|
| Learn basics | `core/` | `example_usage.py` |
| Use AutoGen | `autogen/` | `simple_chat.py` |
| Use CrewAI | `crewai/` | `crewai_adapter_examples.py` |
| Use LangChain | `langchain/` | `langchain_adapter_examples.py` |
| Use Semantic Kernel | `semantic_kernel/` | `simple_functions.py` |
| Compare frameworks | `cross_framework/` | `comparison_example.py` |
| Production config | `configs/` | `config_production.yaml` |

### 3. Run an Example

```bash
# Basic usage
python examples/core/example_usage.py

# Framework-specific
python examples/autogen/simple_chat.py
python examples/langchain/langchain_adapter_examples.py

# Cross-framework comparison
python examples/cross_framework/comparison_example.py
```

## 📋 Example Categories

### 🔧 Core Examples (`core/`)
Basic functionality without framework dependencies:
- **example_usage.py** - Provider setup, basic chat, chains
- **logging_example.py** - Logging configuration and debugging

### 🤖 Framework Examples

#### AutoGen (`autogen/`)
Microsoft AutoGen for multi-agent conversations:
- **simple_chat.py** - Basic agent conversations
- **group_conversation.py** - Multi-agent group chats
- **config_autogen.yaml** - AutoGen-specific configuration

#### CrewAI (`crewai/`)
CrewAI for collaborative agent teams:
- **crewai_adapter_examples.py** - Agents, tasks, and crews
- **crewai_workflow.yaml** - Workflow configuration

#### LangChain (`langchain/`)
LangChain and LangGraph for chains and workflows:
- **langchain_adapter_examples.py** - Chains, graphs, workflows
- **langchain_workflow.yaml** - LangGraph workflow configuration

#### Semantic Kernel (`semantic_kernel/`)
Microsoft Semantic Kernel for AI orchestration:
- **simple_functions.py** - Semantic and native functions
- **plugins_example.py** - Plugin creation and usage
- **config_sk.yaml** - Semantic Kernel configuration

### 🔄 Cross-Framework Examples (`cross_framework/`)
Tools that work across multiple frameworks:
- **comparison_example.py** - Compare same task across frameworks
- **switching_example.py** - Dynamic framework switching with fallback
- **benchmark_example.py** - Performance benchmarking
- **hybrid_workflow.py** - Multi-framework workflows

### ⚙️ Configuration Examples (`configs/`)
YAML configurations for different scenarios:
- **config_openai_only.yaml** - Simple OpenAI setup
- **config_multi_provider.yaml** - Multiple providers
- **config_local_development.yaml** - Local Ollama models
- **config_production.yaml** - Production-ready configuration
- **config_specialized.yaml** - Task-specific configurations

## 🚀 Quick Start

Select the configuration file that best matches your use case:

| Use Case | Configuration File | Description |
|----------|-------------------|-------------|
| Simple OpenAI usage | `configs/config_openai_only.yaml` | Single provider, minimal setup |
| Multi-provider setup | `configs/config_multi_provider.yaml` | Multiple providers with fallbacks |
| Local development | `configs/config_local_development.yaml` | Privacy-focused local models |
| Production deployment | `configs/config_production.yaml` | Enterprise-grade with monitoring |
| Specialized tasks | `configs/config_specialized.yaml` | Different providers for different tasks |

### 2. Load and Use Configuration

```python
from llm_interface import ConfigLoader, LLMManager

# Load configuration
configs = ConfigLoader.from_yaml('examples/configs/config_multi_provider.yaml')

# Create manager and add providers
manager = LLMManager()
for name, config in configs.items():
    manager.add_provider(name, config)

# Use the providers
chat_model = manager.get_chat_model('openai_primary')
response = chat_model.invoke("Hello, how are you?")
print(response.content)
```

## 🎯 Common Use Cases

### Use Case 1: Multi-Agent Conversation (AutoGen)
```python
# See examples/autogen/group_conversation.py
from llm_interface import LLMManager
from llm_interface.framework.autogen import AutoGenAdapter

manager = LLMManager()
# ... add providers ...

adapter = AutoGenAdapter(manager)
agent1 = adapter.create_agent("assistant", "openai")
agent2 = adapter.create_agent("critic", "openai")
# ... create group chat ...
```

### Use Case 2: Task Automation (CrewAI)
```python
# See examples/crewai/crewai_adapter_examples.py
from llm_interface.framework.crewai import CrewAIAdapter

adapter = CrewAIAdapter(manager)
agent = adapter.create_agent(
    role="Researcher",
    goal="Find information",
    provider_name="openai"
)
# ... create tasks and crew ...
```

### Use Case 3: Workflow Orchestration (LangGraph)
```python
# See examples/langchain/langchain_adapter_examples.py
from llm_interface.framework.langchain import LLMGraph

graph = LLMGraph(manager)
workflow = graph.create_simple_chat_graph()
result = graph.run_simple_chat(messages, "openai")
```

### Use Case 4: Framework Comparison
```python
# See examples/cross_framework/comparison_example.py
from llm_interface.cross_framework import FrameworkComparison

comparison = FrameworkComparison(manager)
comparison.add_framework("langchain", langchain_adapter)
comparison.add_framework("crewai", crewai_adapter)

report = comparison.compare_simple_prompt(
    "Explain quantum computing",
    "openai"
)
print(report.summary())
```

## 📚 Learning Path

**Beginner:**
1. Start with `core/example_usage.py` - Learn basic provider setup
2. Try `configs/config_openai_only.yaml` - Simple configuration
3. Run `core/logging_example.py` - Understand debugging

**Intermediate:**
4. Explore `langchain/langchain_adapter_examples.py` - Chains and graphs
5. Try `crewai/crewai_adapter_examples.py` - Multi-agent systems
6. Experiment with `configs/config_multi_provider.yaml` - Multiple providers

**Advanced:**
7. Study `cross_framework/comparison_example.py` - Framework comparison
8. Learn `cross_framework/switching_example.py` - Dynamic switching
9. Optimize with `cross_framework/benchmark_example.py` - Performance tuning
10. Build with `cross_framework/hybrid_workflow.py` - Multi-framework apps

## 🛠️ Testing Your Configuration

```python
from llm_interface import ConfigLoader, LLMManager

try:
    configs = ConfigLoader.from_yaml('configs/config_openai_only.yaml')
    manager = LLMManager()
    
    for name, config in configs.items():
        manager.add_provider(name, config)
        print(f"✅ Successfully loaded provider: {name}")
        
except Exception as e:
    print(f"❌ Configuration error: {e}")
```

## 📚 Additional Resources

- **Main Documentation**: `../README.md` - Library overview and installation
- **Comprehensive Code Review**: `../COMPREHENSIVE_CODE_REVIEW.md` - Code quality assessment
- **Architecture Guide**: `../ARCHITECTURE_EXPANSION.md` - Design and architecture
- **Test Cases**: `../tests/` - Example usage patterns in tests

## 🆘 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure dependencies installed: `pip install -r requirements.txt`
2. **API Key Errors**: Verify environment variables are set correctly
3. **Model Not Found**: Check model names match provider documentation
4. **Framework Not Available**: Install optional dependencies as needed

### Getting Help

1. Check the error message and stack trace
2. Verify configuration syntax in YAML files
3. Test with minimal configuration first (`configs/config_openai_only.yaml`)
4. Review examples in the appropriate framework directory
5. Check README files in each subdirectory for specific guidance

## 🎉 Next Steps

1. **Explore the examples** in each directory
2. **Read the README** files for detailed guidance
3. **Run the examples** to see them in action
4. **Customize configurations** for your needs
5. **Build your application** using the patterns you've learned

Happy coding! 🚀