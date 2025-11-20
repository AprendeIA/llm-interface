#!/usr/bin/env python3
"""
Example script demonstrating how to use the LLM Interface with different configurations.
This script shows various usage patterns and configuration loading methods.
"""

import sys
import os
from pathlib import Path

# Add src to path for the example
project_root = Path(__file__).parent.parent
src_path = project_root / "llm_interface" / "src"
sys.path.insert(0, str(src_path))

def example_basic_usage():
    """Example 1: Basic usage with programmatic configuration"""
    print("🔧 Example 1: Basic Usage")
    print("-" * 40)
    
    try:
        # Import the modules
        import importlib.util
        
        def load_module(name, file_path):
            spec = importlib.util.spec_from_file_location(name, file_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            return module
        
        # Load required modules
        config_module = load_module("config", src_path / "core" / "config.py")
        
        LLMConfig = config_module.LLMConfig
        ProviderType = config_module.ProviderType
        
        # Create a simple configuration
        config = LLMConfig(
            provider=ProviderType.OPENAI,
            model_name="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=1000
        )
        
        print(f"✅ Created config for {config.provider.value}")
        print(f"   Model: {config.model_name}")
        print(f"   Temperature: {config.temperature}")
        print(f"   Max tokens: {config.max_tokens}")
        
    except Exception as e:
        print(f"❌ Basic usage example failed: {e}")
    
    print()


def example_yaml_configuration():
    """Example 2: Loading configuration from YAML files"""
    print("📄 Example 2: YAML Configuration Loading")
    print("-" * 40)
    
    examples_dir = Path(__file__).parent
    config_files = [
        "config_openai_only.yaml",
        "config_multi_provider.yaml", 
        "config_local_development.yaml",
        "config_specialized.yaml"
    ]
    
    for config_file in config_files:
        config_path = examples_dir / config_file
        if config_path.exists():
            print(f"✅ Found: {config_file}")
            
            # Show how you would load it (without actually loading due to import issues)
            print(f"   Load with: ConfigLoader.from_yaml('{config_file}')")
        else:
            print(f"❌ Missing: {config_file}")
    
    print("\n📝 Example loading code:")
    print("""
    from llm_interface import ConfigLoader, LLMManager
    
    # Load configuration from YAML
    configs = ConfigLoader.from_yaml('examples/config_multi_provider.yaml')
    
    # Create manager and add all providers
    manager = LLMManager()
    for name, config in configs.items():
        manager.add_provider(name, config)
    
    # Use different providers for different tasks
    primary_model = manager.get_chat_model('openai_primary')
    fast_model = manager.get_chat_model('azure_enterprise')
    local_model = manager.get_chat_model('local_llama')
    """)
    print()


def example_environment_setup():
    """Example 3: Environment variable setup"""
    print("🌍 Example 3: Environment Variable Setup")
    print("-" * 40)
    
    env_vars = [
        ("OPENAI_API_KEY", "Your OpenAI API key"),
        ("AZURE_OPENAI_API_KEY", "Your Azure OpenAI API key"),
        ("AZURE_OPENAI_ENDPOINT", "Your Azure OpenAI endpoint URL"),
        ("ANTHROPIC_API_KEY", "Your Anthropic API key")
    ]
    
    print("Environment variables to set:")
    for var_name, description in env_vars:
        current_value = os.environ.get(var_name, "Not set")
        status = "✅" if current_value != "Not set" else "❌"
        print(f"  {status} {var_name}: {description}")
        if current_value != "Not set":
            print(f"      Current: {current_value[:10]}...")
    
    print("\n💡 Setup instructions:")
    print("# Windows (PowerShell)")
    print('$env:OPENAI_API_KEY = "your-api-key-here"')
    print()
    print("# Windows (Command Prompt)")
    print('set OPENAI_API_KEY=your-api-key-here')
    print()
    print("# macOS/Linux (Bash)")
    print('export OPENAI_API_KEY="your-api-key-here"')
    print()


def example_use_cases():
    """Example 4: Different use cases and their configurations"""
    print("🎯 Example 4: Use Case Scenarios")
    print("-" * 40)
    
    use_cases = [
        {
            "name": "Creative Writing",
            "config": "config_specialized.yaml",
            "provider": "creative_writer",
            "description": "High temperature for creative content generation",
            "example": "Write a short story about robots learning emotions"
        },
        {
            "name": "Code Generation",
            "config": "config_specialized.yaml", 
            "provider": "code_generator",
            "description": "Low temperature for accurate code generation",
            "example": "Create a Python function to parse JSON files"
        },
        {
            "name": "Data Analysis",
            "config": "config_specialized.yaml",
            "provider": "data_analyst",
            "description": "Analytical reasoning with Claude",
            "example": "Analyze customer satisfaction trends from survey data"
        },
        {
            "name": "Local Development",
            "config": "config_local_development.yaml",
            "provider": "ollama_main",
            "description": "Privacy-focused local processing",
            "example": "Process sensitive customer data without external APIs"
        },
        {
            "name": "Production Workflow",
            "config": "config_production.yaml",
            "provider": "production_primary",
            "description": "Enterprise-grade reliability with fallbacks",
            "example": "Customer service chatbot with 99.9% uptime"
        }
    ]
    
    for use_case in use_cases:
        print(f"📋 {use_case['name']}")
        print(f"   Config: {use_case['config']}")
        print(f"   Provider: {use_case['provider']}")
        print(f"   Purpose: {use_case['description']}")
        print(f"   Example: {use_case['example']}")
        print()


def example_workflow_integration():
    """Example 5: LangGraph workflow integration"""
    print("🔄 Example 5: LangGraph Workflow Integration")
    print("-" * 40)
    
    print("Multi-provider workflow example:")
    print("""
    from llm_interface import LLMManager, LLMGraph
    from langchain_core.messages import HumanMessage
    
    # Setup multiple providers
    manager = LLMManager()
    # ... add providers from config ...
    
    # Create workflow graph
    graph = LLMGraph(manager)
    
    # Define a multi-step workflow
    workflow_definition = {
        "nodes": {
            "draft": {"provider": "creative_writer"},      # Creative first draft
            "review": {"provider": "data_analyst"},        # Analytical review
            "refine": {"provider": "code_generator"},       # Technical refinement
            "final": {"provider": "production_primary"}    # Final polish
        },
        "edges": {
            "draft": "review",
            "review": "refine", 
            "refine": "final"
        },
        "entry_point": "draft"
    }
    
    # Build and run workflow
    custom_graph = graph.build_custom_graph(workflow_definition)
    compiled_graph = graph.compile_graph(custom_graph)
    
    # Execute multi-provider workflow
    messages = [HumanMessage(content="Create a technical blog post about AI")]
    result = compiled_graph.invoke({"messages": messages})
    """)
    print()


def main():
    """Run all examples"""
    print("🚀 LLM Interface Configuration Examples")
    print("=" * 60)
    print()
    
    examples = [
        example_basic_usage,
        example_yaml_configuration,
        example_environment_setup,
        example_use_cases,
        example_workflow_integration
    ]
    
    for example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"❌ Example failed: {e}")
            print()
    
    print("🎉 Examples completed!")
    print()
    print("📚 Next Steps:")
    print("1. Choose a configuration file that matches your use case")
    print("2. Set up the required environment variables") 
    print("3. Copy the example code and adapt it to your needs")
    print("4. Explore the LangGraph integration for complex workflows")
    print("5. Read the README.md for detailed documentation")


if __name__ == "__main__":
    main()