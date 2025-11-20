"""CrewAI Adapter Usage Examples.

Comprehensive examples demonstrating the CrewAI adapter for multi-agent
workflows with the unified LLM interface.
"""

# Example 1: Basic Agent and Task Creation
# ==========================================

def example_basic_agent_creation():
    """Example: Create a basic agent."""
    from llm_interface.src.manager import LLMManager
    from llm_interface.src.core.config import LLMConfig, ProviderType
    from llm_interface.src.framework.crewai import CrewAIAdapter
    
    # Setup manager with providers
    manager = LLMManager()
    config = LLMConfig(
        provider=ProviderType.OPENAI,
        model_name="gpt-4",
        api_key="your-api-key"
    )
    manager.add_provider("openai", config)
    
    # Create adapter
    adapter = CrewAIAdapter(manager)
    
    # Create agent
    agent = adapter.create_agent(
        name="researcher",
        role="Research Analyst",
        goal="Find and analyze information",
        backstory="Expert researcher with attention to detail"
    )
    
    print(f"Created agent: {adapter.get_agent('researcher')}")
    print(f"Agents: {adapter.list_agents()}")


# Example 2: Create Tasks for Agents
# ===================================

def example_task_creation():
    """Example: Create tasks for agents."""
    from llm_interface.src.manager import LLMManager
    from llm_interface.src.core.config import LLMConfig, ProviderType
    from llm_interface.src.framework.crewai import CrewAIAdapter
    
    manager = LLMManager()
    config = LLMConfig(
        provider=ProviderType.OPENAI,
        model_name="gpt-4",
        api_key="your-api-key"
    )
    manager.add_provider("openai", config)
    
    adapter = CrewAIAdapter(manager)
    
    # Create agent
    adapter.create_agent(
        name="researcher",
        role="Research Analyst",
        goal="Find information",
        backstory="Expert researcher"
    )
    
    # Create task
    task = adapter.create_task(
        name="research_ai",
        description="Research the latest developments in AI",
        expected_output="Comprehensive report on AI advancements",
        agent_name="researcher"
    )
    
    print(f"Created task: {task}")
    print(f"Tasks: {adapter.list_tasks()}")


# Example 3: Create and Execute a Crew
# =====================================

def example_crew_execution():
    """Example: Create and execute a multi-agent crew."""
    from llm_interface.src.manager import LLMManager
    from llm_interface.src.core.config import LLMConfig, ProviderType
    from llm_interface.src.framework.crewai import CrewAIAdapter
    
    manager = LLMManager()
    config = LLMConfig(
        provider=ProviderType.OPENAI,
        model_name="gpt-4",
        api_key="your-api-key"
    )
    manager.add_provider("openai", config)
    
    adapter = CrewAIAdapter(manager)
    
    # Create agents
    adapter.create_agent(
        name="researcher",
        role="Research Analyst",
        goal="Find information",
        backstory="Expert researcher"
    )
    
    adapter.create_agent(
        name="writer",
        role="Technical Writer",
        goal="Write content",
        backstory="Skilled writer"
    )
    
    # Create tasks
    adapter.create_task(
        name="research",
        description="Research AI",
        expected_output="Research findings",
        agent_name="researcher"
    )
    
    adapter.create_task(
        name="write",
        description="Write article",
        expected_output="Published article",
        agent_name="writer"
    )
    
    # Create crew
    crew = adapter.create_crew(
        name="content_crew",
        agents=["researcher", "writer"],
        tasks=["research", "write"],
        process="sequential"
    )
    
    # Execute crew
    result = adapter.kickoff_crew(
        "content_crew",
        inputs={"topic": "Artificial Intelligence"}
    )
    
    print(f"Crew execution result: {result}")


# Example 4: Task Dependencies
# =============================

def example_task_dependencies():
    """Example: Create tasks with dependencies."""
    from llm_interface.src.manager import LLMManager
    from llm_interface.src.core.config import LLMConfig, ProviderType
    from llm_interface.src.framework.crewai import CrewAIAdapter
    
    manager = LLMManager()
    config = LLMConfig(
        provider=ProviderType.OPENAI,
        model_name="gpt-4",
        api_key="your-api-key"
    )
    manager.add_provider("openai", config)
    
    adapter = CrewAIAdapter(manager)
    
    # Create agents
    adapter.create_agent(
        name="researcher",
        role="Researcher",
        goal="Research",
        backstory="Expert"
    )
    
    adapter.create_agent(
        name="analyzer",
        role="Analyst",
        goal="Analyze",
        backstory="Expert analyst"
    )
    
    # Create tasks with dependencies
    task1 = adapter.create_task(
        name="collect_data",
        description="Collect data",
        expected_output="Data",
        agent_name="researcher"
    )
    
    # Task2 depends on task1
    task2 = adapter.create_task(
        name="analyze_data",
        description="Analyze collected data",
        expected_output="Analysis",
        agent_name="analyzer",
        context=[task1]  # Dependency on task1
    )
    
    print(f"Created dependent tasks: {[task1, task2]}")


# Example 5: Programmatic Crew Configuration
# ===========================================

def example_programmatic_configuration():
    """Example: Build configuration programmatically."""
    from llm_interface.src.framework.crewai import CrewAIConfigBuilder
    
    # Build configuration
    builder = (CrewAIConfigBuilder()
        .add_agent(
            name="researcher",
            role="Research Analyst",
            goal="Find information",
            backstory="Expert researcher",
            provider="openai",
            memory=True
        )
        .add_agent(
            name="writer",
            role="Writer",
            goal="Write content",
            backstory="Skilled writer",
            provider="openai"
        )
        .add_task(
            name="research_task",
            description="Research AI",
            expected_output="Report",
            agent_name="researcher"
        )
        .add_task(
            name="write_task",
            description="Write article",
            expected_output="Article",
            agent_name="writer"
        )
        .add_crew(
            name="content_crew",
            agents=["researcher", "writer"],
            tasks=["research_task", "write_task"],
            process="sequential"
        )
    )
    
    config = builder.build()
    print(f"Configuration: {config}")
    return config


# Example 6: Multi-Provider Setup
# ================================

def example_multi_provider():
    """Example: Use multiple LLM providers."""
    from llm_interface.src.manager import LLMManager
    from llm_interface.src.core.config import LLMConfig, ProviderType
    from llm_interface.src.framework.crewai import CrewAIAdapter
    
    manager = LLMManager()
    
    # Add multiple providers
    openai_config = LLMConfig(
        provider=ProviderType.OPENAI,
        model_name="gpt-4",
        api_key="your-openai-key"
    )
    manager.add_provider("openai", openai_config)
    
    anthropic_config = LLMConfig(
        provider=ProviderType.ANTHROPIC,
        model_name="claude-3-opus",
        api_key="your-anthropic-key"
    )
    manager.add_provider("anthropic", anthropic_config)
    
    adapter = CrewAIAdapter(manager)
    
    # Create agents with different providers
    adapter.create_agent(
        name="researcher",
        role="Researcher",
        goal="Research",
        backstory="Expert",
        provider_name="openai"
    )
    
    adapter.create_agent(
        name="analyst",
        role="Analyst",
        goal="Analyze",
        backstory="Expert analyst",
        provider_name="anthropic"
    )
    
    print(f"Available providers: {adapter.list_providers()}")


# Example 7: Agent with Tools
# =============================

def example_agent_with_tools():
    """Example: Create agent with tools."""
    from llm_interface.src.manager import LLMManager
    from llm_interface.src.core.config import LLMConfig, ProviderType
    from llm_interface.src.framework.crewai import CrewAIAdapter
    
    manager = LLMManager()
    config = LLMConfig(
        provider=ProviderType.OPENAI,
        model_name="gpt-4",
        api_key="your-api-key"
    )
    manager.add_provider("openai", config)
    
    adapter = CrewAIAdapter(manager)
    
    # Create mock tools
    tools = []  # In practice, import actual CrewAI tools
    
    # Create agent with tools
    agent = adapter.create_agent(
        name="researcher",
        role="Research Analyst",
        goal="Find information",
        backstory="Expert researcher",
        tools=tools
    )
    
    print(f"Created agent with tools: {agent}")


# Example 8: Hierarchical Crew Process
# =====================================

def example_hierarchical_crew():
    """Example: Create crew with hierarchical process."""
    from llm_interface.src.manager import LLMManager
    from llm_interface.src.core.config import LLMConfig, ProviderType
    from llm_interface.src.framework.crewai import CrewAIAdapter
    
    manager = LLMManager()
    config = LLMConfig(
        provider=ProviderType.OPENAI,
        model_name="gpt-4",
        api_key="your-api-key"
    )
    manager.add_provider("openai", config)
    
    adapter = CrewAIAdapter(manager)
    
    # Create agents
    adapter.create_agent(
        name="researcher",
        role="Researcher",
        goal="Research",
        backstory="Expert"
    )
    
    adapter.create_agent(
        name="writer",
        role="Writer",
        goal="Write",
        backstory="Expert writer"
    )
    
    # Create tasks
    adapter.create_task(
        name="research",
        description="Research",
        expected_output="Report",
        agent_name="researcher"
    )
    
    adapter.create_task(
        name="write",
        description="Write",
        expected_output="Article",
        agent_name="writer"
    )
    
    # Create hierarchical crew
    crew = adapter.create_crew(
        name="hierarchical_crew",
        agents=["researcher", "writer"],
        tasks=["research", "write"],
        process="hierarchical"  # Uses manager agent for delegation
    )
    
    print(f"Created hierarchical crew: {crew}")


# Example 9: Save and Load Configuration
# =======================================

def example_save_load_configuration():
    """Example: Save and load configuration."""
    import tempfile
    from llm_interface.src.framework.crewai import (
        CrewAIConfigBuilder,
        CrewAIConfigLoader,
    )
    
    # Build configuration
    builder = (CrewAIConfigBuilder()
        .add_agent("researcher", "Researcher", "Research", "Expert")
        .add_task("task1", "Task 1", "Output 1", "researcher")
        .add_crew("crew1", ["researcher"], ["task1"])
    )
    
    config = builder.build()
    
    # Save to file
    with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as f:
        filepath = f.name
    
    CrewAIConfigLoader.save_to_file(config, filepath)
    print(f"Saved configuration to: {filepath}")
    
    # Load from file
    loaded_config = CrewAIConfigLoader.load_from_file(filepath)
    print(f"Loaded configuration: {loaded_config}")
    
    # Convert to YAML string
    yaml_str = CrewAIConfigLoader.to_yaml(config)
    print(f"YAML representation:\n{yaml_str}")


# Example 10: Configuration Validation
# =====================================

def example_configuration_validation():
    """Example: Validate configurations."""
    from llm_interface.src.manager import LLMManager
    from llm_interface.src.core.config import LLMConfig, ProviderType
    from llm_interface.src.framework.crewai import CrewAIAdapter
    
    manager = LLMManager()
    config = LLMConfig(
        provider=ProviderType.OPENAI,
        model_name="gpt-4",
        api_key="your-api-key"
    )
    manager.add_provider("openai", config)
    
    adapter = CrewAIAdapter(manager)
    
    # Valid configurations
    valid_config = {
        'default_provider': 'openai',
        'verbose': True,
        'memory': True,
    }
    
    assert adapter.validate_config(valid_config) is True
    
    # Invalid configuration
    invalid_config = {
        'invalid_key': 'value',
    }
    
    assert adapter.validate_config(invalid_config) is False
    
    print("Configuration validation examples completed")


# Main execution
# ==============

if __name__ == "__main__":
    print("CrewAI Adapter Examples")
    print("=" * 50)
    
    # Note: These examples require API keys and CrewAI to be installed
    # Uncomment to run:
    
    # example_basic_agent_creation()
    # example_task_creation()
    # example_crew_execution()
    # example_task_dependencies()
    example_programmatic_configuration()
    # example_multi_provider()
    # example_agent_with_tools()
    # example_hierarchical_crew()
    example_save_load_configuration()
    example_configuration_validation()
    
    print("\nExamples completed!")
