"""
Simple AutoGen chat example.

Demonstrates basic two-agent conversation using llm_interface providers.
"""

import os
from llm_interface import LLMManager
from llm_interface.core.config import LLMConfig, ProviderType
from llm_interface.framework.autogen import AutoGenAdapter


def main():
    """Run a simple two-agent conversation."""
    
    # Initialize LLM Manager with provider
    manager = LLMManager()
    
    # Configure OpenAI provider
    openai_config = LLMConfig(
        provider=ProviderType.OPENAI,
        model_name="gpt-4",
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0.7,
        max_tokens=1000
    )
    manager.add_provider("gpt4", openai_config)
    
    # Create AutoGen adapter
    adapter = AutoGenAdapter(manager)
    
    # Create assistant agent
    assistant = adapter.create_agent(
        name="assistant",
        provider_name="gpt4",
        system_message="You are a helpful AI assistant. Provide clear and concise answers.",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=5
    )
    
    # Create user proxy agent (no LLM)
    user_proxy = adapter.create_user_proxy(
        name="user_proxy",
        human_input_mode="NEVER",  # Set to "ALWAYS" for interactive mode
        max_consecutive_auto_reply=5,
        code_execution_config=False
    )
    
    print("=" * 80)
    print("Simple AutoGen Chat Example")
    print("=" * 80)
    print(f"Assistant using: {openai_config.model_name}")
    print()
    
    # Initiate conversation
    task_message = """
    Explain in 2-3 sentences what AutoGen is and why it's useful for 
    multi-agent systems.
    """
    
    print(f"User: {task_message.strip()}")
    print()
    
    result = adapter.initiate_chat(
        sender=user_proxy,
        recipient=assistant,
        message=task_message,
        max_turns=2
    )
    
    print()
    print("=" * 80)
    print("Conversation Complete")
    print("=" * 80)


def example_with_multiple_providers():
    """Example using different providers for different agents."""
    
    manager = LLMManager()
    
    # Add multiple providers
    openai_config = LLMConfig(
        provider=ProviderType.OPENAI,
        model_name="gpt-4",
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0.7
    )
    manager.add_provider("openai", openai_config)
    
    # Add Anthropic if available
    if os.getenv("ANTHROPIC_API_KEY"):
        anthropic_config = LLMConfig(
            provider=ProviderType.ANTHROPIC,
            model_name="claude-3-sonnet-20240229",
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            temperature=0.7,
            max_tokens=1000
        )
        manager.add_provider("claude", anthropic_config)
    
    adapter = AutoGenAdapter(manager)
    
    # Create agents with different providers
    agent1 = adapter.create_agent(
        name="researcher",
        provider_name="openai",
        system_message="You are a research specialist. Provide factual information.",
        max_consecutive_auto_reply=3
    )
    
    agent2_provider = "claude" if "claude" in manager.list_providers() else "openai"
    agent2 = adapter.create_agent(
        name="writer",
        provider_name=agent2_provider,
        system_message="You are a creative writer. Transform information into engaging content.",
        max_consecutive_auto_reply=3
    )
    
    user_proxy = adapter.create_user_proxy(
        name="user",
        max_consecutive_auto_reply=2
    )
    
    print("=" * 80)
    print("Multi-Provider AutoGen Example")
    print("=" * 80)
    print(f"Researcher using: {openai_config.model_name}")
    print(f"Writer using: {agent2_provider}")
    print()
    
    # First conversation: user -> researcher
    result1 = adapter.initiate_chat(
        sender=user_proxy,
        recipient=agent1,
        message="What are the key benefits of using language models in software development?",
        max_turns=2
    )
    
    print()
    print("=" * 80)
    print()
    
    # Get researcher's response and pass to writer
    if hasattr(result1, 'chat_history'):
        research_content = result1.chat_history[-1]['content']
    else:
        research_content = "Language models improve productivity, code quality, and developer experience."
    
    # Second conversation: user -> writer
    result2 = adapter.initiate_chat(
        sender=user_proxy,
        recipient=agent2,
        message=f"Based on this research, write a brief engaging paragraph:\n\n{research_content}",
        max_turns=2
    )
    
    print()
    print("=" * 80)
    print("Multi-Provider Conversation Complete")
    print("=" * 80)


if __name__ == "__main__":
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Set it with: export OPENAI_API_KEY='your-key-here'")
        exit(1)
    
    # Run basic example
    main()
    
    print("\n\n")
    
    # Run multi-provider example
    example_with_multiple_providers()
