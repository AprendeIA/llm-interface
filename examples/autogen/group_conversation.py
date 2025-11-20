"""
AutoGen group chat example.

Demonstrates multi-agent collaboration using group chat with llm_interface.
"""

import os
from llm_interface import LLMManager
from llm_interface.core.config import LLMConfig, ProviderType
from llm_interface.framework.autogen import AutoGenAdapter


def main():
    """Run a group chat with multiple specialized agents."""
    
    # Initialize LLM Manager
    manager = LLMManager()
    
    # Configure provider
    openai_config = LLMConfig(
        provider=ProviderType.OPENAI,
        model_name="gpt-4",
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0.7,
        max_tokens=1500
    )
    manager.add_provider("gpt4", openai_config)
    
    # Create AutoGen adapter
    adapter = AutoGenAdapter(manager)
    
    # Create specialized agents
    print("Creating specialized agents...")
    
    # Product Manager - defines requirements
    pm_agent = adapter.create_agent(
        name="ProductManager",
        provider_name="gpt4",
        system_message="""You are a Product Manager. Your role is to:
        - Define clear requirements
        - Ensure user needs are met
        - Make final decisions on features
        Be concise and focus on user value.""",
        max_consecutive_auto_reply=3
    )
    
    # Software Engineer - implements solutions
    engineer_agent = adapter.create_agent(
        name="Engineer",
        provider_name="gpt4",
        system_message="""You are a Senior Software Engineer. Your role is to:
        - Propose technical solutions
        - Write clean, efficient code
        - Consider scalability and maintainability
        Provide code examples when relevant.""",
        max_consecutive_auto_reply=3
    )
    
    # QA Tester - validates quality
    qa_agent = adapter.create_agent(
        name="QA_Tester",
        provider_name="gpt4",
        system_message="""You are a QA Tester. Your role is to:
        - Identify potential issues
        - Suggest test cases
        - Ensure quality standards
        Be thorough but constructive.""",
        max_consecutive_auto_reply=3
    )
    
    # User proxy to initiate
    user_proxy = adapter.create_user_proxy(
        name="User",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=0  # Only sends initial message
    )
    
    # Create group chat
    agents = [user_proxy, pm_agent, engineer_agent, qa_agent]
    
    group_chat = adapter.create_group_chat(
        name="dev_team",
        agents=agents,
        max_round=12,
        speaker_selection_method="auto",
        allow_repeat_speaker=False  # Encourage diverse participation
    )
    
    # Create group chat manager
    manager_agent = adapter.create_group_chat_manager(
        group_chat=group_chat,
        provider_name="gpt4",
        system_message="You coordinate the team discussion. Ensure everyone contributes."
    )
    
    print("=" * 80)
    print("AutoGen Group Chat: Development Team")
    print("=" * 80)
    print(f"Agents: {', '.join([a.name for a in agents])}")
    print(f"Model: {openai_config.model_name}")
    print(f"Max rounds: {group_chat.max_round}")
    print()
    
    # Define the task
    task = """
    We need to build a simple Python function that validates email addresses.
    
    Requirements:
    - Validate email format
    - Handle common edge cases
    - Return clear error messages
    
    Please discuss the approach, implementation, and testing strategy.
    """
    
    print("Task:")
    print(task)
    print()
    print("=" * 80)
    print("Starting group discussion...")
    print("=" * 80)
    print()
    
    # Initiate group chat
    result = user_proxy.initiate_chat(
        recipient=manager_agent,
        message=task
    )
    
    print()
    print("=" * 80)
    print("Group Discussion Complete")
    print("=" * 80)
    print(f"Total messages: {len(group_chat.messages)}")


def example_with_code_execution():
    """Example with code execution enabled."""
    
    manager = LLMManager()
    
    openai_config = LLMConfig(
        provider=ProviderType.OPENAI,
        model_name="gpt-4",
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0.5
    )
    manager.add_provider("gpt4", openai_config)
    
    adapter = AutoGenAdapter(manager)
    
    # Coder agent
    coder = adapter.create_agent(
        name="Coder",
        provider_name="gpt4",
        system_message="You write Python code. Provide complete, runnable code.",
        max_consecutive_auto_reply=2
    )
    
    # Executor with code execution enabled
    executor = adapter.create_user_proxy(
        name="Executor",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=2,
        code_execution_config={
            "work_dir": "coding",
            "use_docker": False  # Set True for safer execution
        }
    )
    
    print("=" * 80)
    print("AutoGen Code Execution Example")
    print("=" * 80)
    print()
    
    task = """
    Write a Python function that calculates the factorial of a number.
    Then test it with factorial(5).
    """
    
    print(f"Task: {task}")
    print()
    
    result = executor.initiate_chat(
        recipient=coder,
        message=task
    )
    
    print()
    print("=" * 80)
    print("Code Execution Complete")
    print("=" * 80)


if __name__ == "__main__":
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Set it with: export OPENAI_API_KEY='your-key-here'")
        exit(1)
    
    # Run main example
    main()
    
    print("\n\n")
    
    # Uncomment to run code execution example
    # example_with_code_execution()
