"""
Framework Switching Example.

Demonstrates dynamic switching between AI frameworks with
fallback strategies and monitoring.
"""

from llm_interface import LLMManager, LLMConfig, ProviderType
from llm_interface.src.cross_framework import FrameworkSwitcher
from llm_interface.src.cross_framework.switcher import SwitchStrategy

# Import framework adapters
try:
    from llm_interface.src.framework.langchain import LangChainAdapter
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

try:
    from llm_interface.src.framework.semantic_kernel import SemanticKernelAdapter
    SK_AVAILABLE = True
except ImportError:
    SK_AVAILABLE = False


def setup_manager():
    """Setup LLM manager with providers."""
    manager = LLMManager()
    
    openai_config = LLMConfig(
        provider=ProviderType.OPENAI,
        model_name="gpt-4",
        api_key="your-openai-key"
    )
    manager.add_provider("gpt4", openai_config)
    
    return manager


def example_manual_switching():
    """Example 1: Manual framework switching."""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Manual Framework Switching")
    print("=" * 70)
    
    manager = setup_manager()
    switcher = FrameworkSwitcher(manager)
    
    # Register frameworks
    if LANGCHAIN_AVAILABLE:
        switcher.register("langchain", LangChainAdapter(manager), set_as_current=True)
    
    if SK_AVAILABLE:
        switcher.register("semantic_kernel", SemanticKernelAdapter(manager))
    
    print(f"\nAvailable frameworks: {switcher.list_frameworks()}")
    print(f"Current framework: {switcher.get_current_name()}")
    
    # Execute with current framework
    def task(adapter):
        return f"Executed with {adapter.framework_name}"
    
    result = switcher.execute(task)
    print(f"\nResult: {result}")
    
    # Switch to different framework
    if SK_AVAILABLE and switcher.get_current_name() != "semantic_kernel":
        switcher.switch_to("semantic_kernel", "User preference")
        print(f"\nSwitched to: {switcher.get_current_name()}")
        result = switcher.execute(task)
        print(f"Result: {result}")


def example_fallback_strategy():
    """Example 2: Automatic fallback on error."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Automatic Fallback Strategy")
    print("=" * 70)
    
    manager = setup_manager()
    switcher = FrameworkSwitcher(manager, SwitchStrategy.FALLBACK)
    
    # Register frameworks
    if LANGCHAIN_AVAILABLE:
        switcher.register("langchain", LangChainAdapter(manager))
    
    if SK_AVAILABLE:
        switcher.register("semantic_kernel", SemanticKernelAdapter(manager))
    
    # Set fallback order
    fallback_order = []
    if LANGCHAIN_AVAILABLE:
        fallback_order.append("langchain")
    if SK_AVAILABLE:
        fallback_order.append("semantic_kernel")
    
    if fallback_order:
        switcher.set_fallback_order(fallback_order)
        print(f"\nFallback order: {fallback_order}")
    
    # Task that might fail
    def risky_task(adapter):
        # Simulate framework-specific behavior
        if adapter.framework_name == "langchain":
            raise Exception("Simulated LangChain error")
        return f"Success with {adapter.framework_name}"
    
    try:
        result = switcher.execute(risky_task, fallback_on_error=True)
        print(f"\nResult: {result}")
        print(f"Final framework: {switcher.get_current_name()}")
    except Exception as e:
        print(f"\nAll frameworks failed: {e}")


def example_round_robin():
    """Example 3: Round-robin framework rotation."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Round-Robin Framework Rotation")
    print("=" * 70)
    
    manager = setup_manager()
    switcher = FrameworkSwitcher(manager, SwitchStrategy.ROUND_ROBIN)
    
    # Register frameworks
    if LANGCHAIN_AVAILABLE:
        switcher.register("langchain", LangChainAdapter(manager))
    
    if SK_AVAILABLE:
        switcher.register("semantic_kernel", SemanticKernelAdapter(manager))
    
    def task(adapter):
        return f"Processed by {adapter.framework_name}"
    
    # Execute multiple tasks with rotation
    print("\nExecuting tasks with round-robin rotation:")
    for i in range(5):
        result = switcher.execute(task)
        print(f"  Task {i+1}: {result}")
        if i < 4:  # Don't switch after last task
            switcher.switch_next()


def example_switch_monitoring():
    """Example 4: Monitor framework switches."""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Framework Switch Monitoring")
    print("=" * 70)
    
    manager = setup_manager()
    switcher = FrameworkSwitcher(manager)
    
    # Register frameworks
    if LANGCHAIN_AVAILABLE:
        switcher.register("langchain", LangChainAdapter(manager))
    
    if SK_AVAILABLE:
        switcher.register("semantic_kernel", SemanticKernelAdapter(manager))
    
    # Perform several switches
    if LANGCHAIN_AVAILABLE:
        switcher.switch_to("langchain", "Initial setup")
    
    if SK_AVAILABLE:
        switcher.switch_to("semantic_kernel", "Testing SK")
    
    if LANGCHAIN_AVAILABLE:
        switcher.switch_to("langchain", "Back to LangChain")
    
    # View switch history
    history = switcher.get_switch_history()
    print(f"\nSwitch history ({len(history)} events):")
    for event in history:
        print(f"  {event.from_framework or 'None'} → {event.to_framework}: {event.reason}")
    
    # View framework statistics
    stats = switcher.framework_stats()
    print("\nFramework Statistics:")
    for framework, data in stats.items():
        current_marker = " (CURRENT)" if data.get("is_current") else ""
        print(f"  {framework}{current_marker}:")
        print(f"    Switches to: {data['switches_to']}")
        print(f"    Switches from: {data['switches_from']}")


def example_intelligent_switching():
    """Example 5: Intelligent framework selection."""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Intelligent Framework Selection")
    print("=" * 70)
    
    manager = setup_manager()
    switcher = FrameworkSwitcher(manager)
    
    # Register frameworks
    if LANGCHAIN_AVAILABLE:
        switcher.register("langchain", LangChainAdapter(manager))
    
    if SK_AVAILABLE:
        switcher.register("semantic_kernel", SemanticKernelAdapter(manager))
    
    # Simulate task-based framework selection
    def select_framework_for_task(task_type: str):
        """Select best framework for task type."""
        if task_type == "workflow":
            return "langchain" if LANGCHAIN_AVAILABLE else None
        elif task_type == "plugin":
            return "semantic_kernel" if SK_AVAILABLE else None
        else:
            return switcher.list_frameworks()[0] if switcher.list_frameworks() else None
    
    tasks = [
        ("workflow", "Complex multi-step workflow"),
        ("plugin", "Plugin-based task"),
        ("simple", "Simple prompt")
    ]
    
    print("\nIntelligent task routing:")
    for task_type, description in tasks:
        framework = select_framework_for_task(task_type)
        if framework:
            switcher.switch_to(framework, f"Optimal for {task_type}")
            print(f"  {description} → {framework}")
        else:
            print(f"  {description} → No suitable framework")


def main():
    """Run all switching examples."""
    print("=" * 70)
    print("CROSS-FRAMEWORK SWITCHING EXAMPLES")
    print("=" * 70)
    
    if not LANGCHAIN_AVAILABLE and not SK_AVAILABLE:
        print("\nError: No frameworks available. Install at least one framework:")
        print("  - LangChain: pip install langchain-core langchain-openai")
        print("  - Semantic Kernel: pip install semantic-kernel")
        return
    
    try:
        example_manual_switching()
    except Exception as e:
        print(f"\nExample 1 failed: {e}")
    
    try:
        example_fallback_strategy()
    except Exception as e:
        print(f"\nExample 2 failed: {e}")
    
    try:
        example_round_robin()
    except Exception as e:
        print(f"\nExample 3 failed: {e}")
    
    try:
        example_switch_monitoring()
    except Exception as e:
        print(f"\nExample 4 failed: {e}")
    
    try:
        example_intelligent_switching()
    except Exception as e:
        print(f"\nExample 5 failed: {e}")
    
    print("\n" + "=" * 70)
    print("SWITCHING EXAMPLES COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
