"""
Hybrid Workflow Example.

Demonstrates combining multiple AI frameworks in a single workflow,
leveraging the strengths of each framework for different tasks.
"""

from llm_interface import LLMManager, LLMConfig, ProviderType
from llm_interface.src.cross_framework import FrameworkSwitcher, FrameworkComparison

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


def example_research_writing_workflow():
    """
    Example 1: Research → Writing → Formatting Workflow.
    
    Uses different frameworks for each stage:
    - LangChain for research (graph-based workflows)
    - Semantic Kernel for writing (plugin-based composition)
    - Back to LangChain for formatting (chains)
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Research → Writing → Formatting Workflow")
    print("=" * 70)
    
    manager = setup_manager()
    switcher = FrameworkSwitcher(manager)
    
    # Register frameworks
    if LANGCHAIN_AVAILABLE:
        switcher.register("langchain", LangChainAdapter(manager))
    
    if SK_AVAILABLE:
        switcher.register("semantic_kernel", SemanticKernelAdapter(manager))
    
    topic = "Artificial Intelligence in Healthcare"
    
    # Stage 1: Research with LangChain (graph-based)
    print(f"\n📊 Stage 1: Research on '{topic}'")
    if LANGCHAIN_AVAILABLE:
        switcher.switch_to("langchain", "Best for research workflows")
        
        def research_task(adapter):
            prompt = f"List 5 key applications of {topic}"
            if hasattr(adapter, 'create_model'):
                model = adapter.create_model("gpt4")
                if hasattr(model, 'invoke'):
                    return model.invoke(prompt)
            return "Research completed"
        
        research_result = switcher.execute(research_task)
        print(f"Research output: {research_result[:100]}...")
    
    # Stage 2: Writing with Semantic Kernel (plugin-based)
    print(f"\n✍️ Stage 2: Writing content")
    if SK_AVAILABLE:
        switcher.switch_to("semantic_kernel", "Best for plugin-based writing")
        
        def writing_task(adapter):
            prompt = f"Write a 2-paragraph introduction about {topic}"
            if hasattr(adapter, 'create_model'):
                model = adapter.create_model("gpt4")
                if hasattr(model, 'invoke'):
                    return model.invoke(prompt)
            return "Article written"
        
        writing_result = switcher.execute(writing_task)
        print(f"Writing output: {writing_result[:100]}...")
    
    # Stage 3: Formatting with LangChain (chains)
    print(f"\n📝 Stage 3: Formatting output")
    if LANGCHAIN_AVAILABLE:
        switcher.switch_to("langchain", "Best for formatting chains")
        
        def format_task(adapter):
            prompt = "Format this as a professional report abstract (mock task)"
            if hasattr(adapter, 'create_model'):
                model = adapter.create_model("gpt4")
                if hasattr(model, 'invoke'):
                    return model.invoke(prompt)
            return "Report formatted"
        
        format_result = switcher.execute(format_task)
        print(f"Formatting output: {format_result[:100]}...")
    
    # Show workflow summary
    print("\n" + "-" * 70)
    print("WORKFLOW SUMMARY:")
    history = switcher.get_switch_history()
    for i, event in enumerate(history, 1):
        print(f"  {i}. {event.to_framework}: {event.reason}")


def example_parallel_processing():
    """
    Example 2: Parallel Processing Across Frameworks.
    
    Distributes tasks across multiple frameworks simultaneously
    for faster processing.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Parallel Processing Across Frameworks")
    print("=" * 70)
    
    manager = setup_manager()
    comparison = FrameworkComparison(manager)
    
    # Register frameworks
    if LANGCHAIN_AVAILABLE:
        comparison.add_framework("langchain", LangChainAdapter(manager))
    
    if SK_AVAILABLE:
        comparison.add_framework("semantic_kernel", SemanticKernelAdapter(manager))
    
    # Define multiple independent tasks
    tasks = [
        ("Summarize AI ethics", "Ethics Summary"),
        ("Explain machine learning", "ML Explanation"),
        ("List AI tools", "Tools List")
    ]
    
    print("\nProcessing tasks in parallel across frameworks:")
    
    for prompt, description in tasks:
        def task(adapter):
            if hasattr(adapter, 'create_model'):
                model = adapter.create_model("gpt4")
                if hasattr(model, 'invoke'):
                    return model.invoke(prompt)
            return f"Processed: {prompt}"
        
        report = comparison.compare(task, description, timeout=30)
        fastest = report.fastest_framework
        print(f"  ✓ {description}: {fastest} was fastest")


def example_adaptive_workflow():
    """
    Example 3: Adaptive Workflow with Dynamic Framework Selection.
    
    Automatically selects the best framework for each task type.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Adaptive Workflow")
    print("=" * 70)
    
    manager = setup_manager()
    switcher = FrameworkSwitcher(manager)
    
    # Register frameworks
    if LANGCHAIN_AVAILABLE:
        switcher.register("langchain", LangChainAdapter(manager))
    
    if SK_AVAILABLE:
        switcher.register("semantic_kernel", SemanticKernelAdapter(manager))
    
    # Set up fallback
    fallback_order = []
    if LANGCHAIN_AVAILABLE:
        fallback_order.append("langchain")
    if SK_AVAILABLE:
        fallback_order.append("semantic_kernel")
    
    if fallback_order:
        switcher.set_fallback_order(fallback_order)
    
    # Task selector
    def select_framework(task_type: str) -> str:
        """Select optimal framework for task type."""
        framework_strengths = {
            "chain": "langchain",
            "graph": "langchain",
            "plugin": "semantic_kernel",
            "skill": "semantic_kernel",
        }
        
        selected = framework_strengths.get(task_type)
        
        # Fallback to first available if preferred not available
        if not selected or selected not in switcher.list_frameworks():
            selected = switcher.list_frameworks()[0] if switcher.list_frameworks() else None
        
        return selected
    
    # Workflow with different task types
    workflow_steps = [
        ("chain", "Create sequential analysis chain"),
        ("plugin", "Apply text processing plugins"),
        ("graph", "Build workflow graph"),
        ("skill", "Execute specialized skill")
    ]
    
    print("\nAdaptive task routing:")
    for task_type, description in workflow_steps:
        framework = select_framework(task_type)
        if framework:
            switcher.switch_to(framework, f"Optimal for {task_type}")
            print(f"  {description} → {framework}")
            
            # Execute task
            def task(adapter):
                return f"{description} completed"
            
            result = switcher.execute(task, fallback_on_error=True)
            print(f"    Result: {result}")
        else:
            print(f"  {description} → No suitable framework")


def example_quality_vs_speed_tradeoff():
    """
    Example 4: Quality vs Speed Trade-off.
    
    Uses different frameworks based on priority (quality or speed).
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Quality vs Speed Trade-off")
    print("=" * 70)
    
    manager = setup_manager()
    comparison = FrameworkComparison(manager)
    
    # Register frameworks
    if LANGCHAIN_AVAILABLE:
        comparison.add_framework("langchain", LangChainAdapter(manager))
    
    if SK_AVAILABLE:
        comparison.add_framework("semantic_kernel", SemanticKernelAdapter(manager))
    
    # Quick task (prioritize speed)
    print("\n⚡ Quick Task (prioritize speed):")
    
    def quick_task(adapter):
        if hasattr(adapter, 'create_model'):
            model = adapter.create_model("gpt4")
            if hasattr(model, 'invoke'):
                return model.invoke("What is AI?")
        return "AI explained"
    
    quick_report = comparison.compare(quick_task, "Quick AI Definition")
    fastest = quick_report.fastest_framework
    print(f"  Selected framework: {fastest} (fastest)")
    
    # Complex task (prioritize quality)
    print("\n🎯 Complex Task (prioritize quality):")
    
    def complex_task(adapter):
        if hasattr(adapter, 'create_model'):
            model = adapter.create_model("gpt4")
            if hasattr(model, 'invoke'):
                return model.invoke("Write a detailed analysis of AI impact")
        return "Analysis completed"
    
    complex_report = comparison.compare(complex_task, "Detailed AI Analysis")
    most_reliable = complex_report.most_reliable
    print(f"  Selected framework: {most_reliable} (most reliable)")


def main():
    """Run all hybrid workflow examples."""
    print("=" * 70)
    print("HYBRID WORKFLOW EXAMPLES")
    print("=" * 70)
    
    if not LANGCHAIN_AVAILABLE and not SK_AVAILABLE:
        print("\nError: No frameworks available. Install at least one framework:")
        print("  - LangChain: pip install langchain-core langchain-openai")
        print("  - Semantic Kernel: pip install semantic-kernel")
        return
    
    try:
        example_research_writing_workflow()
    except Exception as e:
        print(f"\nExample 1 failed: {e}")
    
    try:
        example_parallel_processing()
    except Exception as e:
        print(f"\nExample 2 failed: {e}")
    
    try:
        example_adaptive_workflow()
    except Exception as e:
        print(f"\nExample 3 failed: {e}")
    
    try:
        example_quality_vs_speed_tradeoff()
    except Exception as e:
        print(f"\nExample 4 failed: {e}")
    
    print("\n" + "=" * 70)
    print("HYBRID WORKFLOW EXAMPLES COMPLETE")
    print("=" * 70)
    print("\n💡 KEY TAKEAWAY: Combine frameworks to leverage their unique strengths!")


if __name__ == "__main__":
    main()
