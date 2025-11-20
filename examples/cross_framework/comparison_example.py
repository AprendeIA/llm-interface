"""
Framework Comparison Example.

Demonstrates how to compare the same task across multiple AI frameworks
to evaluate performance, reliability, and suitability.
"""

from llm_interface import LLMManager, LLMConfig, ProviderType
from llm_interface.src.cross_framework import FrameworkComparison, compare_frameworks

# Import framework adapters
try:
    from llm_interface.src.framework.langchain import LangChainAdapter
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("Warning: LangChain not available")

try:
    from llm_interface.src.framework.crewai import CrewAIAdapter
    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False
    print("Warning: CrewAI not available")

try:
    from llm_interface.src.framework.autogen import AutoGenAdapter
    AUTOGEN_AVAILABLE = True
except ImportError:
    AUTOGEN_AVAILABLE = False
    print("Warning: AutoGen not available")

try:
    from llm_interface.src.framework.semantic_kernel import SemanticKernelAdapter
    SK_AVAILABLE = True
except ImportError:
    SK_AVAILABLE = False
    print("Warning: Semantic Kernel not available")


def setup_manager():
    """Setup LLM manager with providers."""
    manager = LLMManager()
    
    # Add OpenAI provider
    openai_config = LLMConfig(
        provider=ProviderType.OPENAI,
        model_name="gpt-4",
        api_key="your-openai-key"  # Replace with actual key or use env var
    )
    manager.add_provider("gpt4", openai_config)
    
    # Add GPT-3.5 for faster responses
    gpt35_config = LLMConfig(
        provider=ProviderType.OPENAI,
        model_name="gpt-3.5-turbo",
        api_key="your-openai-key"
    )
    manager.add_provider("gpt35", gpt35_config)
    
    return manager


def example_basic_comparison():
    """Example 1: Basic framework comparison."""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Basic Framework Comparison")
    print("=" * 70)
    
    manager = setup_manager()
    comparison = FrameworkComparison(manager)
    
    # Register available frameworks
    if LANGCHAIN_AVAILABLE:
        comparison.add_framework("langchain", LangChainAdapter(manager))
    
    if CREWAI_AVAILABLE:
        comparison.add_framework("crewai", CrewAIAdapter(manager))
    
    if AUTOGEN_AVAILABLE:
        comparison.add_framework("autogen", AutoGenAdapter(manager))
    
    if SK_AVAILABLE:
        comparison.add_framework("semantic_kernel", SemanticKernelAdapter(manager))
    
    print(f"\nRegistered frameworks: {comparison.list_frameworks()}")
    
    # Define a simple task
    def summarize_task(adapter):
        """Task: Summarize AI trends in 2024."""
        prompt = "Summarize the top 3 AI trends in 2024 in bullet points."
        
        # Try different invocation methods based on adapter capabilities
        if hasattr(adapter, 'create_model'):
            model = adapter.create_model("gpt35")
            if hasattr(model, 'invoke'):
                return model.invoke(prompt)
        
        return f"Adapter {adapter.framework_name} executed successfully"
    
    # Run comparison
    report = comparison.compare(
        summarize_task,
        "AI Trends Summarization",
        timeout=30.0
    )
    
    print("\n" + report.summary())
    
    # Print fastest framework
    if report.fastest_framework:
        print(f"\n🏆 Winner: {report.fastest_framework}")


def example_simple_prompt_comparison():
    """Example 2: Compare frameworks on a simple prompt."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Simple Prompt Comparison")
    print("=" * 70)
    
    manager = setup_manager()
    comparison = FrameworkComparison(manager)
    
    # Register frameworks
    if LANGCHAIN_AVAILABLE:
        comparison.add_framework("langchain", LangChainAdapter(manager))
    
    if SK_AVAILABLE:
        comparison.add_framework("semantic_kernel", SemanticKernelAdapter(manager))
    
    # Compare using the convenience method
    report = comparison.compare_simple_prompt(
        prompt="What is the capital of France?",
        provider_name="gpt35"
    )
    
    print("\n" + report.summary())


def example_detailed_comparison():
    """Example 3: Detailed comparison with custom metrics."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Detailed Comparison with Metrics")
    print("=" * 70)
    
    manager = setup_manager()
    comparison = FrameworkComparison(manager)
    
    # Register frameworks
    if LANGCHAIN_AVAILABLE:
        langchain = LangChainAdapter(manager)
        comparison.add_framework("langchain", langchain)
        print(f"\nLangChain Info: {comparison.get_framework_info('langchain')}")
    
    if SK_AVAILABLE:
        sk = SemanticKernelAdapter(manager)
        comparison.add_framework("semantic_kernel", sk)
        print(f"\nSemantic Kernel Info: {comparison.get_framework_info('semantic_kernel')}")
    
    # Complex task with multiple steps
    def complex_task(adapter):
        """Multi-step analysis task."""
        prompts = [
            "List 3 programming languages",
            "Which one is best for AI?",
            "Why?"
        ]
        
        results = []
        for prompt in prompts:
            if hasattr(adapter, 'create_model'):
                model = adapter.create_model("gpt35")
                if hasattr(model, 'invoke'):
                    result = model.invoke(prompt)
                    results.append(result)
        
        return f"Completed {len(results)} steps"
    
    report = comparison.compare(
        complex_task,
        "Multi-step Programming Analysis",
        timeout=60.0
    )
    
    print("\n" + report.summary())
    
    # Detailed analysis
    print("\n" + "-" * 70)
    print("DETAILED ANALYSIS:")
    print("-" * 70)
    
    for result in report.results:
        print(f"\n{result.framework_name}:")
        print(f"  Success: {result.success}")
        print(f"  Time: {result.execution_time:.3f}s")
        if result.metadata:
            print(f"  Metadata: {result.metadata}")
        if result.error:
            print(f"  Error: {result.error}")


def example_selective_comparison():
    """Example 4: Compare only specific frameworks."""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Selective Framework Comparison")
    print("=" * 70)
    
    manager = setup_manager()
    
    # Use the convenience function
    adapters = {}
    
    if LANGCHAIN_AVAILABLE:
        adapters["langchain"] = LangChainAdapter(manager)
    
    if SK_AVAILABLE:
        adapters["semantic_kernel"] = SemanticKernelAdapter(manager)
    
    if not adapters:
        print("No frameworks available for comparison")
        return
    
    def greeting_task(adapter):
        """Simple greeting task."""
        if hasattr(adapter, 'create_model'):
            model = adapter.create_model("gpt35")
            if hasattr(model, 'invoke'):
                return model.invoke("Say hello in 3 different languages")
        return "Hello, Bonjour, Hola"
    
    report = compare_frameworks(
        manager,
        adapters,
        greeting_task,
        "Multilingual Greeting"
    )
    
    print("\n" + report.summary())
    
    # Print success rates
    print("\n" + "-" * 70)
    print("SUCCESS RATES:")
    for framework, success in report.success_rate.items():
        status = "✅" if success else "❌"
        print(f"  {status} {framework}")


def main():
    """Run all comparison examples."""
    print("=" * 70)
    print("CROSS-FRAMEWORK COMPARISON EXAMPLES")
    print("=" * 70)
    
    try:
        example_basic_comparison()
    except Exception as e:
        print(f"\nExample 1 failed: {e}")
    
    try:
        example_simple_prompt_comparison()
    except Exception as e:
        print(f"\nExample 2 failed: {e}")
    
    try:
        example_detailed_comparison()
    except Exception as e:
        print(f"\nExample 3 failed: {e}")
    
    try:
        example_selective_comparison()
    except Exception as e:
        print(f"\nExample 4 failed: {e}")
    
    print("\n" + "=" * 70)
    print("COMPARISON EXAMPLES COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
