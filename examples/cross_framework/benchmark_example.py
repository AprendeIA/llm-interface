"""
Framework Benchmarking Example.

Demonstrates performance benchmarking and comparison across
different AI frameworks.
"""

from llm_interface import LLMManager, LLMConfig, ProviderType
from llm_interface.src.cross_framework import FrameworkBenchmark

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
    
    # Add multiple providers for comparison
    gpt4_config = LLMConfig(
        provider=ProviderType.OPENAI,
        model_name="gpt-4",
        api_key="your-openai-key"
    )
    manager.add_provider("gpt4", gpt4_config)
    
    gpt35_config = LLMConfig(
        provider=ProviderType.OPENAI,
        model_name="gpt-3.5-turbo",
        api_key="your-openai-key"
    )
    manager.add_provider("gpt35", gpt35_config)
    
    return manager


def example_single_framework_benchmark():
    """Example 1: Benchmark a single framework."""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Single Framework Benchmark")
    print("=" * 70)
    
    manager = setup_manager()
    benchmark = FrameworkBenchmark(manager)
    
    # Register frameworks
    if LANGCHAIN_AVAILABLE:
        benchmark.add_framework("langchain", LangChainAdapter(manager))
    else:
        print("\nLangChain not available, skipping example")
        return
    
    # Define benchmark task
    def simple_task(adapter):
        """Simple invocation task."""
        if hasattr(adapter, 'create_model'):
            model = adapter.create_model("gpt35")
            if hasattr(model, 'invoke'):
                return model.invoke("Say hello")
        return "Hello"
    
    # Run benchmark
    result = benchmark.benchmark_single(
        "langchain",
        simple_task,
        "Simple Greeting Task",
        runs=5,
        warmup_runs=1
    )
    
    print("\n" + result.summary())


def example_comparative_benchmark():
    """Example 2: Compare multiple frameworks."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Comparative Framework Benchmark")
    print("=" * 70)
    
    manager = setup_manager()
    benchmark = FrameworkBenchmark(manager)
    
    # Register all available frameworks
    if LANGCHAIN_AVAILABLE:
        benchmark.add_framework("langchain", LangChainAdapter(manager))
    
    if SK_AVAILABLE:
        benchmark.add_framework("semantic_kernel", SemanticKernelAdapter(manager))
    
    if not benchmark.list_frameworks():
        print("\nNo frameworks available for benchmarking")
        return
    
    print(f"\nBenchmarking frameworks: {benchmark.list_frameworks()}")
    
    # Define benchmark task
    def summarization_task(adapter):
        """Summarization task."""
        prompt = "Summarize the benefits of AI in 3 bullet points."
        
        if hasattr(adapter, 'create_model'):
            model = adapter.create_model("gpt35")
            if hasattr(model, 'invoke'):
                return model.invoke(prompt)
        
        return "AI benefits summarized"
    
    # Run comparative benchmark
    report = benchmark.run_comparative(
        summarization_task,
        "AI Summarization Task",
        runs=3,
        warmup_runs=1
    )
    
    print("\n" + report.summary())


def example_quick_compare():
    """Example 3: Quick comparison with minimal runs."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Quick Framework Comparison")
    print("=" * 70)
    
    manager = setup_manager()
    benchmark = FrameworkBenchmark(manager)
    
    # Register frameworks
    if LANGCHAIN_AVAILABLE:
        benchmark.add_framework("langchain", LangChainAdapter(manager))
    
    if SK_AVAILABLE:
        benchmark.add_framework("semantic_kernel", SemanticKernelAdapter(manager))
    
    if not benchmark.list_frameworks():
        print("\nNo frameworks available")
        return
    
    # Quick compare with 3 runs (no warmup)
    report = benchmark.quick_compare(
        prompt="What is machine learning?",
        provider_name="gpt35",
        runs=3
    )
    
    print("\n" + report.summary())


def example_performance_analysis():
    """Example 4: Detailed performance analysis."""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Detailed Performance Analysis")
    print("=" * 70)
    
    manager = setup_manager()
    benchmark = FrameworkBenchmark(manager)
    
    # Register frameworks
    if LANGCHAIN_AVAILABLE:
        benchmark.add_framework("langchain", LangChainAdapter(manager))
    
    if SK_AVAILABLE:
        benchmark.add_framework("semantic_kernel", SemanticKernelAdapter(manager))
    
    if not benchmark.list_frameworks():
        print("\nNo frameworks available")
        return
    
    # Complex task with multiple operations
    def complex_task(adapter):
        """Complex multi-step task."""
        steps = [
            "Define AI",
            "List 3 applications",
            "Explain one benefit"
        ]
        
        results = []
        for step in steps:
            if hasattr(adapter, 'create_model'):
                model = adapter.create_model("gpt35")
                if hasattr(model, 'invoke'):
                    result = model.invoke(step)
                    results.append(result)
        
        return f"Completed {len(results)}/{len(steps)} steps"
    
    # Run with more iterations for statistical significance
    report = benchmark.run_comparative(
        complex_task,
        "Multi-step AI Analysis",
        runs=5,
        warmup_runs=2
    )
    
    print("\n" + report.summary())
    
    # Additional analysis
    print("\n" + "-" * 70)
    print("PERFORMANCE METRICS:")
    print("-" * 70)
    
    for result in report.results:
        print(f"\n{result.framework_name}:")
        print(f"  Runs: {result.runs}")
        print(f"  Success Rate: {result.success_rate:.1f}%")
        print(f"  Mean Time: {result.mean_time:.3f}s")
        print(f"  Std Dev: {result.std_dev_time:.3f}s")
        print(f"  Range: {result.min_time:.3f}s - {result.max_time:.3f}s")
        
        # Calculate coefficient of variation (consistency metric)
        if result.mean_time > 0:
            cv = (result.std_dev_time / result.mean_time) * 100
            print(f"  Consistency (CV): {cv:.1f}%")


def example_provider_comparison():
    """Example 5: Compare same framework with different providers."""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Provider Comparison (Same Framework)")
    print("=" * 70)
    
    if not LANGCHAIN_AVAILABLE:
        print("\nLangChain not available, skipping example")
        return
    
    manager = setup_manager()
    benchmark = FrameworkBenchmark(manager)
    
    # Register same framework instance
    adapter = LangChainAdapter(manager)
    benchmark.add_framework("langchain", adapter)
    
    # Benchmark with different providers
    def task_gpt4(adapter):
        if hasattr(adapter, 'create_model'):
            model = adapter.create_model("gpt4")
            if hasattr(model, 'invoke'):
                return model.invoke("Explain quantum computing briefly")
        return "Explained"
    
    def task_gpt35(adapter):
        if hasattr(adapter, 'create_model'):
            model = adapter.create_model("gpt35")
            if hasattr(model, 'invoke'):
                return model.invoke("Explain quantum computing briefly")
        return "Explained"
    
    print("\nBenchmarking LangChain with GPT-4:")
    result_gpt4 = benchmark.benchmark_single(
        "langchain",
        task_gpt4,
        "Quantum Computing (GPT-4)",
        runs=3
    )
    print(result_gpt4.summary())
    
    print("\n" + "-" * 70)
    print("\nBenchmarking LangChain with GPT-3.5-Turbo:")
    result_gpt35 = benchmark.benchmark_single(
        "langchain",
        task_gpt35,
        "Quantum Computing (GPT-3.5-Turbo)",
        runs=3
    )
    print(result_gpt35.summary())
    
    # Compare results
    print("\n" + "-" * 70)
    print("PROVIDER COMPARISON:")
    print(f"  GPT-4 Mean: {result_gpt4.mean_time:.3f}s")
    print(f"  GPT-3.5 Mean: {result_gpt35.mean_time:.3f}s")
    
    if result_gpt35.mean_time > 0:
        speedup = result_gpt4.mean_time / result_gpt35.mean_time
        faster = "GPT-3.5" if speedup > 1 else "GPT-4"
        print(f"  Faster: {faster} ({abs(speedup):.2f}x)")


def main():
    """Run all benchmarking examples."""
    print("=" * 70)
    print("CROSS-FRAMEWORK BENCHMARKING EXAMPLES")
    print("=" * 70)
    
    if not LANGCHAIN_AVAILABLE and not SK_AVAILABLE:
        print("\nError: No frameworks available. Install at least one framework:")
        print("  - LangChain: pip install langchain-core langchain-openai")
        print("  - Semantic Kernel: pip install semantic-kernel")
        return
    
    try:
        example_single_framework_benchmark()
    except Exception as e:
        print(f"\nExample 1 failed: {e}")
    
    try:
        example_comparative_benchmark()
    except Exception as e:
        print(f"\nExample 2 failed: {e}")
    
    try:
        example_quick_compare()
    except Exception as e:
        print(f"\nExample 3 failed: {e}")
    
    try:
        example_performance_analysis()
    except Exception as e:
        print(f"\nExample 4 failed: {e}")
    
    try:
        example_provider_comparison()
    except Exception as e:
        print(f"\nExample 5 failed: {e}")
    
    print("\n" + "=" * 70)
    print("BENCHMARKING EXAMPLES COMPLETE")
    print("=" * 70)
    print("\n💡 TIP: Run with actual API keys to see real performance data!")


if __name__ == "__main__":
    main()
