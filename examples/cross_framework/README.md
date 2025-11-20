# Cross-Framework Examples

Examples demonstrating cross-framework features of the LLM Interface library.

## Overview

These examples show how to:
- **Compare** different AI frameworks for the same task
- **Switch** between frameworks dynamically
- **Benchmark** framework performance
- **Create hybrid workflows** combining multiple frameworks

## Prerequisites

```bash
# Install base library
pip install -e .

# Install framework extras (install what you need)
pip install langchain-core langchain-openai  # For LangChain
pip install semantic-kernel                  # For Semantic Kernel
pip install crewai                           # For CrewAI
pip install pyautogen                        # For AutoGen
```

## Examples

### 1. Framework Comparison (`comparison_example.py`)

Compare how different frameworks handle the same task.

**Features:**
- Basic framework comparison
- Simple prompt comparison
- Detailed comparison with metrics
- Selective framework comparison

**Usage:**
```python
from llm_interface.src.cross_framework import FrameworkComparison

comparison = FrameworkComparison(manager)
comparison.add_framework("langchain", langchain_adapter)
comparison.add_framework("semantic_kernel", sk_adapter)

report = comparison.compare(task, "Task Description")
print(report.summary())
```

**Run:**
```bash
python examples/cross_framework/comparison_example.py
```

### 2. Framework Switching (`switching_example.py`)

Dynamically switch between frameworks with fallback strategies.

**Features:**
- Manual framework switching
- Automatic fallback on error
- Round-robin rotation
- Switch monitoring and statistics

**Usage:**
```python
from llm_interface.src.cross_framework import FrameworkSwitcher

switcher = FrameworkSwitcher(manager)
switcher.register("langchain", langchain_adapter)
switcher.register("semantic_kernel", sk_adapter)

# Set fallback order
switcher.set_fallback_order(["langchain", "semantic_kernel"])

# Execute with auto-fallback
result = switcher.execute(task, fallback_on_error=True)
```

**Run:**
```bash
python examples/cross_framework/switching_example.py
```

### 3. Performance Benchmarking (`benchmark_example.py`)

Measure and compare performance across frameworks.

**Features:**
- Single framework benchmarking
- Comparative benchmarking
- Quick comparison (3 runs)
- Detailed performance analysis
- Provider comparison

**Usage:**
```python
from llm_interface.src.cross_framework import FrameworkBenchmark

benchmark = FrameworkBenchmark(manager)
benchmark.add_framework("langchain", langchain_adapter)
benchmark.add_framework("semantic_kernel", sk_adapter)

report = benchmark.run_comparative(task, "Task Name", runs=10)
print(report.summary())
```

**Run:**
```bash
python examples/cross_framework/benchmark_example.py
```

### 4. Hybrid Workflows (`hybrid_workflow.py`)

Combine multiple frameworks in a single workflow.

**Features:**
- Research → Writing → Formatting workflow
- Parallel processing across frameworks
- Adaptive framework selection
- Quality vs Speed trade-offs

**Usage:**
```python
from llm_interface.src.cross_framework import FrameworkSwitcher

switcher = FrameworkSwitcher(manager)
# Register multiple frameworks
switcher.register("langchain", langchain_adapter)
switcher.register("semantic_kernel", sk_adapter)

# Research stage with LangChain
switcher.switch_to("langchain", "Best for research")
research = switcher.execute(research_task)

# Writing stage with Semantic Kernel
switcher.switch_to("semantic_kernel", "Best for plugins")
content = switcher.execute(writing_task)
```

**Run:**
```bash
python examples/cross_framework/hybrid_workflow.py
```

## Example Output

### Comparison Report
```
==============================================================
FRAMEWORK COMPARISON REPORT
==============================================================
Task: AI Summarization Task
Timestamp: 2025-01-15 10:30:00

Results:
  semantic_kernel: ✅ SUCCESS (1.234s)
  langchain: ✅ SUCCESS (1.456s)

Fastest: semantic_kernel
Success: 2/2
==============================================================
```

### Benchmark Report
```
======================================================================
COMPARATIVE BENCHMARK REPORT
======================================================================
Task: Summarization Task
Timestamp: 2025-01-15 10:30:00

----------------------------------------------------------------------
Framework: semantic_kernel
Task: Summarization Task
Runs: 10 (10 successful, 0 failed)
Success Rate: 100.0%

Timing:
  Mean: 1.234s
  Median: 1.220s
  Min: 1.100s
  Max: 1.450s
  Std Dev: 0.095s

----------------------------------------------------------------------
Framework: langchain
Task: Summarization Task
Runs: 10 (10 successful, 0 failed)
Success Rate: 100.0%

Timing:
  Mean: 1.456s
  Median: 1.430s
  Min: 1.300s
  Max: 1.650s
  Std Dev: 0.110s

======================================================================

COMPARISON:
  Fastest: semantic_kernel (1.234s mean)
  Most Reliable: semantic_kernel (100.0% success)

  Speed Ranking:
    1. semantic_kernel: 1.234s (1.00x)
    2. langchain: 1.456s (0.85x)
======================================================================
```

## Configuration

All examples use the same manager setup pattern:

```python
from llm_interface import LLMManager, LLMConfig, ProviderType

def setup_manager():
    manager = LLMManager()
    
    # Add OpenAI provider
    config = LLMConfig(
        provider=ProviderType.OPENAI,
        model_name="gpt-4",
        api_key="your-api-key"  # Or use environment variable
    )
    manager.add_provider("gpt4", config)
    
    return manager
```

**Environment Variables:**
```bash
export OPENAI_API_KEY="your-key-here"
export ANTHROPIC_API_KEY="your-key-here"  # If using Anthropic
```

## Use Cases

### When to Use Comparison
- Evaluating frameworks for a new project
- Testing framework capabilities
- Choosing the best framework for a task type

### When to Use Switching
- Building resilient applications with fallback
- Supporting multiple frameworks in production
- A/B testing different frameworks

### When to Use Benchmarking
- Performance optimization
- Cost analysis (execution time)
- Framework selection based on metrics

### When to Use Hybrid Workflows
- Leveraging framework-specific strengths
- Complex multi-stage pipelines
- Optimizing for different priorities (speed vs quality)

## Best Practices

1. **Start Simple**: Begin with comparison to understand differences
2. **Measure First**: Use benchmarks before making decisions
3. **Plan Fallbacks**: Set up fallback strategies for production
4. **Monitor Performance**: Track framework switches and metrics
5. **Document Choices**: Record why specific frameworks were chosen

## Tips

- Run benchmarks with multiple iterations for statistical significance
- Use warmup runs to account for initialization overhead
- Consider both speed and reliability when choosing frameworks
- Combine frameworks to leverage their unique strengths
- Monitor framework switch history for optimization opportunities

## Next Steps

- Explore individual framework examples in `examples/langchain/`, `examples/crewai/`, etc.
- Read framework adapter documentation
- Experiment with different provider configurations
- Build custom hybrid workflows for your use case

## Support

For issues or questions:
- Check the main README.md
- Review framework-specific documentation
- See ARCHITECTURE_EXPANSION.md for design details
