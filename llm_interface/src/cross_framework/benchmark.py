"""
Framework benchmarking utilities.

Provides tools to measure and compare performance metrics across
different AI frameworks.
"""

from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
import time
import statistics
from ..manager import LLMManager
from ..framework.base import FrameworkAdapter


@dataclass
class BenchmarkMetrics:
    """Performance metrics for a single benchmark run."""
    
    execution_time: float
    success: bool
    error: Optional[str] = None
    tokens_used: Optional[int] = None
    memory_used: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Aggregated benchmark results for a framework."""
    
    framework_name: str
    task_description: str
    runs: int
    successful_runs: int
    failed_runs: int
    
    # Timing statistics
    mean_time: float
    median_time: float
    min_time: float
    max_time: float
    std_dev_time: float
    
    # Additional metrics
    total_tokens: Optional[int] = None
    avg_tokens_per_run: Optional[float] = None
    
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.runs == 0:
            return 0.0
        return (self.successful_runs / self.runs) * 100
    
    def summary(self) -> str:
        """Generate a text summary of the benchmark."""
        lines = [
            f"Framework: {self.framework_name}",
            f"Task: {self.task_description}",
            f"Runs: {self.runs} ({self.successful_runs} successful, {self.failed_runs} failed)",
            f"Success Rate: {self.success_rate:.1f}%",
            "",
            "Timing:",
            f"  Mean: {self.mean_time:.3f}s",
            f"  Median: {self.median_time:.3f}s",
            f"  Min: {self.min_time:.3f}s",
            f"  Max: {self.max_time:.3f}s",
            f"  Std Dev: {self.std_dev_time:.3f}s",
        ]
        
        if self.total_tokens:
            lines.extend([
                "",
                f"Tokens: {self.total_tokens} total, {self.avg_tokens_per_run:.1f} avg/run"
            ])
        
        return "\n".join(lines)


@dataclass
class ComparativeBenchmark:
    """Comparative benchmark results across multiple frameworks."""
    
    task_description: str
    results: List[BenchmarkResult]
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def fastest_framework(self) -> Optional[str]:
        """Get the framework with lowest mean execution time."""
        successful = [r for r in self.results if r.successful_runs > 0]
        if not successful:
            return None
        return min(successful, key=lambda r: r.mean_time).framework_name
    
    @property
    def most_reliable(self) -> Optional[str]:
        """Get the framework with highest success rate."""
        if not self.results:
            return None
        return max(self.results, key=lambda r: r.success_rate).framework_name
    
    def summary(self) -> str:
        """Generate a comparative summary."""
        lines = [
            "=" * 70,
            "COMPARATIVE BENCHMARK REPORT",
            "=" * 70,
            f"Task: {self.task_description}",
            f"Timestamp: {self.timestamp}",
            "",
        ]
        
        # Add individual framework summaries
        for result in sorted(self.results, key=lambda r: r.mean_time):
            lines.extend([
                "-" * 70,
                result.summary(),
            ])
        
        lines.extend([
            "=" * 70,
            "",
            "COMPARISON:",
        ])
        
        # Fastest
        fastest = self.fastest_framework
        if fastest:
            fastest_result = next(r for r in self.results if r.framework_name == fastest)
            lines.append(f"  Fastest: {fastest} ({fastest_result.mean_time:.3f}s mean)")
        
        # Most reliable
        most_reliable = self.most_reliable
        if most_reliable:
            reliable_result = next(r for r in self.results if r.framework_name == most_reliable)
            lines.append(f"  Most Reliable: {most_reliable} ({reliable_result.success_rate:.1f}% success)")
        
        # Speed comparison
        if len(self.results) >= 2:
            sorted_by_speed = sorted(self.results, key=lambda r: r.mean_time)
            fastest_time = sorted_by_speed[0].mean_time
            lines.append("")
            lines.append("  Speed Ranking:")
            for i, result in enumerate(sorted_by_speed, 1):
                speedup = fastest_time / result.mean_time if result.mean_time > 0 else 0
                lines.append(f"    {i}. {result.framework_name}: {result.mean_time:.3f}s ({speedup:.2f}x)")
        
        lines.append("=" * 70)
        
        return "\n".join(lines)


class FrameworkBenchmark:
    """
    Tool for benchmarking framework performance.
    
    Example:
        >>> benchmark = FrameworkBenchmark(manager)
        >>> benchmark.add_framework("langchain", langchain_adapter)
        >>> benchmark.add_framework("crewai", crewai_adapter)
        >>> 
        >>> def task(adapter):
        ...     return adapter.invoke("Summarize AI trends")
        >>> 
        >>> report = benchmark.run_comparative(
        ...     task,
        ...     "Summarization",
        ...     runs=10
        ... )
        >>> print(report.summary())
    """
    
    def __init__(self, llm_manager: LLMManager):
        """
        Initialize framework benchmark tool.
        
        Args:
            llm_manager: LLM manager instance
        """
        self.llm_manager = llm_manager
        self.frameworks: Dict[str, FrameworkAdapter] = {}
    
    def add_framework(self, name: str, adapter: FrameworkAdapter) -> None:
        """
        Register a framework for benchmarking.
        
        Args:
            name: Framework identifier
            adapter: Framework adapter instance
        """
        self.frameworks[name] = adapter
    
    def remove_framework(self, name: str) -> None:
        """
        Remove a framework from benchmarking.
        
        Args:
            name: Framework identifier
        """
        if name in self.frameworks:
            del self.frameworks[name]
    
    def benchmark_single(
        self,
        framework_name: str,
        task: Callable[[FrameworkAdapter], Any],
        description: str,
        runs: int = 5,
        warmup_runs: int = 1
    ) -> BenchmarkResult:
        """
        Benchmark a single framework.
        
        Args:
            framework_name: Framework to benchmark
            task: Task function to execute
            description: Task description
            runs: Number of benchmark runs
            warmup_runs: Number of warmup runs (excluded from results)
            
        Returns:
            BenchmarkResult
            
        Example:
            >>> result = benchmark.benchmark_single(
            ...     "langchain",
            ...     lambda adapter: adapter.invoke("Hello"),
            ...     "Simple Greeting",
            ...     runs=10
            ... )
        """
        if framework_name not in self.frameworks:
            raise ValueError(f"Framework '{framework_name}' not registered")
        
        adapter = self.frameworks[framework_name]
        metrics_list: List[BenchmarkMetrics] = []
        
        # Warmup runs
        for _ in range(warmup_runs):
            try:
                task(adapter)
            except Exception:
                pass  # Ignore warmup errors
        
        # Actual benchmark runs
        for _ in range(runs):
            start_time = time.time()
            
            try:
                result = task(adapter)
                execution_time = time.time() - start_time
                
                metrics = BenchmarkMetrics(
                    execution_time=execution_time,
                    success=True,
                    metadata={"result": str(result)[:100]}
                )
                
            except Exception as e:
                execution_time = time.time() - start_time
                metrics = BenchmarkMetrics(
                    execution_time=execution_time,
                    success=False,
                    error=str(e)
                )
            
            metrics_list.append(metrics)
        
        # Calculate statistics
        successful_times = [m.execution_time for m in metrics_list if m.success]
        all_times = [m.execution_time for m in metrics_list]
        
        if not successful_times:
            # All runs failed
            return BenchmarkResult(
                framework_name=framework_name,
                task_description=description,
                runs=runs,
                successful_runs=0,
                failed_runs=runs,
                mean_time=statistics.mean(all_times) if all_times else 0.0,
                median_time=statistics.median(all_times) if all_times else 0.0,
                min_time=min(all_times) if all_times else 0.0,
                max_time=max(all_times) if all_times else 0.0,
                std_dev_time=statistics.stdev(all_times) if len(all_times) > 1 else 0.0
            )
        
        return BenchmarkResult(
            framework_name=framework_name,
            task_description=description,
            runs=runs,
            successful_runs=len(successful_times),
            failed_runs=runs - len(successful_times),
            mean_time=statistics.mean(successful_times),
            median_time=statistics.median(successful_times),
            min_time=min(successful_times),
            max_time=max(successful_times),
            std_dev_time=statistics.stdev(successful_times) if len(successful_times) > 1 else 0.0
        )
    
    def run_comparative(
        self,
        task: Callable[[FrameworkAdapter], Any],
        description: str,
        runs: int = 5,
        frameworks: Optional[List[str]] = None,
        warmup_runs: int = 1
    ) -> ComparativeBenchmark:
        """
        Run comparative benchmark across multiple frameworks.
        
        Args:
            task: Task function to execute
            description: Task description
            runs: Number of runs per framework
            frameworks: List of frameworks to benchmark (None = all)
            warmup_runs: Number of warmup runs per framework
            
        Returns:
            ComparativeBenchmark with results
            
        Example:
            >>> report = benchmark.run_comparative(
            ...     lambda adapter: adapter.invoke("Summarize AI"),
            ...     "Summarization Task",
            ...     runs=10
            ... )
            >>> print(report.summary())
        """
        frameworks_to_test = frameworks or list(self.frameworks.keys())
        results = []
        
        for framework_name in frameworks_to_test:
            if framework_name not in self.frameworks:
                continue
            
            result = self.benchmark_single(
                framework_name,
                task,
                description,
                runs,
                warmup_runs
            )
            results.append(result)
        
        return ComparativeBenchmark(
            task_description=description,
            results=results
        )
    
    def quick_compare(
        self,
        prompt: str,
        provider_name: str,
        frameworks: Optional[List[str]] = None,
        runs: int = 3
    ) -> ComparativeBenchmark:
        """
        Quick comparison of frameworks on a simple prompt.
        
        Args:
            prompt: The prompt to execute
            provider_name: LLM provider to use
            frameworks: List of frameworks to compare (None = all)
            runs: Number of runs (default 3 for quick comparison)
            
        Returns:
            ComparativeBenchmark
            
        Example:
            >>> report = benchmark.quick_compare(
            ...     "What is AI?",
            ...     "gpt4",
            ...     runs=3
            ... )
        """
        def task(adapter: FrameworkAdapter) -> Any:
            if hasattr(adapter, 'invoke_simple'):
                return adapter.invoke_simple(prompt, provider_name)
            elif hasattr(adapter, 'create_model'):
                model = adapter.create_model(provider_name)
                if hasattr(model, 'invoke'):
                    return model.invoke(prompt)
                elif callable(model):
                    return model(prompt)
            else:
                raise NotImplementedError(
                    f"Adapter {adapter.framework_name} doesn't support simple prompt execution"
                )
        
        return self.run_comparative(
            task,
            f"Quick Compare: '{prompt[:50]}...'",
            runs,
            frameworks,
            warmup_runs=0  # No warmup for quick compare
        )
    
    def list_frameworks(self) -> List[str]:
        """Get list of registered frameworks."""
        return list(self.frameworks.keys())
