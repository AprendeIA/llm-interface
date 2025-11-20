"""
Framework comparison utilities.

Provides tools to compare the same task across different AI frameworks
to help users choose the best framework for their needs.
"""

from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
import time
from ..manager import LLMManager
from ..framework.base import FrameworkAdapter


@dataclass
class ComparisonResult:
    """Result of comparing a task across frameworks."""
    
    framework_name: str
    success: bool
    execution_time: float
    result: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __str__(self) -> str:
        status = "✅ SUCCESS" if self.success else "❌ FAILED"
        time_str = f"{self.execution_time:.3f}s"
        if self.error:
            return f"{self.framework_name}: {status} ({time_str}) - {self.error}"
        return f"{self.framework_name}: {status} ({time_str})"


@dataclass
class ComparisonReport:
    """Comprehensive comparison report across frameworks."""
    
    task_description: str
    results: List[ComparisonResult]
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def fastest_framework(self) -> Optional[str]:
        """Get the fastest successful framework."""
        successful = [r for r in self.results if r.success]
        if not successful:
            return None
        return min(successful, key=lambda r: r.execution_time).framework_name
    
    @property
    def success_rate(self) -> Dict[str, bool]:
        """Get success status for each framework."""
        return {r.framework_name: r.success for r in self.results}
    
    @property
    def execution_times(self) -> Dict[str, float]:
        """Get execution times for each framework."""
        return {r.framework_name: r.execution_time for r in self.results}
    
    def summary(self) -> str:
        """Generate a text summary of the comparison."""
        lines = [
            "=" * 60,
            "FRAMEWORK COMPARISON REPORT",
            "=" * 60,
            f"Task: {self.task_description}",
            f"Timestamp: {self.timestamp}",
            "",
            "Results:",
        ]
        
        for result in sorted(self.results, key=lambda r: r.execution_time):
            lines.append(f"  {result}")
        
        lines.append("")
        fastest = self.fastest_framework
        if fastest:
            lines.append(f"Fastest: {fastest}")
        
        success_count = sum(1 for r in self.results if r.success)
        lines.append(f"Success: {success_count}/{len(self.results)}")
        lines.append("=" * 60)
        
        return "\n".join(lines)


class FrameworkComparison:
    """
    Tool for comparing task execution across multiple frameworks.
    
    Example:
        >>> comparison = FrameworkComparison(manager)
        >>> comparison.add_framework("langchain", langchain_adapter)
        >>> comparison.add_framework("crewai", crewai_adapter)
        >>> 
        >>> def task(adapter):
        ...     return adapter.execute_simple_task("Summarize AI trends")
        >>> 
        >>> report = comparison.compare(task, "Summarization Task")
        >>> print(report.summary())
    """
    
    def __init__(self, llm_manager: LLMManager):
        """
        Initialize framework comparison tool.
        
        Args:
            llm_manager: LLM manager instance
        """
        self.llm_manager = llm_manager
        self.frameworks: Dict[str, FrameworkAdapter] = {}
    
    def add_framework(self, name: str, adapter: FrameworkAdapter) -> None:
        """
        Register a framework adapter for comparison.
        
        Args:
            name: Framework identifier
            adapter: Framework adapter instance
        """
        self.frameworks[name] = adapter
    
    def remove_framework(self, name: str) -> None:
        """
        Remove a framework from comparison.
        
        Args:
            name: Framework identifier
        """
        if name in self.frameworks:
            del self.frameworks[name]
    
    def compare(
        self,
        task: Callable[[FrameworkAdapter], Any],
        description: str,
        frameworks: Optional[List[str]] = None,
        timeout: float = 60.0
    ) -> ComparisonReport:
        """
        Execute a task across multiple frameworks and compare results.
        
        Args:
            task: Callable that takes a FrameworkAdapter and returns a result
            description: Description of the task being compared
            frameworks: List of framework names to compare (None = all)
            timeout: Maximum execution time per framework (seconds)
            
        Returns:
            ComparisonReport with results from all frameworks
            
        Example:
            >>> def summarize_task(adapter):
            ...     return adapter.invoke("Summarize: AI in 2024")
            >>> 
            >>> report = comparison.compare(
            ...     summarize_task,
            ...     "AI Summarization",
            ...     frameworks=["langchain", "crewai"]
            ... )
        """
        frameworks_to_test = frameworks or list(self.frameworks.keys())
        results = []
        
        for framework_name in frameworks_to_test:
            if framework_name not in self.frameworks:
                results.append(ComparisonResult(
                    framework_name=framework_name,
                    success=False,
                    execution_time=0.0,
                    error=f"Framework '{framework_name}' not registered"
                ))
                continue
            
            adapter = self.frameworks[framework_name]
            start_time = time.time()
            
            try:
                result = task(adapter)
                execution_time = time.time() - start_time
                
                if execution_time > timeout:
                    results.append(ComparisonResult(
                        framework_name=framework_name,
                        success=False,
                        execution_time=execution_time,
                        error=f"Timeout exceeded ({timeout}s)"
                    ))
                else:
                    results.append(ComparisonResult(
                        framework_name=framework_name,
                        success=True,
                        execution_time=execution_time,
                        result=result,
                        metadata={"adapter_version": adapter.framework_version}
                    ))
                    
            except Exception as e:
                execution_time = time.time() - start_time
                results.append(ComparisonResult(
                    framework_name=framework_name,
                    success=False,
                    execution_time=execution_time,
                    error=str(e)
                ))
        
        return ComparisonReport(
            task_description=description,
            results=results
        )
    
    def compare_simple_prompt(
        self,
        prompt: str,
        provider_name: str,
        frameworks: Optional[List[str]] = None
    ) -> ComparisonReport:
        """
        Compare how different frameworks handle a simple prompt.
        
        Args:
            prompt: The prompt to execute
            provider_name: LLM provider to use
            frameworks: List of frameworks to compare (None = all)
            
        Returns:
            ComparisonReport with results
            
        Example:
            >>> report = comparison.compare_simple_prompt(
            ...     "What is the capital of France?",
            ...     "gpt4"
            ... )
        """
        def task(adapter: FrameworkAdapter) -> Any:
            # Try to invoke the prompt using the adapter's interface
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
        
        return self.compare(
            task,
            f"Simple Prompt: '{prompt[:50]}...'",
            frameworks
        )
    
    def list_frameworks(self) -> List[str]:
        """
        Get list of registered frameworks.
        
        Returns:
            List of framework names
        """
        return list(self.frameworks.keys())
    
    def get_framework_info(self, name: str) -> Dict[str, Any]:
        """
        Get information about a registered framework.
        
        Args:
            name: Framework identifier
            
        Returns:
            Dictionary with framework information
        """
        if name not in self.frameworks:
            raise ValueError(f"Framework '{name}' not registered")
        
        adapter = self.frameworks[name]
        return {
            "name": adapter.framework_name,
            "version": adapter.framework_version,
            "available": hasattr(adapter, 'is_available') and adapter.is_available(),
            "supported_providers": (
                adapter.get_provider_info() 
                if hasattr(adapter, 'get_provider_info') 
                else {}
            )
        }


def compare_frameworks(
    llm_manager: LLMManager,
    framework_adapters: Dict[str, FrameworkAdapter],
    task: Callable[[FrameworkAdapter], Any],
    description: str
) -> ComparisonReport:
    """
    Convenience function to quickly compare frameworks.
    
    Args:
        llm_manager: LLM manager instance
        framework_adapters: Dict mapping framework names to adapters
        task: Task function to execute
        description: Task description
        
    Returns:
        ComparisonReport
        
    Example:
        >>> from llm_interface import LLMManager
        >>> from llm_interface.framework.langchain import LangChainAdapter
        >>> from llm_interface.framework.crewai import CrewAIAdapter
        >>> 
        >>> manager = LLMManager()
        >>> # ... add providers ...
        >>> 
        >>> report = compare_frameworks(
        ...     manager,
        ...     {
        ...         "langchain": LangChainAdapter(manager),
        ...         "crewai": CrewAIAdapter(manager)
        ...     },
        ...     lambda adapter: adapter.execute("Summarize AI"),
        ...     "AI Summarization"
        ... )
        >>> print(report.summary())
    """
    comparison = FrameworkComparison(llm_manager)
    for name, adapter in framework_adapters.items():
        comparison.add_framework(name, adapter)
    
    return comparison.compare(task, description)
