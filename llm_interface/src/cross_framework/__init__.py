"""
Cross-framework utilities for the LLM Interface.

This module provides tools for comparing, switching, and benchmarking
different AI frameworks (LangChain, CrewAI, AutoGen, Semantic Kernel).
"""

from .comparison import FrameworkComparison, compare_frameworks
from .switcher import FrameworkSwitcher
from .benchmark import FrameworkBenchmark, BenchmarkResult

__all__ = [
    "FrameworkComparison",
    "compare_frameworks",
    "FrameworkSwitcher",
    "FrameworkBenchmark",
    "BenchmarkResult",
]
