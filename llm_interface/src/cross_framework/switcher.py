"""
Framework switching utilities.

Provides tools to dynamically switch between different AI frameworks
while maintaining consistent provider usage.
"""

from typing import Dict, Any, Optional, Callable
from enum import Enum
from dataclasses import dataclass
from ..manager import LLMManager
from ..framework.base import FrameworkAdapter
from ..framework.exceptions import FrameworkNotAvailableError


class SwitchStrategy(Enum):
    """Strategy for framework switching."""
    
    MANUAL = "manual"  # User explicitly switches
    FALLBACK = "fallback"  # Auto-fallback on error
    ROUND_ROBIN = "round_robin"  # Rotate through frameworks
    FASTEST = "fastest"  # Use fastest from benchmarks
    MOST_CAPABLE = "most_capable"  # Use most feature-rich


@dataclass
class SwitchEvent:
    """Record of a framework switch."""
    
    from_framework: Optional[str]
    to_framework: str
    reason: str
    strategy: SwitchStrategy
    success: bool
    timestamp: float


class FrameworkSwitcher:
    """
    Dynamic framework switching manager.
    
    Allows applications to switch between different AI frameworks
    on-the-fly, with support for fallback strategies and monitoring.
    
    Example:
        >>> switcher = FrameworkSwitcher(manager)
        >>> switcher.register("langchain", langchain_adapter)
        >>> switcher.register("crewai", crewai_adapter)
        >>> 
        >>> # Use current framework
        >>> result = switcher.execute(lambda f: f.invoke("Hello"))
        >>> 
        >>> # Switch explicitly
        >>> switcher.switch_to("crewai")
        >>> result = switcher.execute(lambda f: f.invoke("Hello"))
    """
    
    def __init__(
        self,
        llm_manager: LLMManager,
        default_strategy: SwitchStrategy = SwitchStrategy.MANUAL
    ):
        """
        Initialize framework switcher.
        
        Args:
            llm_manager: LLM manager instance
            default_strategy: Default switching strategy
        """
        self.llm_manager = llm_manager
        self.frameworks: Dict[str, FrameworkAdapter] = {}
        self.current_framework: Optional[str] = None
        self.default_strategy = default_strategy
        self.switch_history: list[SwitchEvent] = []
        self.fallback_order: list[str] = []
        self._round_robin_index = 0
    
    def register(
        self,
        name: str,
        adapter: FrameworkAdapter,
        set_as_current: bool = False
    ) -> None:
        """
        Register a framework adapter.
        
        Args:
            name: Framework identifier
            adapter: Framework adapter instance
            set_as_current: If True, set as current framework
        """
        self.frameworks[name] = adapter
        
        if set_as_current or self.current_framework is None:
            self.current_framework = name
    
    def unregister(self, name: str) -> None:
        """
        Unregister a framework.
        
        Args:
            name: Framework identifier
        """
        if name in self.frameworks:
            del self.frameworks[name]
            
            if self.current_framework == name:
                # Switch to first available framework
                self.current_framework = (
                    next(iter(self.frameworks.keys()), None)
                )
    
    def switch_to(
        self,
        framework_name: str,
        reason: str = "Manual switch"
    ) -> bool:
        """
        Switch to a specific framework.
        
        Args:
            framework_name: Target framework name
            reason: Reason for switching
            
        Returns:
            True if switch successful
            
        Raises:
            FrameworkNotAvailableError: If framework not registered
        """
        if framework_name not in self.frameworks:
            raise FrameworkNotAvailableError(
                f"Framework '{framework_name}' is not registered"
            )
        
        old_framework = self.current_framework
        self.current_framework = framework_name
        
        self.switch_history.append(SwitchEvent(
            from_framework=old_framework,
            to_framework=framework_name,
            reason=reason,
            strategy=SwitchStrategy.MANUAL,
            success=True,
            timestamp=time.time()
        ))
        
        return True
    
    def get_current(self) -> Optional[FrameworkAdapter]:
        """
        Get the current framework adapter.
        
        Returns:
            Current FrameworkAdapter or None
        """
        if self.current_framework is None:
            return None
        return self.frameworks.get(self.current_framework)
    
    def get_current_name(self) -> Optional[str]:
        """
        Get the name of the current framework.
        
        Returns:
            Current framework name or None
        """
        return self.current_framework
    
    def execute(
        self,
        task: Callable[[FrameworkAdapter], Any],
        strategy: Optional[SwitchStrategy] = None,
        fallback_on_error: bool = True
    ) -> Any:
        """
        Execute a task with the current framework.
        
        Args:
            task: Callable that takes a FrameworkAdapter and returns a result
            strategy: Override default strategy for this execution
            fallback_on_error: If True, try fallback frameworks on error
            
        Returns:
            Task result
            
        Raises:
            FrameworkNotAvailableError: If no frameworks available
            
        Example:
            >>> result = switcher.execute(
            ...     lambda adapter: adapter.invoke("Hello"),
            ...     fallback_on_error=True
            ... )
        """
        if not self.frameworks:
            raise FrameworkNotAvailableError("No frameworks registered")
        
        if self.current_framework is None:
            self.current_framework = next(iter(self.frameworks.keys()))
        
        adapter = self.frameworks[self.current_framework]
        
        try:
            return task(adapter)
        except Exception as e:
            if not fallback_on_error or not self.fallback_order:
                raise
            
            # Try fallback frameworks
            for fallback_name in self.fallback_order:
                if fallback_name == self.current_framework:
                    continue
                
                if fallback_name not in self.frameworks:
                    continue
                
                try:
                    fallback_adapter = self.frameworks[fallback_name]
                    result = task(fallback_adapter)
                    
                    # Switch successful, update current
                    self.switch_to(
                        fallback_name,
                        f"Fallback from error: {str(e)[:50]}"
                    )
                    
                    return result
                except Exception:
                    continue
            
            # All fallbacks failed, raise original error
            raise
    
    def set_fallback_order(self, frameworks: list[str]) -> None:
        """
        Set the order of frameworks for fallback strategy.
        
        Args:
            frameworks: List of framework names in priority order
            
        Example:
            >>> switcher.set_fallback_order(["langchain", "crewai", "autogen"])
        """
        # Validate all frameworks are registered
        for name in frameworks:
            if name not in self.frameworks:
                raise ValueError(f"Framework '{name}' is not registered")
        
        self.fallback_order = frameworks
    
    def next_framework(self) -> str:
        """
        Get the next framework in round-robin order.
        
        Returns:
            Next framework name
        """
        if not self.frameworks:
            raise FrameworkNotAvailableError("No frameworks registered")
        
        framework_names = list(self.frameworks.keys())
        self._round_robin_index = (self._round_robin_index + 1) % len(framework_names)
        return framework_names[self._round_robin_index]
    
    def switch_next(self) -> str:
        """
        Switch to the next framework in round-robin order.
        
        Returns:
            Name of framework switched to
        """
        next_name = self.next_framework()
        self.switch_to(next_name, "Round-robin rotation")
        return next_name
    
    def list_frameworks(self) -> list[str]:
        """
        Get list of registered frameworks.
        
        Returns:
            List of framework names
        """
        return list(self.frameworks.keys())
    
    def get_switch_history(self, limit: Optional[int] = None) -> list[SwitchEvent]:
        """
        Get history of framework switches.
        
        Args:
            limit: Maximum number of events to return (None = all)
            
        Returns:
            List of SwitchEvent objects
        """
        if limit is None:
            return self.switch_history.copy()
        return self.switch_history[-limit:]
    
    def clear_history(self) -> None:
        """Clear the switch history."""
        self.switch_history.clear()
    
    def framework_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get usage statistics for each framework.
        
        Returns:
            Dict mapping framework names to stats
        """
        stats = {name: {"switches_to": 0, "switches_from": 0} 
                 for name in self.frameworks.keys()}
        
        for event in self.switch_history:
            if event.to_framework in stats:
                stats[event.to_framework]["switches_to"] += 1
            if event.from_framework and event.from_framework in stats:
                stats[event.from_framework]["switches_from"] += 1
        
        # Add current framework indicator
        if self.current_framework:
            stats[self.current_framework]["is_current"] = True
        
        return stats


import time
