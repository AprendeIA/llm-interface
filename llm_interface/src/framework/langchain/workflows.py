"""LangChain workflow builders and patterns.

Provides reusable workflow patterns built on top of LangChain and LangGraph.
These workflows demonstrate best practices for multi-provider, multi-model scenarios.
"""

from typing import Any, Dict, Optional, List, Callable, Union, TypedDict
from dataclasses import dataclass
from abc import ABC, abstractmethod
from enum import Enum

from llm_interface.src.framework.exceptions import FrameworkConfigurationError


class WorkflowType(str, Enum):
    """Types of workflows supported."""
    CHAIN = "chain"
    CONDITIONAL = "conditional"
    PARALLEL = "parallel"
    SEQUENTIAL = "sequential"


@dataclass
class WorkflowConfig:
    """Configuration for a workflow.
    
    Attributes:
        name: Workflow name
        provider: LLM provider to use
        workflow_type: Type of workflow
        description: Human-readable description
        metadata: Additional configuration
    """
    name: str
    provider: str
    workflow_type: WorkflowType
    description: str = ""
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        """Validate configuration."""
        if not self.name:
            raise FrameworkConfigurationError("Workflow name required", framework="LangChain")
        if not self.provider:
            raise FrameworkConfigurationError("Provider name required", framework="LangChain")
        if self.metadata is None:
            self.metadata = {}


class WorkflowBuilder(ABC):
    """Base class for LangChain workflow builders.
    
    Provides common interface for building different types of workflows.
    """
    
    def __init__(self, config: WorkflowConfig):
        """Initialize workflow builder.
        
        Args:
            config: WorkflowConfig instance
        """
        self.config = config
        self._nodes: Dict[str, Callable] = {}
        self._edges: List[tuple] = []
    
    @abstractmethod
    def add_node(
        self,
        name: str,
        func: Callable,
        **kwargs: Any
    ) -> "WorkflowBuilder":
        """Add a node to the workflow.
        
        Args:
            name: Node identifier
            func: Function or callable to execute
            **kwargs: Additional configuration
            
        Returns:
            Self for chaining
        """
        pass
    
    @abstractmethod
    def add_edge(
        self,
        source: str,
        target: str,
        condition: Optional[Callable] = None
    ) -> "WorkflowBuilder":
        """Add an edge between nodes.
        
        Args:
            source: Source node name
            target: Target node name
            condition: Optional condition function for conditional edges
            
        Returns:
            Self for chaining
        """
        pass
    
    @abstractmethod
    def build(self) -> Any:
        """Build and return the workflow.
        
        Returns:
            Compiled workflow
        """
        pass


class ChainWorkflow(WorkflowBuilder):
    """Simple chain workflow - nodes execute sequentially.
    
    Example:
        >>> config = WorkflowConfig(
        ...     name="summary_chain",
        ...     provider="openai",
        ...     workflow_type=WorkflowType.CHAIN
        ... )
        >>> workflow = ChainWorkflow(config)
        >>> workflow.add_node("extract", extract_func)
        >>> workflow.add_node("summarize", summarize_func)
        >>> workflow.add_edge("extract", "summarize")
        >>> chain = workflow.build()
    """
    
    def __init__(self, config: WorkflowConfig):
        """Initialize chain workflow."""
        super().__init__(config)
        self._node_order: List[str] = []
    
    def add_node(
        self,
        name: str,
        func: Callable,
        **kwargs: Any
    ) -> "ChainWorkflow":
        """Add a node to the chain.
        
        Args:
            name: Node identifier
            func: Function to execute
            **kwargs: Additional configuration
            
        Returns:
            Self for chaining
            
        Raises:
            FrameworkConfigurationError: If node already exists
        """
        if name in self._nodes:
            raise FrameworkConfigurationError(
                f"Node '{name}' already exists",
                framework="LangChain"
            )
        self._nodes[name] = func
        self._node_order.append(name)
        return self
    
    def add_edge(
        self,
        source: str,
        target: str,
        condition: Optional[Callable] = None
    ) -> "ChainWorkflow":
        """Add edge in chain (edges ignored for linear execution).
        
        Args:
            source: Source node name
            target: Target node name
            condition: Ignored for chain workflows
            
        Returns:
            Self for chaining
        """
        if source not in self._nodes:
            raise FrameworkConfigurationError(
                f"Source node '{source}' not found",
                framework="LangChain"
            )
        if target not in self._nodes:
            raise FrameworkConfigurationError(
                f"Target node '{target}' not found",
                framework="LangChain"
            )
        self._edges.append((source, target))
        return self
    
    def build(self) -> Dict[str, Callable]:
        """Build workflow.
        
        Returns:
            Dictionary of nodes in execution order
        """
        return {name: self._nodes[name] for name in self._node_order}


class ConditionalWorkflow(WorkflowBuilder):
    """Conditional workflow - routes based on conditions.
    
    Example:
        >>> config = WorkflowConfig(
        ...     name="routing_workflow",
        ...     provider="openai",
        ...     workflow_type=WorkflowType.CONDITIONAL
        ... )
        >>> workflow = ConditionalWorkflow(config)
        >>> workflow.add_node("route", route_func)
        >>> workflow.add_node("technical", technical_func)
        >>> workflow.add_node("general", general_func)
        >>> workflow.add_edge("route", "technical", is_technical)
        >>> workflow.add_edge("route", "general")
        >>> compiled = workflow.build()
    """
    
    def __init__(self, config: WorkflowConfig):
        """Initialize conditional workflow."""
        super().__init__(config)
        self._conditions: Dict[tuple, Optional[Callable]] = {}
    
    def add_node(
        self,
        name: str,
        func: Callable,
        **kwargs: Any
    ) -> "ConditionalWorkflow":
        """Add a node.
        
        Args:
            name: Node identifier
            func: Function to execute
            **kwargs: Additional configuration
            
        Returns:
            Self for chaining
        """
        if name in self._nodes:
            raise FrameworkConfigurationError(
                f"Node '{name}' already exists",
                framework="LangChain"
            )
        self._nodes[name] = func
        return self
    
    def add_edge(
        self,
        source: str,
        target: str,
        condition: Optional[Callable] = None
    ) -> "ConditionalWorkflow":
        """Add conditional edge.
        
        Args:
            source: Source node name
            target: Target node name
            condition: Function returning bool for routing
            
        Returns:
            Self for chaining
            
        Raises:
            FrameworkConfigurationError: If nodes not found
        """
        if source not in self._nodes:
            raise FrameworkConfigurationError(
                f"Source node '{source}' not found",
                framework="LangChain"
            )
        if target not in self._nodes:
            raise FrameworkConfigurationError(
                f"Target node '{target}' not found",
                framework="LangChain"
            )
        edge_key = (source, target)
        self._conditions[edge_key] = condition
        self._edges.append(edge_key)
        return self
    
    def build(self) -> Dict[str, Any]:
        """Build workflow with conditions.
        
        Returns:
            Dictionary with nodes and routing conditions
        """
        return {
            "nodes": self._nodes,
            "edges": self._edges,
            "conditions": self._conditions,
        }


class ParallelWorkflow(WorkflowBuilder):
    """Parallel workflow - executes nodes concurrently.
    
    Example:
        >>> config = WorkflowConfig(
        ...     name="parallel_analysis",
        ...     provider="openai",
        ...     workflow_type=WorkflowType.PARALLEL
        ... )
        >>> workflow = ParallelWorkflow(config)
        >>> workflow.add_node("sentiment", sentiment_func)
        >>> workflow.add_node("entities", entities_func)
        >>> workflow.add_node("keywords", keywords_func)
        >>> compiled = workflow.build()
    """
    
    def __init__(self, config: WorkflowConfig):
        """Initialize parallel workflow."""
        super().__init__(config)
        self._parallel_groups: List[List[str]] = []
        self._current_group: List[str] = []
    
    def add_node(
        self,
        name: str,
        func: Callable,
        **kwargs: Any
    ) -> "ParallelWorkflow":
        """Add a node.
        
        Args:
            name: Node identifier
            func: Function to execute
            **kwargs: Additional configuration
            
        Returns:
            Self for chaining
        """
        if name in self._nodes:
            raise FrameworkConfigurationError(
                f"Node '{name}' already exists",
                framework="LangChain"
            )
        self._nodes[name] = func
        self._current_group.append(name)
        return self
    
    def add_edge(
        self,
        source: str,
        target: str,
        condition: Optional[Callable] = None
    ) -> "ParallelWorkflow":
        """Add edge (moves to next parallel group).
        
        Args:
            source: Source node name
            target: Target node name
            condition: Ignored for parallel workflows
            
        Returns:
            Self for chaining
        """
        if source not in self._nodes or target not in self._nodes:
            raise FrameworkConfigurationError(
                "Source or target node not found",
                framework="LangChain"
            )
        # Start new parallel group for target
        if self._current_group:
            self._parallel_groups.append(self._current_group.copy())
            self._current_group = [target]
        self._edges.append((source, target))
        return self
    
    def build(self) -> Dict[str, Any]:
        """Build parallel workflow.
        
        Returns:
            Dictionary with nodes and parallel groups
        """
        if self._current_group:
            self._parallel_groups.append(self._current_group)
        
        return {
            "nodes": self._nodes,
            "parallel_groups": self._parallel_groups,
            "edges": self._edges,
        }


class WorkflowFactory:
    """Factory for creating workflow instances.
    
    Example:
        >>> factory = WorkflowFactory()
        >>> workflow = factory.create(
        ...     WorkflowType.CHAIN,
        ...     "my_workflow",
        ...     "openai"
        ... )
    """
    
    _builders: Dict[WorkflowType, type] = {
        WorkflowType.CHAIN: ChainWorkflow,
        WorkflowType.CONDITIONAL: ConditionalWorkflow,
        WorkflowType.PARALLEL: ParallelWorkflow,
    }
    
    @classmethod
    def create(
        cls,
        workflow_type: WorkflowType,
        name: str,
        provider: str,
        **kwargs: Any
    ) -> WorkflowBuilder:
        """Create a workflow builder.
        
        Args:
            workflow_type: Type of workflow
            name: Workflow name
            provider: Provider name
            **kwargs: Additional configuration
            
        Returns:
            Appropriate WorkflowBuilder instance
            
        Raises:
            FrameworkConfigurationError: If workflow type not supported
        """
        if workflow_type not in cls._builders:
            raise FrameworkConfigurationError(
                f"Unknown workflow type: {workflow_type}",
                framework="LangChain"
            )
        
        config = WorkflowConfig(
            name=name,
            provider=provider,
            workflow_type=workflow_type,
            metadata=kwargs
        )
        return cls._builders[workflow_type](config)
    
    @classmethod
    def register(
        cls,
        workflow_type: WorkflowType,
        builder_class: type
    ) -> None:
        """Register a custom workflow builder.
        
        Args:
            workflow_type: Workflow type identifier
            builder_class: WorkflowBuilder subclass
        """
        cls._builders[workflow_type] = builder_class
