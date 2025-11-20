"""Framework-agnostic protocol interfaces.

This module defines protocol interfaces (structural subtyping) that
framework adapters and models should conform to. These are used for
type checking without requiring inheritance.
"""

from typing import Protocol, Any, Callable, Dict, List, Optional, AsyncIterator, runtime_checkable


@runtime_checkable
class FrameworkModel(Protocol):
    """Protocol for framework models.
    
    Any object implementing this protocol can be used as a framework model.
    """
    
    def invoke(self, input: Any, **kwargs) -> Any:
        """Execute model with input.
        
        Args:
            input: Input to process
            **kwargs: Additional arguments
            
        Returns:
            Model output
        """
        ...


@runtime_checkable
class FrameworkStreamableModel(Protocol):
    """Protocol for models that support streaming."""
    
    def invoke(self, input: Any, **kwargs) -> Any:
        """Execute model."""
        ...
    
    def stream(self, input: Any, **kwargs):
        """Stream model output.
        
        Yields:
            Streamed output chunks
        """
        ...


@runtime_checkable
class FrameworkAsyncModel(Protocol):
    """Protocol for models supporting async execution."""
    
    async def ainvoke(self, input: Any, **kwargs) -> Any:
        """Async execute model.
        
        Args:
            input: Input to process
            **kwargs: Additional arguments
            
        Returns:
            Model output
        """
        ...
    
    async def astream(self, input: Any, **kwargs) -> AsyncIterator[Any]:
        """Async stream model output.
        
        Args:
            input: Input to process
            **kwargs: Additional arguments
            
        Yields:
            Streamed output chunks
        """
        ...


@runtime_checkable
class FrameworkWorkflow(Protocol):
    """Protocol for framework workflows."""
    
    def add_step(self, name: str, operation: Callable, **kwargs) -> 'FrameworkWorkflow':
        """Add workflow step.
        
        Args:
            name: Step identifier
            operation: Callable to execute
            **kwargs: Additional configuration
            
        Returns:
            Self for method chaining
        """
        ...
    
    def add_edge(self, source: str, target: str) -> 'FrameworkWorkflow':
        """Add edge between workflow steps.
        
        Args:
            source: Source step name
            target: Target step name
            
        Returns:
            Self for method chaining
        """
        ...
    
    def execute(self, input: Any, **kwargs) -> Any:
        """Execute workflow.
        
        Args:
            input: Initial input
            **kwargs: Execution options
            
        Returns:
            Workflow output
        """
        ...


@runtime_checkable
class FrameworkAgent(Protocol):
    """Protocol for framework agents."""
    
    @property
    def name(self) -> str:
        """Agent name."""
        ...
    
    @property
    def role(self) -> str:
        """Agent role/purpose."""
        ...
    
    def execute(self, task: Any, **kwargs) -> Any:
        """Execute agent task.
        
        Args:
            task: Task to execute
            **kwargs: Execution options
            
        Returns:
            Task result
        """
        ...


@runtime_checkable
class FrameworkTask(Protocol):
    """Protocol for framework tasks."""
    
    @property
    def name(self) -> str:
        """Task name."""
        ...
    
    @property
    def description(self) -> str:
        """Task description."""
        ...
    
    def execute(self, agent: Any, **kwargs) -> Any:
        """Execute task with agent.
        
        Args:
            agent: Agent executing task
            **kwargs: Execution options
            
        Returns:
            Task result
        """
        ...


@runtime_checkable  
class FrameworkConversation(Protocol):
    """Protocol for framework conversation management."""
    
    def add_message(self, role: str, content: str) -> None:
        """Add message to conversation.
        
        Args:
            role: Message role (user, assistant, system)
            content: Message content
        """
        ...
    
    def get_messages(self) -> List[Dict[str, str]]:
        """Get conversation messages.
        
        Returns:
            List of message dictionaries
        """
        ...
    
    def clear(self) -> None:
        """Clear conversation history."""
        ...


@runtime_checkable
class FrameworkOrchestrator(Protocol):
    """Protocol for multi-agent orchestrators."""
    
    def add_agent(self, agent: Any) -> None:
        """Add agent to orchestrator.
        
        Args:
            agent: Agent to add
        """
        ...
    
    def execute(self, input: Any, **kwargs) -> Any:
        """Execute orchestration.
        
        Args:
            input: Input for orchestration
            **kwargs: Execution options
            
        Returns:
            Orchestration result
        """
        ...
    
    def get_result(self) -> Any:
        """Get orchestration result.
        
        Returns:
            Final result
        """
        ...
