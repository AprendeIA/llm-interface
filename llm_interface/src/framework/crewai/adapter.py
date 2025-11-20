"""CrewAI Framework Adapter.

This module provides the CrewAI adapter that integrates the unified LLM Interface
with CrewAI's agent-based framework, enabling multi-agent workflows with task
orchestration and team coordination.
"""

from typing import Dict, Any, List, Optional, Union
from ...manager import LLMManager
from ..base import FrameworkAdapter
from ..exceptions import (
    FrameworkConfigurationError,
    FrameworkModelCreationError,
    FrameworkExecutionError,
)

try:
    from crewai import Agent, Task, Crew, Process
except ImportError:
    raise FrameworkConfigurationError(
        "CrewAI is not installed. Install it with: pip install crewai"
    )


class CrewAIAdapter(FrameworkAdapter):
    """Adapter for CrewAI framework.
    
    Provides integration between the unified LLM Interface and CrewAI,
    enabling agent creation, task orchestration, and team workflows.
    
    Attributes:
        llm_manager: LLMManager instance for provider access
        config: CrewAI-specific configuration
        agents: Dictionary of created agents
        tasks: Dictionary of created tasks
        crews: Dictionary of created crews
    """
    
    def __init__(self, llm_manager: LLMManager, config: Dict[str, Any] = None):
        """Initialize CrewAI adapter.
        
        Args:
            llm_manager: LLMManager instance with configured providers
            config: Optional CrewAI configuration
            
        Raises:
            FrameworkConfigurationError: If configuration is invalid
        """
        super().__init__(llm_manager, config)
        
        self.agents: Dict[str, Agent] = {}
        self.tasks: Dict[str, Task] = {}
        self.crews: Dict[str, Crew] = {}
    
    @property
    def framework_name(self) -> str:
        """Return framework identifier."""
        return "crewai"
    
    @property
    def framework_version(self) -> str:
        """Return minimum supported framework version."""
        return "0.1.0"
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate CrewAI configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            bool: True if valid, False otherwise
        """
        if not isinstance(config, dict):
            return False
        
        # Optional configuration keys
        allowed_keys = {
            'default_provider',
            'verbose',
            'memory',
            'cache',
            'embedder',
            'step_callback',
            'task_callback',
        }
        
        for key in config.keys():
            if key not in allowed_keys:
                return False
        
        return True
    
    def create_model(self, provider_name: str, **kwargs):
        """Create framework-specific model instance.
        
        For CrewAI, this returns the chat model from the specified provider.
        
        Args:
            provider_name: Name of the provider to use
            **kwargs: Additional arguments (unused for CrewAI)
            
        Returns:
            LangChain BaseLanguageModel instance
            
        Raises:
            FrameworkModelCreationError: If model creation fails
        """
        try:
            if not self.has_provider(provider_name):
                available = self.list_providers()
                raise FrameworkModelCreationError(
                    provider_name,
                    self.framework_name,
                    f"Provider not found. Available: {available}"
                )
            
            model = self.get_chat_model(provider_name)
            if model is None:
                raise FrameworkModelCreationError(
                    provider_name,
                    self.framework_name,
                    "Chat model is None"
                )
            
            return model
        
        except Exception as e:
            if isinstance(e, FrameworkModelCreationError):
                raise
            raise FrameworkModelCreationError(
                provider_name,
                self.framework_name,
                str(e)
            )
    
    def create_agent(
        self,
        name: str,
        role: str,
        goal: str,
        backstory: str,
        provider_name: Optional[str] = None,
        tools: Optional[List[Any]] = None,
        memory: bool = False,
        verbose: bool = True,
        allow_delegation: bool = False,
        max_iter: int = 20,
        max_execution_time: Optional[int] = None,
        allow_code_execution: bool = False,
        **kwargs
    ) -> Agent:
        """Create a CrewAI Agent.
        
        Args:
            name: Agent identifier
            role: Agent's role/position
            goal: Agent's goal
            backstory: Agent's background story
            provider_name: Provider to use (defaults to first available)
            tools: List of tools available to agent
            memory: Enable agent memory
            verbose: Enable verbose logging
            allow_delegation: Allow agent to delegate tasks
            max_iter: Maximum iterations before stopping
            max_execution_time: Maximum execution time in seconds
            allow_code_execution: Allow code execution capability
            **kwargs: Additional CrewAI Agent parameters
            
        Returns:
            CrewAI Agent instance
            
        Raises:
            FrameworkModelCreationError: If LLM creation fails
        """
        try:
            # Get provider or use default
            if provider_name is None:
                provider_name = self.get_default_provider()
                if provider_name is None:
                    raise FrameworkModelCreationError(
                        "unknown",
                        self.framework_name,
                        "No providers available"
                    )
            
            # Create model from provider
            model = self.create_model(provider_name)
            
            # Build agent parameters
            agent_params = {
                'role': role,
                'goal': goal,
                'backstory': backstory,
                'llm': model,
                'tools': tools or [],
                'memory': memory,
                'verbose': verbose,
                'allow_delegation': allow_delegation,
                'max_iter': max_iter,
            }
            
            if max_execution_time is not None:
                agent_params['max_execution_time'] = max_execution_time
            
            if allow_code_execution:
                agent_params['allow_code_execution'] = True
                agent_params['code_execution_mode'] = 'safe'
            
            # Add any additional kwargs
            agent_params.update(kwargs)
            
            # Create agent
            agent = Agent(**agent_params)
            
            # Store agent
            self.agents[name] = agent
            
            return agent
        
        except Exception as e:
            if isinstance(e, FrameworkModelCreationError):
                raise
            raise FrameworkExecutionError(
                f"agent creation '{name}'",
                self.framework_name,
                e
            )
    
    def create_task(
        self,
        name: str,
        description: str,
        expected_output: str,
        agent: Optional[Agent] = None,
        agent_name: Optional[str] = None,
        tools: Optional[List[Any]] = None,
        context: Optional[List['Task']] = None,
        async_execution: bool = False,
        human_input: bool = False,
        markdown: bool = False,
        output_file: Optional[str] = None,
        **kwargs
    ) -> Task:
        """Create a CrewAI Task.
        
        Args:
            name: Task identifier
            description: Task description
            expected_output: Description of expected output
            agent: Agent to execute task (or use agent_name)
            agent_name: Name of agent from self.agents (if agent not provided)
            tools: Task-specific tools
            context: Dependent tasks
            async_execution: Execute asynchronously
            human_input: Require human input/approval
            markdown: Format output as markdown
            output_file: File to save output
            **kwargs: Additional CrewAI Task parameters
            
        Returns:
            CrewAI Task instance
            
        Raises:
            FrameworkExecutionError: If task creation fails
        """
        try:
            # Resolve agent
            if agent is None:
                if agent_name is None:
                    raise FrameworkExecutionError(
                        f"task creation '{name}'",
                        self.framework_name,
                        "Either agent or agent_name must be provided"
                    )
                
                if agent_name not in self.agents:
                    available = list(self.agents.keys())
                    raise FrameworkExecutionError(
                        f"task creation '{name}'",
                        self.framework_name,
                        f"Agent '{agent_name}' not found. Available: {available}"
                    )
                
                agent = self.agents[agent_name]
            
            # Build task parameters
            task_params = {
                'description': description,
                'expected_output': expected_output,
                'agent': agent,
                'async_execution': async_execution,
                'human_input': human_input,
                'markdown': markdown,
            }
            
            if tools:
                task_params['tools'] = tools
            
            if context:
                task_params['context'] = context
            
            if output_file:
                task_params['output_file'] = output_file
            
            # Add any additional kwargs
            task_params.update(kwargs)
            
            # Create task
            task = Task(**task_params)
            
            # Store task
            self.tasks[name] = task
            
            return task
        
        except Exception as e:
            if isinstance(e, FrameworkExecutionError):
                raise
            raise FrameworkExecutionError(
                f"task creation '{name}'",
                self.framework_name,
                e
            )
    
    def create_crew(
        self,
        name: str,
        agents: Union[List[Agent], List[str]],
        tasks: Union[List[Task], List[str]],
        process: str = "sequential",
        verbose: bool = True,
        memory: bool = False,
        cache: bool = True,
        output_log_file: Optional[Union[bool, str]] = None,
        **kwargs
    ) -> Crew:
        """Create a CrewAI Crew.
        
        Args:
            name: Crew identifier
            agents: List of Agent objects or agent names (string references)
            tasks: List of Task objects or task names (string references)
            process: Execution process ("sequential" or "hierarchical")
            verbose: Enable verbose logging
            memory: Enable crew memory
            cache: Enable result caching
            output_log_file: Log file path or True
            **kwargs: Additional Crew parameters (manager_llm for hierarchical, etc.)
            
        Returns:
            CrewAI Crew instance
            
        Raises:
            FrameworkExecutionError: If crew creation fails
        """
        try:
            # Resolve agents from names if needed
            resolved_agents = []
            for agent in agents:
                if isinstance(agent, str):
                    if agent not in self.agents:
                        available = list(self.agents.keys())
                        raise FrameworkExecutionError(
                            f"crew creation '{name}'",
                            self.framework_name,
                            f"Agent '{agent}' not found. Available: {available}"
                        )
                    resolved_agents.append(self.agents[agent])
                else:
                    resolved_agents.append(agent)
            
            # Resolve tasks from names if needed
            resolved_tasks = []
            for task in tasks:
                if isinstance(task, str):
                    if task not in self.tasks:
                        available = list(self.tasks.keys())
                        raise FrameworkExecutionError(
                            f"crew creation '{name}'",
                            self.framework_name,
                            f"Task '{task}' not found. Available: {available}"
                        )
                    resolved_tasks.append(self.tasks[task])
                else:
                    resolved_tasks.append(task)
            
            # Convert process string to Process enum
            process_enum = Process.sequential
            if process.lower() == "hierarchical":
                process_enum = Process.hierarchical
                
                # Hierarchical requires manager_llm
                if 'manager_llm' not in kwargs:
                    # Use first provider's model as manager
                    provider = self.get_default_provider()
                    if provider:
                        manager_llm = self.create_model(provider)
                        kwargs['manager_llm'] = manager_llm
            
            # Build crew parameters
            crew_params = {
                'agents': resolved_agents,
                'tasks': resolved_tasks,
                'process': process_enum,
                'verbose': verbose,
                'memory': memory,
                'cache': cache,
            }
            
            if output_log_file is not None:
                crew_params['output_log_file'] = output_log_file
            
            # Add any additional kwargs
            crew_params.update(kwargs)
            
            # Create crew
            crew = Crew(**crew_params)
            
            # Store crew
            self.crews[name] = crew
            
            return crew
        
        except Exception as e:
            if isinstance(e, FrameworkExecutionError):
                raise
            raise FrameworkExecutionError(
                f"crew creation '{name}'",
                self.framework_name,
                e
            )
    
    def kickoff_crew(self, crew_name: str, inputs: Optional[Dict[str, Any]] = None):
        """Execute a crew (kickoff).
        
        Args:
            crew_name: Name of crew to execute
            inputs: Input dictionary for the crew
            
        Returns:
            Crew output/result
            
        Raises:
            FrameworkExecutionError: If execution fails
        """
        try:
            if crew_name not in self.crews:
                available = list(self.crews.keys())
                raise FrameworkExecutionError(
                    f"crew kickoff '{crew_name}'",
                    self.framework_name,
                    f"Crew not found. Available: {available}"
                )
            
            crew = self.crews[crew_name]
            inputs = inputs or {}
            
            result = crew.kickoff(inputs=inputs)
            
            return result
        
        except Exception as e:
            if isinstance(e, FrameworkExecutionError):
                raise
            raise FrameworkExecutionError(
                f"crew execution '{crew_name}'",
                self.framework_name,
                e
            )
    
    def get_agent(self, name: str) -> Optional[Agent]:
        """Get agent by name.
        
        Args:
            name: Agent name
            
        Returns:
            Agent instance or None
        """
        return self.agents.get(name)
    
    def get_task(self, name: str) -> Optional[Task]:
        """Get task by name.
        
        Args:
            name: Task name
            
        Returns:
            Task instance or None
        """
        return self.tasks.get(name)
    
    def get_crew(self, name: str) -> Optional[Crew]:
        """Get crew by name.
        
        Args:
            name: Crew name
            
        Returns:
            Crew instance or None
        """
        return self.crews.get(name)
    
    def list_agents(self) -> List[str]:
        """List all created agents.
        
        Returns:
            List of agent names
        """
        return list(self.agents.keys())
    
    def list_tasks(self) -> List[str]:
        """List all created tasks.
        
        Returns:
            List of task names
        """
        return list(self.tasks.keys())
    
    def list_crews(self) -> List[str]:
        """List all created crews.
        
        Returns:
            List of crew names
        """
        return list(self.crews.keys())
    
    def clear(self) -> None:
        """Clear all created agents, tasks, and crews."""
        self.agents.clear()
        self.tasks.clear()
        self.crews.clear()
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"{self.__class__.__name__}("
            f"providers={len(self.list_providers())}, "
            f"agents={len(self.agents)}, "
            f"tasks={len(self.tasks)}, "
            f"crews={len(self.crews)}"
            f")"
        )
