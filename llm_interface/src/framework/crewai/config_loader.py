"""CrewAI Configuration Loading and Building.

This module provides utilities for loading and building CrewAI agent, task,
and crew configurations from YAML files or programmatic construction.
"""

import json
import yaml
from typing import Dict, Any, List, Optional, Type
from dataclasses import dataclass, field, asdict
from pathlib import Path

try:
    from crewai import Process
except ImportError:
    Process = None  # type: ignore


@dataclass
class AgentConfig:
    """Configuration for a CrewAI Agent.
    
    Attributes:
        name: Unique agent identifier
        role: Agent's role in the crew
        goal: Agent's primary objective
        backstory: Agent's background story
        tools: List of tool names available to agent
        provider: LLM provider to use
        memory: Enable agent memory
        verbose: Enable verbose output
        allow_delegation: Allow agent to delegate tasks
        allow_code_execution: Allow code execution
        metadata: Additional arbitrary data
    """
    name: str
    role: str
    goal: str
    backstory: str
    tools: List[str] = field(default_factory=list)
    provider: str = "openai"
    memory: bool = False
    verbose: bool = True
    allow_delegation: bool = False
    allow_code_execution: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.
        
        Returns:
            Dict representation
        """
        return asdict(self)


@dataclass
class TaskConfig:
    """Configuration for a CrewAI Task.
    
    Attributes:
        name: Unique task identifier
        description: Task description
        expected_output: Expected output description
        agent_name: Name of agent to execute task
        tools: List of task-specific tool names
        async_execution: Execute asynchronously
        human_input: Require human input
        markdown: Format output as markdown
        context_tasks: Names of dependent tasks
        metadata: Additional arbitrary data
    """
    name: str
    description: str
    expected_output: str
    agent_name: str
    tools: List[str] = field(default_factory=list)
    async_execution: bool = False
    human_input: bool = False
    markdown: bool = False
    context_tasks: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.
        
        Returns:
            Dict representation
        """
        return asdict(self)


@dataclass
class CrewConfig:
    """Configuration for a CrewAI Crew.
    
    Attributes:
        name: Unique crew identifier
        agents: List of agent names
        tasks: List of task names
        process: Execution process ("sequential" or "hierarchical")
        verbose: Enable verbose output
        memory: Enable crew memory
        cache: Enable result caching
        metadata: Additional arbitrary data
    """
    name: str
    agents: List[str]
    tasks: List[str]
    process: str = "sequential"
    verbose: bool = True
    memory: bool = False
    cache: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.
        
        Returns:
            Dict representation
        """
        return asdict(self)


class CrewAIConfigBuilder:
    """Builder for programmatic CrewAI configuration creation.
    
    Provides a fluent interface for building agents, tasks, and crews
    without using YAML files.
    """
    
    def __init__(self):
        """Initialize builder."""
        self.agents: Dict[str, AgentConfig] = {}
        self.tasks: Dict[str, TaskConfig] = {}
        self.crews: Dict[str, CrewConfig] = {}
    
    def add_agent(
        self,
        name: str,
        role: str,
        goal: str,
        backstory: str,
        tools: Optional[List[str]] = None,
        provider: str = "openai",
        memory: bool = False,
        verbose: bool = True,
        allow_delegation: bool = False,
        **metadata
    ) -> 'CrewAIConfigBuilder':
        """Add agent configuration.
        
        Args:
            name: Agent identifier
            role: Agent's role
            goal: Agent's goal
            backstory: Agent's backstory
            tools: Optional list of tools
            provider: LLM provider
            memory: Enable memory
            verbose: Enable verbosity
            allow_delegation: Allow delegation
            **metadata: Additional metadata
            
        Returns:
            Self for chaining
        """
        self.agents[name] = AgentConfig(
            name=name,
            role=role,
            goal=goal,
            backstory=backstory,
            tools=tools or [],
            provider=provider,
            memory=memory,
            verbose=verbose,
            allow_delegation=allow_delegation,
            metadata=metadata
        )
        return self
    
    def add_task(
        self,
        name: str,
        description: str,
        expected_output: str,
        agent_name: str,
        tools: Optional[List[str]] = None,
        async_execution: bool = False,
        human_input: bool = False,
        markdown: bool = False,
        context_tasks: Optional[List[str]] = None,
        **metadata
    ) -> 'CrewAIConfigBuilder':
        """Add task configuration.
        
        Args:
            name: Task identifier
            description: Task description
            expected_output: Expected output description
            agent_name: Name of agent
            tools: Optional task-specific tools
            async_execution: Execute asynchronously
            human_input: Require human input
            markdown: Format as markdown
            context_tasks: Dependent task names
            **metadata: Additional metadata
            
        Returns:
            Self for chaining
        """
        self.tasks[name] = TaskConfig(
            name=name,
            description=description,
            expected_output=expected_output,
            agent_name=agent_name,
            tools=tools or [],
            async_execution=async_execution,
            human_input=human_input,
            markdown=markdown,
            context_tasks=context_tasks or [],
            metadata=metadata
        )
        return self
    
    def add_crew(
        self,
        name: str,
        agents: List[str],
        tasks: List[str],
        process: str = "sequential",
        verbose: bool = True,
        memory: bool = False,
        cache: bool = True,
        **metadata
    ) -> 'CrewAIConfigBuilder':
        """Add crew configuration.
        
        Args:
            name: Crew identifier
            agents: List of agent names
            tasks: List of task names
            process: Execution process
            verbose: Enable verbosity
            memory: Enable memory
            cache: Enable caching
            **metadata: Additional metadata
            
        Returns:
            Self for chaining
        """
        # Validate process
        if process.lower() not in ("sequential", "hierarchical"):
            raise ValueError(f"Invalid process: {process}")
        
        # Validate agents exist
        for agent in agents:
            if agent not in self.agents:
                raise ValueError(f"Agent '{agent}' not configured")
        
        # Validate tasks exist and reference valid agents
        for task in tasks:
            if task not in self.tasks:
                raise ValueError(f"Task '{task}' not configured")
            task_config = self.tasks[task]
            if task_config.agent_name not in self.agents:
                raise ValueError(
                    f"Task '{task}' references unknown agent '{task_config.agent_name}'"
                )
        
        self.crews[name] = CrewConfig(
            name=name,
            agents=agents,
            tasks=tasks,
            process=process,
            verbose=verbose,
            memory=memory,
            cache=cache,
            metadata=metadata
        )
        return self
    
    def build(self) -> Dict[str, Any]:
        """Build configuration dictionary.
        
        Returns:
            Dict with agents, tasks, and crews
        """
        return {
            'agents': {name: config.to_dict() for name, config in self.agents.items()},
            'tasks': {name: config.to_dict() for name, config in self.tasks.items()},
            'crews': {name: config.to_dict() for name, config in self.crews.items()},
        }


class CrewAIConfigLoader:
    """Load CrewAI configurations from files and dictionaries."""
    
    @staticmethod
    def load_from_file(filepath: str) -> Dict[str, Any]:
        """Load configuration from YAML or JSON file.
        
        Args:
            filepath: Path to configuration file
            
        Returns:
            Configuration dictionary
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
        """
        path = Path(filepath)
        
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")
        
        suffix = path.suffix.lower()
        
        with open(filepath, 'r') as f:
            if suffix in ('.yaml', '.yml'):
                config = yaml.safe_load(f)
            elif suffix == '.json':
                config = json.load(f)
            else:
                raise ValueError(f"Unsupported file format: {suffix}")
        
        if not isinstance(config, dict):
            raise ValueError("Configuration must be a dictionary")
        
        return config
    
    @staticmethod
    def load_from_dict(data: Dict[str, Any]) -> Dict[str, Any]:
        """Load configuration from dictionary.
        
        Args:
            data: Configuration dictionary
            
        Returns:
            Validated configuration
            
        Raises:
            ValueError: If configuration is invalid
        """
        if not isinstance(data, dict):
            raise ValueError("Configuration must be a dictionary")
        
        # Validate structure
        if 'agents' in data and not isinstance(data['agents'], dict):
            raise ValueError("agents must be a dictionary")
        
        if 'tasks' in data and not isinstance(data['tasks'], dict):
            raise ValueError("tasks must be a dictionary")
        
        if 'crews' in data and not isinstance(data['crews'], dict):
            raise ValueError("crews must be a dictionary")
        
        return data
    
    @staticmethod
    def save_to_file(config: Dict[str, Any], filepath: str) -> None:
        """Save configuration to file.
        
        Args:
            config: Configuration dictionary
            filepath: Output file path
            
        Raises:
            ValueError: If file format is unsupported
        """
        path = Path(filepath)
        suffix = path.suffix.lower()
        
        # Create parent directory if needed
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            if suffix in ('.yaml', '.yml'):
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            elif suffix == '.json':
                json.dump(config, f, indent=2)
            else:
                raise ValueError(f"Unsupported file format: {suffix}")
    
    @staticmethod
    def to_yaml(config: Dict[str, Any]) -> str:
        """Convert configuration to YAML string.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            YAML string representation
        """
        return yaml.dump(config, default_flow_style=False, sort_keys=False)
    
    @staticmethod
    def to_json(config: Dict[str, Any]) -> str:
        """Convert configuration to JSON string.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            JSON string representation
        """
        return json.dumps(config, indent=2)
