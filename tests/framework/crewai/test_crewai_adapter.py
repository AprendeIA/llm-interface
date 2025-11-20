"""Tests for CrewAI Adapter.

Comprehensive test suite for CrewAI adapter functionality including
agent creation, task management, crew orchestration, and configuration loading.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, call
from typing import Dict, Any

from llm_interface.src.manager import LLMManager
from llm_interface.src.core.config import LLMConfig, ProviderType
from llm_interface.src.framework.crewai.adapter import CrewAIAdapter
from llm_interface.src.framework.crewai.config_loader import (
    AgentConfig,
    TaskConfig,
    CrewConfig,
    CrewAIConfigBuilder,
    CrewAIConfigLoader,
)
from llm_interface.src.framework.exceptions import (
    FrameworkConfigurationError,
    FrameworkExecutionError,
    FrameworkModelCreationError,
)


@pytest.fixture
def mock_manager():
    """Create mock LLMManager."""
    manager = Mock(spec=LLMManager)
    manager.list_providers.return_value = ["openai", "anthropic"]
    
    mock_model = Mock()
    manager.get_chat_model.return_value = mock_model
    manager.get_embeddings.return_value = None
    
    return manager


@pytest.fixture
def adapter(mock_manager):
    """Create CrewAI adapter."""
    return CrewAIAdapter(mock_manager)


class TestCrewAIAdapterBasics:
    """Test basic adapter functionality."""
    
    def test_adapter_initialization(self, mock_manager):
        """Test adapter can be initialized."""
        adapter = CrewAIAdapter(mock_manager)
        assert adapter is not None
        assert adapter.llm_manager == mock_manager
    
    def test_framework_name(self, adapter):
        """Test framework name property."""
        assert adapter.framework_name == "crewai"
    
    def test_framework_version(self, adapter):
        """Test framework version property."""
        assert adapter.framework_version == "0.1.0"
    
    def test_validate_config_valid(self, adapter):
        """Test validation of valid config."""
        config = {
            'default_provider': 'openai',
            'verbose': True,
        }
        assert adapter.validate_config(config) is True
    
    def test_validate_config_invalid_type(self, adapter):
        """Test validation rejects non-dict config."""
        assert adapter.validate_config("not a dict") is False
        assert adapter.validate_config([]) is False
    
    def test_validate_config_invalid_keys(self, adapter):
        """Test validation rejects unknown keys."""
        config = {
            'invalid_key': 'value',
            'another_invalid': True,
        }
        assert adapter.validate_config(config) is False
    
    def test_validate_config_empty(self, adapter):
        """Test validation accepts empty config."""
        assert adapter.validate_config({}) is True
    
    def test_repr(self, adapter):
        """Test string representation."""
        repr_str = repr(adapter)
        assert "CrewAIAdapter" in repr_str
        assert "providers=2" in repr_str


class TestAgentCreation:
    """Test agent creation."""
    
    def test_create_agent_basic(self, adapter):
        """Test creating basic agent."""
        agent = adapter.create_agent(
            name="researcher",
            role="Research Analyst",
            goal="Find information",
            backstory="Expert researcher"
        )
        
        assert agent is not None
        assert "researcher" in adapter.list_agents()
    
    def test_create_agent_with_tools(self, adapter):
        """Test creating agent with tools."""
        tools = [Mock(), Mock()]
        agent = adapter.create_agent(
            name="researcher",
            role="Researcher",
            goal="Research",
            backstory="Expert",
            tools=tools
        )
        
        assert agent is not None
    
    def test_create_agent_with_provider(self, adapter):
        """Test creating agent with specific provider."""
        agent = adapter.create_agent(
            name="researcher",
            role="Researcher",
            goal="Research",
            backstory="Expert",
            provider_name="openai"
        )
        
        assert agent is not None
        adapter.llm_manager.get_chat_model.assert_called_with("openai")
    
    def test_create_agent_no_provider(self, adapter):
        """Test creating agent uses default provider."""
        adapter.get_default_provider = Mock(return_value="openai")
        
        agent = adapter.create_agent(
            name="researcher",
            role="Researcher",
            goal="Research",
            backstory="Expert"
        )
        
        assert agent is not None
        adapter.get_default_provider.assert_called_once()
    
    def test_create_agent_no_providers_available(self, adapter):
        """Test error when no providers available."""
        adapter.get_default_provider = Mock(return_value=None)
        
        with pytest.raises(FrameworkModelCreationError):
            adapter.create_agent(
                name="researcher",
                role="Researcher",
                goal="Research",
                backstory="Expert"
            )
    
    def test_create_agent_invalid_provider(self, adapter):
        """Test error with invalid provider."""
        with pytest.raises(FrameworkModelCreationError):
            adapter.create_agent(
                name="researcher",
                role="Researcher",
                goal="Research",
                backstory="Expert",
                provider_name="invalid_provider"
            )
    
    def test_create_agent_with_options(self, adapter):
        """Test creating agent with various options."""
        agent = adapter.create_agent(
            name="researcher",
            role="Researcher",
            goal="Research",
            backstory="Expert",
            memory=True,
            verbose=False,
            allow_delegation=True,
            max_iter=30,
            max_execution_time=600,
            allow_code_execution=True
        )
        
        assert agent is not None
    
    def test_list_agents(self, adapter):
        """Test listing agents."""
        adapter.create_agent(
            name="agent1",
            role="Role 1",
            goal="Goal 1",
            backstory="Story 1"
        )
        adapter.create_agent(
            name="agent2",
            role="Role 2",
            goal="Goal 2",
            backstory="Story 2"
        )
        
        agents = adapter.list_agents()
        assert len(agents) == 2
        assert "agent1" in agents
        assert "agent2" in agents
    
    def test_get_agent(self, adapter):
        """Test retrieving agent."""
        adapter.create_agent(
            name="researcher",
            role="Researcher",
            goal="Research",
            backstory="Expert"
        )
        
        agent = adapter.get_agent("researcher")
        assert agent is not None
    
    def test_get_agent_not_found(self, adapter):
        """Test retrieving non-existent agent."""
        agent = adapter.get_agent("nonexistent")
        assert agent is None


class TestTaskCreation:
    """Test task creation."""
    
    def test_create_task_with_agent_object(self, adapter):
        """Test creating task with agent object."""
        agent = adapter.create_agent(
            name="researcher",
            role="Researcher",
            goal="Research",
            backstory="Expert"
        )
        
        task = adapter.create_task(
            name="research_task",
            description="Research AI",
            expected_output="Report",
            agent=agent
        )
        
        assert task is not None
        assert "research_task" in adapter.list_tasks()
    
    def test_create_task_with_agent_name(self, adapter):
        """Test creating task with agent name."""
        adapter.create_agent(
            name="researcher",
            role="Researcher",
            goal="Research",
            backstory="Expert"
        )
        
        task = adapter.create_task(
            name="research_task",
            description="Research AI",
            expected_output="Report",
            agent_name="researcher"
        )
        
        assert task is not None
    
    def test_create_task_no_agent(self, adapter):
        """Test error when no agent specified."""
        with pytest.raises(FrameworkExecutionError):
            adapter.create_task(
                name="research_task",
                description="Research AI",
                expected_output="Report"
            )
    
    def test_create_task_agent_not_found(self, adapter):
        """Test error with non-existent agent."""
        with pytest.raises(FrameworkExecutionError):
            adapter.create_task(
                name="research_task",
                description="Research AI",
                expected_output="Report",
                agent_name="nonexistent"
            )
    
    def test_create_task_with_options(self, adapter):
        """Test creating task with various options."""
        adapter.create_agent(
            name="researcher",
            role="Researcher",
            goal="Research",
            backstory="Expert"
        )
        
        task = adapter.create_task(
            name="research_task",
            description="Research AI",
            expected_output="Report",
            agent_name="researcher",
            tools=[Mock()],
            async_execution=True,
            human_input=True,
            markdown=True,
            output_file="output.md"
        )
        
        assert task is not None
    
    def test_create_task_with_context(self, adapter):
        """Test creating task with context (dependencies)."""
        adapter.create_agent(
            name="researcher",
            role="Researcher",
            goal="Research",
            backstory="Expert"
        )
        
        task1 = adapter.create_task(
            name="task1",
            description="First task",
            expected_output="Output 1",
            agent_name="researcher"
        )
        
        task2 = adapter.create_task(
            name="task2",
            description="Second task",
            expected_output="Output 2",
            agent_name="researcher",
            context=[task1]
        )
        
        assert task2 is not None
    
    def test_list_tasks(self, adapter):
        """Test listing tasks."""
        adapter.create_agent(
            name="researcher",
            role="Researcher",
            goal="Research",
            backstory="Expert"
        )
        
        adapter.create_task(
            name="task1",
            description="Task 1",
            expected_output="Output 1",
            agent_name="researcher"
        )
        adapter.create_task(
            name="task2",
            description="Task 2",
            expected_output="Output 2",
            agent_name="researcher"
        )
        
        tasks = adapter.list_tasks()
        assert len(tasks) == 2
        assert "task1" in tasks
        assert "task2" in tasks


class TestCrewCreation:
    """Test crew creation and execution."""
    
    def test_create_crew_basic(self, adapter):
        """Test creating basic crew."""
        agent = adapter.create_agent(
            name="researcher",
            role="Researcher",
            goal="Research",
            backstory="Expert"
        )
        
        task = adapter.create_task(
            name="research_task",
            description="Research",
            expected_output="Report",
            agent=agent
        )
        
        crew = adapter.create_crew(
            name="research_crew",
            agents=["researcher"],
            tasks=["research_task"]
        )
        
        assert crew is not None
        assert "research_crew" in adapter.list_crews()
    
    def test_create_crew_sequential_process(self, adapter):
        """Test crew with sequential process."""
        adapter.create_agent(
            name="agent1",
            role="Role 1",
            goal="Goal 1",
            backstory="Story 1"
        )
        adapter.create_task(
            name="task1",
            description="Task 1",
            expected_output="Output 1",
            agent_name="agent1"
        )
        
        crew = adapter.create_crew(
            name="crew1",
            agents=["agent1"],
            tasks=["task1"],
            process="sequential"
        )
        
        assert crew is not None
    
    def test_create_crew_hierarchical_process(self, adapter):
        """Test crew with hierarchical process."""
        adapter.create_agent(
            name="agent1",
            role="Role 1",
            goal="Goal 1",
            backstory="Story 1"
        )
        adapter.create_task(
            name="task1",
            description="Task 1",
            expected_output="Output 1",
            agent_name="agent1"
        )
        
        crew = adapter.create_crew(
            name="crew1",
            agents=["agent1"],
            tasks=["task1"],
            process="hierarchical"
        )
        
        assert crew is not None
    
    def test_create_crew_invalid_agent(self, adapter):
        """Test error with invalid agent."""
        with pytest.raises(FrameworkExecutionError):
            adapter.create_crew(
                name="crew1",
                agents=["nonexistent"],
                tasks=[]
            )
    
    def test_create_crew_invalid_task(self, adapter):
        """Test error with invalid task."""
        adapter.create_agent(
            name="agent1",
            role="Role 1",
            goal="Goal 1",
            backstory="Story 1"
        )
        
        with pytest.raises(FrameworkExecutionError):
            adapter.create_crew(
                name="crew1",
                agents=["agent1"],
                tasks=["nonexistent"]
            )
    
    def test_get_crew(self, adapter):
        """Test retrieving crew."""
        adapter.create_agent(
            name="agent1",
            role="Role 1",
            goal="Goal 1",
            backstory="Story 1"
        )
        adapter.create_task(
            name="task1",
            description="Task 1",
            expected_output="Output 1",
            agent_name="agent1"
        )
        
        adapter.create_crew(
            name="crew1",
            agents=["agent1"],
            tasks=["task1"]
        )
        
        crew = adapter.get_crew("crew1")
        assert crew is not None
    
    def test_list_crews(self, adapter):
        """Test listing crews."""
        adapter.create_agent(
            name="agent1",
            role="Role 1",
            goal="Goal 1",
            backstory="Story 1"
        )
        adapter.create_task(
            name="task1",
            description="Task 1",
            expected_output="Output 1",
            agent_name="agent1"
        )
        
        adapter.create_crew(
            name="crew1",
            agents=["agent1"],
            tasks=["task1"]
        )
        adapter.create_crew(
            name="crew2",
            agents=["agent1"],
            tasks=["task1"]
        )
        
        crews = adapter.list_crews()
        assert len(crews) == 2
        assert "crew1" in crews
        assert "crew2" in crews


class TestConfigBuilder:
    """Test CrewAI config builder."""
    
    def test_builder_add_agent(self):
        """Test adding agent to builder."""
        builder = CrewAIConfigBuilder()
        builder.add_agent(
            name="researcher",
            role="Researcher",
            goal="Research",
            backstory="Expert"
        )
        
        config = builder.build()
        assert "researcher" in config['agents']
    
    def test_builder_add_task(self):
        """Test adding task to builder."""
        builder = CrewAIConfigBuilder()
        builder.add_agent(
            name="researcher",
            role="Researcher",
            goal="Research",
            backstory="Expert"
        )
        builder.add_task(
            name="research_task",
            description="Research",
            expected_output="Report",
            agent_name="researcher"
        )
        
        config = builder.build()
        assert "research_task" in config['tasks']
    
    def test_builder_add_crew(self):
        """Test adding crew to builder."""
        builder = CrewAIConfigBuilder()
        builder.add_agent(
            name="researcher",
            role="Researcher",
            goal="Research",
            backstory="Expert"
        )
        builder.add_task(
            name="research_task",
            description="Research",
            expected_output="Report",
            agent_name="researcher"
        )
        builder.add_crew(
            name="research_crew",
            agents=["researcher"],
            tasks=["research_task"]
        )
        
        config = builder.build()
        assert "research_crew" in config['crews']
    
    def test_builder_invalid_process(self):
        """Test builder rejects invalid process."""
        builder = CrewAIConfigBuilder()
        builder.add_agent(
            name="researcher",
            role="Researcher",
            goal="Research",
            backstory="Expert"
        )
        builder.add_task(
            name="task1",
            description="Task",
            expected_output="Output",
            agent_name="researcher"
        )
        
        with pytest.raises(ValueError):
            builder.add_crew(
                name="crew1",
                agents=["researcher"],
                tasks=["task1"],
                process="invalid_process"
            )
    
    def test_builder_invalid_agent_reference(self):
        """Test builder rejects unknown agent in crew."""
        builder = CrewAIConfigBuilder()
        
        with pytest.raises(ValueError):
            builder.add_crew(
                name="crew1",
                agents=["nonexistent"],
                tasks=[]
            )
    
    def test_builder_chaining(self):
        """Test builder method chaining."""
        builder = (CrewAIConfigBuilder()
            .add_agent("researcher", "Researcher", "Research", "Expert")
            .add_task("task1", "Research", "Report", "researcher")
            .add_crew("crew1", ["researcher"], ["task1"]))
        
        config = builder.build()
        assert "researcher" in config['agents']
        assert "task1" in config['tasks']
        assert "crew1" in config['crews']


class TestConfigLoader:
    """Test CrewAI config loader."""
    
    def test_loader_load_from_dict(self):
        """Test loading from dictionary."""
        config_dict = {
            'agents': {
                'researcher': {
                    'name': 'researcher',
                    'role': 'Researcher',
                    'goal': 'Research',
                    'backstory': 'Expert',
                }
            },
            'tasks': {},
            'crews': {}
        }
        
        config = CrewAIConfigLoader.load_from_dict(config_dict)
        assert config is not None
        assert 'agents' in config
    
    def test_loader_invalid_dict(self):
        """Test loader rejects non-dict."""
        with pytest.raises(ValueError):
            CrewAIConfigLoader.load_from_dict("not a dict")
    
    def test_loader_yaml_to_yaml(self):
        """Test YAML conversion."""
        config = {
            'agents': {
                'researcher': {
                    'role': 'Researcher',
                    'goal': 'Research',
                }
            }
        }
        
        yaml_str = CrewAIConfigLoader.to_yaml(config)
        assert 'researcher' in yaml_str
        assert 'Researcher' in yaml_str
    
    def test_loader_yaml_to_json(self):
        """Test JSON conversion."""
        config = {
            'agents': {
                'researcher': {
                    'role': 'Researcher',
                    'goal': 'Research',
                }
            }
        }
        
        json_str = CrewAIConfigLoader.to_json(config)
        assert 'researcher' in json_str
        assert 'Researcher' in json_str


class TestClearOperations:
    """Test clearing operations."""
    
    def test_clear_agents(self, adapter):
        """Test clearing all agents."""
        adapter.create_agent(
            name="agent1",
            role="Role 1",
            goal="Goal 1",
            backstory="Story 1"
        )
        adapter.create_agent(
            name="agent2",
            role="Role 2",
            goal="Goal 2",
            backstory="Story 2"
        )
        
        assert len(adapter.list_agents()) == 2
        
        adapter.agents.clear()
        assert len(adapter.list_agents()) == 0
    
    def test_clear_all(self, adapter):
        """Test clearing everything."""
        adapter.create_agent(
            name="agent1",
            role="Role 1",
            goal="Goal 1",
            backstory="Story 1"
        )
        adapter.create_task(
            name="task1",
            description="Task 1",
            expected_output="Output 1",
            agent_name="agent1"
        )
        
        assert len(adapter.list_agents()) == 1
        assert len(adapter.list_tasks()) == 1
        
        adapter.clear()
        
        assert len(adapter.list_agents()) == 0
        assert len(adapter.list_tasks()) == 0
