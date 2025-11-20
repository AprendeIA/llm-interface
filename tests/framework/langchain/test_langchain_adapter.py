"""Tests for LangChain adapter and workflows."""

import pytest
from unittest.mock import Mock, MagicMock, patch
from typing import List

from llm_interface.src.framework.langchain.adapter import LangChainAdapter
from llm_interface.src.framework.langchain.workflows import (
    ChainWorkflow,
    ConditionalWorkflow,
    ParallelWorkflow,
    WorkflowConfig,
    WorkflowType,
    WorkflowFactory,
)
from llm_interface.src.framework.langchain.config_loader import (
    LangChainConfigLoader,
    LangChainConfigBuilder,
    LangChainWorkflowConfig,
    LangChainNodeConfig,
)
from llm_interface.src.framework.exceptions import (
    NotAvailableError,
    ModelCreationError,
    ConfigurationError,
)


class TestLangChainAdapter:
    """Test suite for LangChainAdapter."""
    
    @pytest.fixture
    def mock_manager(self):
        """Create a mock LLMManager."""
        manager = Mock()
        manager.providers = {"openai": Mock(), "anthropic": Mock()}
        manager.register_framework = Mock()
        manager.get_framework = Mock(return_value=None)
        manager.list_frameworks = Mock(return_value=[])
        manager.unregister_framework = Mock()
        return manager
    
    @pytest.fixture
    def adapter(self, mock_manager):
        """Create LangChainAdapter instance."""
        return LangChainAdapter(mock_manager)
    
    def test_init(self, mock_manager):
        """Test adapter initialization."""
        adapter = LangChainAdapter(mock_manager)
        assert adapter.manager == mock_manager
        assert adapter._chains == {}
        assert adapter._models == {}
    
    def test_framework_name(self, adapter):
        """Test framework name property."""
        assert adapter.framework_name == "LangChain"
    
    def test_validate_config_valid(self, adapter):
        """Test config validation with valid setup."""
        adapter.list_providers = Mock(return_value=["openai"])
        adapter.has_provider = Mock(return_value=True)
        
        is_valid, message = adapter.validate_config()
        assert is_valid
        assert message == ""
    
    def test_validate_config_no_manager(self):
        """Test validation fails without manager."""
        adapter = LangChainAdapter(None)
        is_valid, message = adapter.validate_config()
        assert not is_valid
        assert "Manager not configured" in message
    
    def test_validate_config_no_providers(self, adapter):
        """Test validation fails with no providers."""
        adapter.list_providers = Mock(return_value=[])
        
        is_valid, message = adapter.validate_config()
        assert not is_valid
        assert "No LLM providers available" in message
    
    def test_extract_variables(self):
        """Test variable extraction from templates."""
        template = "Summarize this: {text}, Length: {length}"
        variables = LangChainAdapter._extract_variables(template)
        assert set(variables) == {"text", "length"}
    
    def test_extract_variables_no_vars(self):
        """Test template with no variables."""
        template = "Just a static prompt"
        variables = LangChainAdapter._extract_variables(template)
        assert variables == []
    
    def test_clear_cache(self, adapter):
        """Test cache clearing."""
        adapter._models = {"test": Mock()}
        adapter._chains = {"test": Mock()}
        
        adapter.clear_cache()
        assert adapter._models == {}
        assert adapter._chains == {}


class TestWorkflows:
    """Test suite for workflow classes."""
    
    @pytest.fixture
    def config(self):
        """Create workflow config."""
        return WorkflowConfig(
            name="test_workflow",
            provider="openai",
            workflow_type=WorkflowType.CHAIN
        )
    
    def test_workflow_config_validation(self):
        """Test WorkflowConfig validation."""
        with pytest.raises(ConfigurationError):
            WorkflowConfig(
                name="",
                provider="openai",
                workflow_type=WorkflowType.CHAIN
            )
    
    def test_chain_workflow_add_node(self, config):
        """Test adding nodes to chain workflow."""
        workflow = ChainWorkflow(config)
        func1 = Mock()
        func2 = Mock()
        
        workflow.add_node("node1", func1)
        workflow.add_node("node2", func2)
        
        assert "node1" in workflow._nodes
        assert "node2" in workflow._nodes
    
    def test_chain_workflow_add_duplicate_node(self, config):
        """Test adding duplicate node raises error."""
        workflow = ChainWorkflow(config)
        func = Mock()
        
        workflow.add_node("node1", func)
        
        with pytest.raises(ConfigurationError):
            workflow.add_node("node1", func)
    
    def test_chain_workflow_add_edge(self, config):
        """Test adding edges to chain workflow."""
        workflow = ChainWorkflow(config)
        func = Mock()
        
        workflow.add_node("node1", func)
        workflow.add_node("node2", func)
        workflow.add_edge("node1", "node2")
        
        assert ("node1", "node2") in workflow._edges
    
    def test_chain_workflow_add_invalid_edge(self, config):
        """Test adding edge with non-existent node."""
        workflow = ChainWorkflow(config)
        func = Mock()
        
        workflow.add_node("node1", func)
        
        with pytest.raises(ConfigurationError):
            workflow.add_edge("node1", "nonexistent")
    
    def test_chain_workflow_build(self, config):
        """Test building chain workflow."""
        workflow = ChainWorkflow(config)
        func1 = Mock()
        func2 = Mock()
        
        workflow.add_node("node1", func1)
        workflow.add_node("node2", func2)
        
        result = workflow.build()
        
        assert isinstance(result, dict)
        assert "node1" in result
        assert "node2" in result
    
    def test_conditional_workflow(self, config):
        """Test conditional workflow."""
        config.workflow_type = WorkflowType.CONDITIONAL
        workflow = ConditionalWorkflow(config)
        func1 = Mock()
        func2 = Mock()
        condition = Mock(return_value=True)
        
        workflow.add_node("route", func1)
        workflow.add_node("target", func2)
        workflow.add_edge("route", "target", condition)
        
        result = workflow.build()
        assert "nodes" in result
        assert "conditions" in result
    
    def test_parallel_workflow(self, config):
        """Test parallel workflow."""
        config.workflow_type = WorkflowType.PARALLEL
        workflow = ParallelWorkflow(config)
        func1 = Mock()
        func2 = Mock()
        func3 = Mock()
        
        workflow.add_node("input", func1)
        workflow.add_node("process1", func2)
        workflow.add_node("process2", func3)
        
        result = workflow.build()
        assert "nodes" in result
        assert "parallel_groups" in result
    
    def test_workflow_factory_create_chain(self):
        """Test WorkflowFactory creating chain workflow."""
        workflow = WorkflowFactory.create(
            WorkflowType.CHAIN,
            "test",
            "openai"
        )
        assert isinstance(workflow, ChainWorkflow)
    
    def test_workflow_factory_create_conditional(self):
        """Test WorkflowFactory creating conditional workflow."""
        workflow = WorkflowFactory.create(
            WorkflowType.CONDITIONAL,
            "test",
            "openai"
        )
        assert isinstance(workflow, ConditionalWorkflow)
    
    def test_workflow_factory_invalid_type(self):
        """Test WorkflowFactory with invalid type."""
        with pytest.raises(ConfigurationError):
            WorkflowFactory.create(
                "invalid_type",  # type: ignore
                "test",
                "openai"
            )


class TestLangChainConfigLoader:
    """Test suite for configuration loading."""
    
    def test_load_from_dict_valid(self):
        """Test loading config from valid dictionary."""
        data = {
            "name": "test_workflow",
            "description": "Test workflow",
            "default_provider": "openai",
            "nodes": [
                {
                    "name": "node1",
                    "type": "prompt",
                    "provider": "openai",
                    "prompt_template": "Test: {text}"
                }
            ],
            "edges": [],
            "metadata": {}
        }
        
        config = LangChainConfigLoader.load_from_dict(data)
        assert config.name == "test_workflow"
        assert len(config.nodes) == 1
        assert config.nodes[0].name == "node1"
    
    def test_load_from_dict_invalid_no_name(self):
        """Test loading config without name."""
        data = {
            "description": "No name",
            "nodes": []
        }
        
        with pytest.raises(ConfigurationError):
            LangChainConfigLoader.load_from_dict(data)
    
    def test_load_from_dict_invalid_edge(self):
        """Test loading config with invalid edge."""
        data = {
            "name": "test",
            "nodes": [
                {"name": "node1", "type": "prompt", "provider": "openai"}
            ],
            "edges": [["node1", "nonexistent"]]
        }
        
        with pytest.raises(ConfigurationError):
            LangChainConfigLoader.load_from_dict(data)
    
    def test_to_dict(self):
        """Test converting config to dictionary."""
        config = LangChainWorkflowConfig(
            name="test",
            description="Test config",
            nodes=[
                LangChainNodeConfig(
                    name="node1",
                    type="prompt",
                    provider="openai",
                    prompt_template="Test"
                )
            ]
        )
        
        data = LangChainConfigLoader.to_dict(config)
        assert data["name"] == "test"
        assert len(data["nodes"]) == 1
    
    def test_config_builder_basic(self):
        """Test basic config builder."""
        builder = LangChainConfigBuilder(
            "test_workflow",
            "Test description"
        )
        
        builder.add_node("node1", "prompt", prompt_template="Test: {text}")
        builder.add_node("node2", "chain")
        builder.add_edge("node1", "node2")
        
        config = builder.build()
        assert config.name == "test_workflow"
        assert len(config.nodes) == 2
        assert len(config.edges) == 1
    
    def test_config_builder_duplicate_node(self):
        """Test builder with duplicate node."""
        builder = LangChainConfigBuilder("test")
        builder.add_node("node1", "prompt")
        
        with pytest.raises(ConfigurationError):
            builder.add_node("node1", "prompt")
    
    def test_config_builder_invalid_edge(self):
        """Test builder with invalid edge."""
        builder = LangChainConfigBuilder("test")
        builder.add_node("node1", "prompt")
        
        with pytest.raises(ConfigurationError):
            builder.add_edge("node1", "nonexistent")
    
    def test_config_builder_chaining(self):
        """Test builder method chaining."""
        config = (LangChainConfigBuilder("test")
                 .add_node("node1", "prompt")
                 .add_node("node2", "chain")
                 .add_edge("node1", "node2")
                 .set_metadata("key", "value")
                 .build())
        
        assert config.name == "test"
        assert config.metadata["key"] == "value"


class TestLangChainNodeConfig:
    """Test suite for node configuration."""
    
    def test_node_config_creation(self):
        """Test creating node config."""
        node = LangChainNodeConfig(
            name="test_node",
            type="prompt",
            provider="openai",
            prompt_template="Test: {text}"
        )
        
        assert node.name == "test_node"
        assert node.type == "prompt"
        assert node.provider == "openai"
    
    def test_node_config_with_metadata(self):
        """Test node config with metadata."""
        node = LangChainNodeConfig(
            name="test",
            type="prompt",
            provider="openai",
            metadata={"temperature": 0.7}
        )
        
        assert node.metadata["temperature"] == 0.7


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
