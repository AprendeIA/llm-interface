"""
Tests for AutoGen adapter.

Tests the AutoGen framework integration with llm_interface providers.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from llm_interface.src.manager import LLMManager
from llm_interface.src.core.config import LLMConfig, ProviderType
from llm_interface.src.framework.exceptions import FrameworkConfigurationError

# Skip all tests if AutoGen is not available
try:
    from llm_interface.src.framework.autogen import AutoGenAdapter
    AUTOGEN_AVAILABLE = True
except ImportError:
    AUTOGEN_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not AUTOGEN_AVAILABLE,
    reason="AutoGen not installed"
)


class TestAutoGenAdapter:
    """Test suite for AutoGenAdapter."""
    
    def test_adapter_initialization(self):
        """Test adapter initializes correctly."""
        manager = LLMManager()
        adapter = AutoGenAdapter(manager)
        
        assert adapter.framework_name == "autogen"
        assert adapter.llm_manager is manager
        assert isinstance(adapter.agents, dict)
        assert isinstance(adapter.group_chats, dict)
    
    def test_adapter_requires_autogen(self):
        """Test adapter raises error if AutoGen not available."""
        # This test verifies the import check, but won't run if AutoGen is missing
        # due to the module-level skip
        manager = LLMManager()
        
        with patch('llm_interface.src.framework.autogen.adapter.AUTOGEN_AVAILABLE', False):
            with pytest.raises(FrameworkConfigurationError, match="AutoGen is not installed"):
                # Need to reload the module to trigger the check
                from importlib import reload
                import llm_interface.src.framework.autogen.adapter as autogen_module
                reload(autogen_module)
                autogen_module.AutoGenAdapter(manager)
    
    def test_create_agent_basic(self):
        """Test creating a basic agent."""
        manager = LLMManager()
        
        # Add OpenAI provider
        config = LLMConfig(
            provider=ProviderType.OPENAI,
            model_name="gpt-4",
            api_key="test-key",
            temperature=0.7
        )
        manager.add_provider("openai", config)
        
        adapter = AutoGenAdapter(manager)
        
        # Mock ConversableAgent
        with patch('llm_interface.src.framework.autogen.adapter.ConversableAgent') as mock_agent:
            mock_instance = MagicMock()
            mock_agent.return_value = mock_instance
            
            agent = adapter.create_agent(
                name="test_agent",
                provider_name="openai",
                system_message="Test system message"
            )
            
            # Verify agent was created
            assert mock_agent.called
            assert "test_agent" in adapter.agents
            
            # Verify LLM config was passed correctly
            call_kwargs = mock_agent.call_args[1]
            assert call_kwargs["name"] == "test_agent"
            assert call_kwargs["system_message"] == "Test system message"
            assert "llm_config" in call_kwargs
            
            llm_config = call_kwargs["llm_config"]
            assert llm_config["model"] == "gpt-4"
            assert llm_config["temperature"] == 0.7
            assert llm_config["api_key"] == "test-key"
    
    def test_create_agent_provider_not_found(self):
        """Test error when provider not found."""
        manager = LLMManager()
        adapter = AutoGenAdapter(manager)
        
        with pytest.raises(FrameworkConfigurationError, match="Provider 'nonexistent' not found"):
            adapter.create_agent(
                name="test",
                provider_name="nonexistent"
            )
    
    def test_create_agent_with_azure(self):
        """Test creating agent with Azure provider."""
        manager = LLMManager()
        
        azure_config = LLMConfig(
            provider=ProviderType.AZURE,
            model_name="gpt-4",
            api_key="azure-key",
            azure_endpoint="https://test.openai.azure.com",
            azure_deployment="gpt-4-deployment",
            api_version="2024-02-15-preview",
            temperature=0.5,
            max_tokens=2000
        )
        manager.add_provider("azure", azure_config)
        
        adapter = AutoGenAdapter(manager)
        
        with patch('llm_interface.src.framework.autogen.adapter.ConversableAgent') as mock_agent:
            adapter.create_agent(
                name="azure_agent",
                provider_name="azure"
            )
            
            call_kwargs = mock_agent.call_args[1]
            llm_config = call_kwargs["llm_config"]
            
            assert llm_config["api_type"] == "azure"
            assert llm_config["api_base"] == "https://test.openai.azure.com"
            assert llm_config["api_version"] == "2024-02-15-preview"
            assert llm_config["max_tokens"] == 2000
    
    def test_create_agent_with_anthropic(self):
        """Test creating agent with Anthropic provider."""
        manager = LLMManager()
        
        anthropic_config = LLMConfig(
            provider=ProviderType.ANTHROPIC,
            model_name="claude-3-sonnet-20240229",
            api_key="anthropic-key",
            temperature=0.7
        )
        manager.add_provider("claude", anthropic_config)
        
        adapter = AutoGenAdapter(manager)
        
        with patch('llm_interface.src.framework.autogen.adapter.ConversableAgent') as mock_agent:
            adapter.create_agent(
                name="claude_agent",
                provider_name="claude"
            )
            
            call_kwargs = mock_agent.call_args[1]
            llm_config = call_kwargs["llm_config"]
            
            assert llm_config["model"] == "claude-3-sonnet-20240229"
            assert llm_config["api_key"] == "anthropic-key"
    
    def test_create_user_proxy(self):
        """Test creating a user proxy agent."""
        manager = LLMManager()
        adapter = AutoGenAdapter(manager)
        
        with patch('llm_interface.src.framework.autogen.adapter.ConversableAgent') as mock_agent:
            mock_instance = MagicMock()
            mock_agent.return_value = mock_instance
            
            user_proxy = adapter.create_user_proxy(
                name="user",
                human_input_mode="NEVER"
            )
            
            assert mock_agent.called
            call_kwargs = mock_agent.call_args[1]
            
            # User proxy should have no LLM
            assert call_kwargs["llm_config"] is False
            assert call_kwargs["name"] == "user"
            assert "user" in adapter.agents
    
    def test_create_group_chat(self):
        """Test creating a group chat."""
        manager = LLMManager()
        adapter = AutoGenAdapter(manager)
        
        # Create mock agents
        mock_agent1 = MagicMock()
        mock_agent2 = MagicMock()
        mock_agent3 = MagicMock()
        
        with patch('llm_interface.src.framework.autogen.adapter.GroupChat') as mock_gc:
            mock_gc_instance = MagicMock()
            mock_gc.return_value = mock_gc_instance
            
            group_chat = adapter.create_group_chat(
                name="test_group",
                agents=[mock_agent1, mock_agent2, mock_agent3],
                max_round=15,
                speaker_selection_method="round_robin"
            )
            
            assert mock_gc.called
            assert "test_group" in adapter.group_chats
            
            call_kwargs = mock_gc.call_args[1]
            assert len(call_kwargs["agents"]) == 3
            assert call_kwargs["max_round"] == 15
            assert call_kwargs["speaker_selection_method"] == "round_robin"
    
    def test_create_group_chat_manager(self):
        """Test creating a group chat manager."""
        manager = LLMManager()
        
        config = LLMConfig(
            provider=ProviderType.OPENAI,
            model_name="gpt-4",
            api_key="test-key",
            temperature=0.5
        )
        manager.add_provider("openai", config)
        
        adapter = AutoGenAdapter(manager)
        
        # Create mock group chat
        mock_gc = MagicMock()
        
        with patch('llm_interface.src.framework.autogen.adapter.GroupChatManager') as mock_gcm:
            gc_manager = adapter.create_group_chat_manager(
                group_chat=mock_gc,
                provider_name="openai",
                system_message="Manage the chat"
            )
            
            assert mock_gcm.called
            call_kwargs = mock_gcm.call_args[1]
            
            assert call_kwargs["groupchat"] is mock_gc
            assert call_kwargs["system_message"] == "Manage the chat"
            assert "llm_config" in call_kwargs
    
    def test_initiate_chat(self):
        """Test initiating a chat between agents."""
        manager = LLMManager()
        adapter = AutoGenAdapter(manager)
        
        # Create mock agents
        mock_sender = MagicMock()
        mock_recipient = MagicMock()
        mock_sender.initiate_chat.return_value = {"status": "success"}
        
        result = adapter.initiate_chat(
            sender=mock_sender,
            recipient=mock_recipient,
            message="Hello"
        )
        
        assert mock_sender.initiate_chat.called
        assert result["status"] == "success"
    
    def test_create_model_raises_not_implemented(self):
        """Test that create_model raises NotImplementedError."""
        manager = LLMManager()
        adapter = AutoGenAdapter(manager)
        
        with pytest.raises(NotImplementedError, match="use create_agent"):
            adapter.create_model("openai")
    
    def test_validate_config(self):
        """Test configuration validation."""
        manager = LLMManager()
        adapter = AutoGenAdapter(manager)
        
        # Valid config
        valid_config = {
            "agents": [
                {"name": "agent1", "provider_name": "openai"}
            ]
        }
        assert adapter.validate_config(valid_config) is True
        
        # Invalid config (not a dict)
        with pytest.raises(FrameworkConfigurationError, match="must be a dictionary"):
            adapter.validate_config("not a dict")
        
        # Invalid agents config
        invalid_config = {"agents": "not a list"}
        with pytest.raises(FrameworkConfigurationError, match="must be list or dict"):
            adapter.validate_config(invalid_config)
    
    def test_get_agent(self):
        """Test retrieving an agent by name."""
        manager = LLMManager()
        adapter = AutoGenAdapter(manager)
        
        mock_agent = MagicMock()
        adapter.agents["test"] = mock_agent
        
        assert adapter.get_agent("test") is mock_agent
        
        with pytest.raises(KeyError, match="Agent 'nonexistent' not found"):
            adapter.get_agent("nonexistent")
    
    def test_list_agents(self):
        """Test listing all agents."""
        manager = LLMManager()
        adapter = AutoGenAdapter(manager)
        
        adapter.agents["agent1"] = MagicMock()
        adapter.agents["agent2"] = MagicMock()
        
        agents = adapter.list_agents()
        assert len(agents) == 2
        assert "agent1" in agents
        assert "agent2" in agents
    
    def test_get_group_chat(self):
        """Test retrieving a group chat by name."""
        manager = LLMManager()
        adapter = AutoGenAdapter(manager)
        
        mock_gc = MagicMock()
        adapter.group_chats["test_gc"] = mock_gc
        
        assert adapter.get_group_chat("test_gc") is mock_gc
        
        with pytest.raises(KeyError, match="Group chat 'nonexistent' not found"):
            adapter.get_group_chat("nonexistent")
    
    def test_list_group_chats(self):
        """Test listing all group chats."""
        manager = LLMManager()
        adapter = AutoGenAdapter(manager)
        
        adapter.group_chats["gc1"] = MagicMock()
        adapter.group_chats["gc2"] = MagicMock()
        
        gcs = adapter.list_group_chats()
        assert len(gcs) == 2
        assert "gc1" in gcs
        assert "gc2" in gcs
    
    def test_list_providers(self):
        """Test listing providers from manager."""
        manager = LLMManager()
        
        config1 = LLMConfig(
            provider=ProviderType.OPENAI,
            model_name="gpt-4",
            api_key="key1"
        )
        config2 = LLMConfig(
            provider=ProviderType.ANTHROPIC,
            model_name="claude-3-sonnet",
            api_key="key2"
        )
        
        manager.add_provider("openai", config1)
        manager.add_provider("claude", config2)
        
        adapter = AutoGenAdapter(manager)
        providers = adapter.list_providers()
        
        assert len(providers) == 2
        assert "openai" in providers
        assert "claude" in providers
    
    def test_get_provider_info(self):
        """Test getting provider information."""
        manager = LLMManager()
        
        config = LLMConfig(
            provider=ProviderType.OPENAI,
            model_name="gpt-4",
            api_key="key"
        )
        manager.add_provider("openai", config)
        
        adapter = AutoGenAdapter(manager)
        info = adapter.get_provider_info()
        
        assert "openai" in info
        assert info["openai"]["type"] == "openai"
        assert info["openai"]["model"] == "gpt-4"
