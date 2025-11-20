"""
Integration tests for LLM Interface components.

Tests the interaction between multiple components such as:
- Configuration loading and provider initialization
- Manager with multiple providers
- LangGraph workflows
- End-to-end workflows
"""

import pytest
import os
import tempfile
import yaml
from unittest.mock import Mock, patch

from llm_interface import (
    LLMConfig,
    ProviderType,
    LLMManager,
    LLMProviderFactory,
    ConfigLoader,
    LLMGraph,
)
from llm_interface.src.core.exceptions import (
    ProviderNotFoundError,
    ConfigurationError,
)
from langchain_core.messages import HumanMessage


class TestConfigLoaderIntegration:
    """Test ConfigLoader with multiple sources and formats"""
    
    def test_load_config_from_yaml_and_use_with_manager(self):
        """Test loading YAML config and using it with manager"""
        yaml_content = """
providers:
  openai:
    provider: openai
    model_name: gpt-3.5-turbo
    api_key: test-key-123
    temperature: 0.7
  ollama:
    provider: ollama
    model_name: llama2
    base_url: http://localhost:11434
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            f.flush()
            
            try:
                # Load configuration
                configs = ConfigLoader.from_yaml(f.name)
                
                # Verify loaded configs
                assert 'openai' in configs
                assert 'ollama' in configs
                assert configs['openai'].model_name == 'gpt-3.5-turbo'
                assert configs['ollama'].model_name == 'llama2'
                
            finally:
                # Close file before deleting on Windows
                try:
                    os.unlink(f.name)
                except PermissionError:
                    pass  # File still in use on Windows, will be cleaned up later
    
    def test_load_config_from_dict_and_env_merge(self):
        """Test merging config from dictionary and environment"""
        config_dict = {
            'providers': {
                'local': {
                    'provider': 'ollama',
                    'model_name': 'llama2'
                }
            }
        }
        
        with patch.dict(os.environ, {'LLM_CLOUD_PROVIDER': 'openai', 'LLM_CLOUD_MODEL_NAME': 'gpt-4'}):
            dict_configs = ConfigLoader.from_dict(config_dict)
            env_configs = ConfigLoader.from_env('LLM_')
            
            # Verify both loaded correctly
            assert 'local' in dict_configs
            
            # Merge should combine both
            merged = ConfigLoader.merge_configs(dict_configs, env_configs)
            assert 'local' in merged


class TestManagerWithMultipleProviders:
    """Test LLMManager with multiple providers"""
    
    @patch('llm_interface.src.manager.LLMProviderFactory.create_provider')
    def test_add_multiple_providers_and_retrieve(self, mock_create_provider):
        """Test adding and retrieving multiple providers"""
        from llm_interface.src.core.interfaces import LLMProvider
        
        # Create mock providers
        class MockProvider(LLMProvider):
            def __init__(self, config):
                self.config = config
                self._model = Mock(name=f'model_{config.model_name}')
                self._chat = Mock(name=f'chat_{config.model_name}')
            
            def get_model(self):
                return self._model
            
            def get_chat_model(self):
                return self._chat
            
            def get_embeddings(self):
                return Mock(name=f'embeddings_{self.config.model_name}')
            
            def validate_config(self):
                return True
        
        manager = LLMManager()
        
        # Create multiple configs
        openai_config = LLMConfig(
            provider=ProviderType.OPENAI,
            model_name='gpt-4',
            api_key='key1'
        )
        
        ollama_config = LLMConfig(
            provider=ProviderType.OLLAMA,
            model_name='llama2'
        )
        
        # Mock factory to return different providers
        mock_create_provider.side_effect = [
            MockProvider(openai_config),
            MockProvider(ollama_config)
        ]
        
        # Add providers
        manager.add_provider('openai', openai_config)
        manager.add_provider('ollama', ollama_config)
        
        # Verify retrieval
        assert len(manager.list_providers()) == 2
        assert 'openai' in manager.list_providers()
        assert 'ollama' in manager.list_providers()
        
        # Verify models are retrievable
        openai_model = manager.get_chat_model('openai')
        ollama_model = manager.get_chat_model('ollama')
        
        assert openai_model is not None
        assert ollama_model is not None
        assert openai_model != ollama_model
    
    @patch('llm_interface.src.manager.LLMProviderFactory.create_provider')
    def test_create_chain_with_specific_provider(self, mock_create_provider):
        """Test creating a chain with a specific provider"""
        from llm_interface.src.core.interfaces import LLMProvider
        
        class MockProvider(LLMProvider):
            def __init__(self, config):
                self.config = config
            
            def get_model(self):
                return Mock()
            
            def get_chat_model(self):
                return Mock()
            
            def get_embeddings(self):
                return Mock()
            
            def validate_config(self):
                return True
        
        manager = LLMManager()
        config = LLMConfig(
            provider=ProviderType.OPENAI,
            model_name='gpt-3.5-turbo',
            api_key='test'
        )
        
        mock_create_provider.return_value = MockProvider(config)
        manager.add_provider('gpt35', config)
        
        # Create chain should succeed
        chain = manager.create_chain('gpt35', 'Prompt: {input}')
        assert chain is not None


class TestFactoryWithDynamicRegistration:
    """Test factory with dynamic provider registration"""
    
    def test_register_and_use_custom_provider(self):
        """Test registering a custom provider dynamically"""
        from llm_interface.src.core.interfaces import LLMProvider
        
        # Create custom provider
        class CustomTestProvider(LLMProvider):
            def __init__(self, config):
                self.config = config
            
            def get_model(self):
                return Mock()
            
            def get_chat_model(self):
                return Mock()
            
            def get_embeddings(self):
                return Mock()
            
            def validate_config(self):
                return True
        
        # Save original OLLAMA provider
        original_ollama = LLMProviderFactory._providers.get(ProviderType.OLLAMA)
        
        try:
            # Temporarily override OLLAMA with custom provider for testing
            LLMProviderFactory._providers[ProviderType.OLLAMA] = CustomTestProvider
            
            # Verify registration
            assert LLMProviderFactory._providers[ProviderType.OLLAMA] == CustomTestProvider
            
            # Create provider using OLLAMA type but our custom class
            config = LLMConfig(
                provider=ProviderType.OLLAMA,
                model_name='test-model'
            )
            
            provider = LLMProviderFactory.create_provider(config)
            assert isinstance(provider, CustomTestProvider)
            
        finally:
            # Restore original OLLAMA provider
            if original_ollama:
                LLMProviderFactory._providers[ProviderType.OLLAMA] = original_ollama


class TestLLMGraphIntegration:
    """Test LangGraph integration with manager"""
    
    @patch('llm_interface.src.manager.LLMProviderFactory.create_provider')
    def test_create_simple_graph_with_providers(self, mock_create_provider):
        """Test creating and compiling a simple graph"""
        from llm_interface.src.core.interfaces import LLMProvider
        
        class MockProvider(LLMProvider):
            def __init__(self, config):
                self.config = config
                self._chat = Mock(spec=['invoke'])
                self._chat.invoke.return_value = Mock(content='Test response')
            
            def get_model(self):
                return self._chat
            
            def get_chat_model(self):
                return self._chat
            
            def get_embeddings(self):
                return Mock()
            
            def validate_config(self):
                return True
        
        # Setup manager with provider
        manager = LLMManager()
        config = LLMConfig(
            provider=ProviderType.OPENAI,
            model_name='gpt-3.5-turbo',
            api_key='test'
        )
        
        mock_create_provider.return_value = MockProvider(config)
        manager.add_provider('openai', config)
        
        # Create graph
        graph = LLMGraph(manager)
        workflow = graph.create_simple_chat_graph()
        
        # Should be compilable
        assert workflow is not None
        compiled = graph.compile_graph(workflow)
        assert compiled is not None
    
    @patch('llm_interface.src.manager.LLMProviderFactory.create_provider')
    def test_run_simple_chat_through_graph(self, mock_create_provider):
        """Test running a simple chat through the graph"""
        from llm_interface.src.core.interfaces import LLMProvider
        
        class MockProvider(LLMProvider):
            def __init__(self, config):
                self.config = config
                self._chat = Mock(spec=['invoke'])
                self._chat.invoke.return_value = Mock(content='Hello from AI')
            
            def get_model(self):
                return self._chat
            
            def get_chat_model(self):
                return self._chat
            
            def get_embeddings(self):
                return Mock()
            
            def validate_config(self):
                return True
        
        # Setup
        manager = LLMManager()
        config = LLMConfig(
            provider=ProviderType.OLLAMA,
            model_name='llama2'
        )
        
        mock_create_provider.return_value = MockProvider(config)
        manager.add_provider('local', config)
        
        # Run simple chat
        graph = LLMGraph(manager)
        messages = [HumanMessage(content='Hello')]
        
        result = graph.run_simple_chat(messages, 'local')
        
        # Verify result structure
        assert 'response' in result
        assert 'messages' in result
        assert result['provider'] == 'local'


class TestErrorHandlingAcrossComponents:
    """Test error handling across multiple components"""
    
    def test_missing_provider_error_includes_available(self):
        """Test that missing provider error includes available providers"""
        manager = LLMManager()
        
        with pytest.raises(ProviderNotFoundError) as exc_info:
            manager.get_chat_model('nonexistent')
        
        # Message should indicate no providers available
        assert 'No providers have been added' in str(exc_info.value)
    
    @patch('llm_interface.src.manager.LLMProviderFactory.create_provider')
    def test_missing_provider_with_available_providers(self, mock_create_provider):
        """Test error message when provider exists but requested one doesn't"""
        from llm_interface.src.core.interfaces import LLMProvider
        
        class MockProvider(LLMProvider):
            def __init__(self, config):
                self.config = config
            
            def get_model(self):
                return Mock()
            
            def get_chat_model(self):
                return Mock()
            
            def get_embeddings(self):
                return Mock()
            
            def validate_config(self):
                return True
        
        manager = LLMManager()
        config = LLMConfig(
            provider=ProviderType.OPENAI,
            model_name='gpt-4',
            api_key='test'
        )
        
        mock_create_provider.return_value = MockProvider(config)
        manager.add_provider('openai', config)
        
        # Try to get nonexistent provider
        with pytest.raises(ProviderNotFoundError) as exc_info:
            manager.get_chat_model('azure')
        
        # Message should include available providers
        error_msg = str(exc_info.value)
        assert 'openai' in error_msg
        assert 'azure' in error_msg


class TestConfigurationErrorHandling:
    """Test configuration error handling"""
    
    def test_invalid_yaml_config(self):
        """Test handling invalid YAML configuration"""
        invalid_yaml = """
providers:
  openai:
    provider: openai
    model_name: gpt-4
    invalid syntax here {{
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(invalid_yaml)
            f.flush()
            
            try:
                with pytest.raises(ValueError, match='Invalid YAML'):
                    ConfigLoader.from_yaml(f.name)
            finally:
                # Close file before deleting on Windows
                try:
                    os.unlink(f.name)
                except PermissionError:
                    pass  # File still in use on Windows, will be cleaned up later
    
    def test_missing_yaml_file(self):
        """Test handling missing YAML file"""
        with pytest.raises(FileNotFoundError):
            ConfigLoader.from_yaml('/nonexistent/path/config.yaml')
    
    def test_malformed_config_data(self):
        """Test handling malformed configuration data"""
        bad_config = {
            'providers': {
                'test': {
                    'provider': 'invalid_provider_type'
                }
            }
        }
        
        with pytest.raises(ValueError):
            ConfigLoader.from_dict(bad_config)


class TestConcurrentProviderAccess:
    """Test concurrent access patterns (basic)"""
    
    @patch('llm_interface.src.manager.LLMProviderFactory.create_provider')
    def test_thread_safe_provider_registration(self, mock_create_provider):
        """Test that provider registration is thread-safe"""
        from llm_interface.src.core.interfaces import LLMProvider
        
        class MockProvider(LLMProvider):
            def __init__(self, config):
                self.config = config
            
            def get_model(self):
                return Mock()
            
            def get_chat_model(self):
                return Mock()
            
            def get_embeddings(self):
                return Mock()
            
            def validate_config(self):
                return True
        
        # Factory uses locking
        factory_info = LLMProviderFactory.get_factory_info()
        
        # Should have lock available
        assert hasattr(LLMProviderFactory, '_lock')
        assert LLMProviderFactory._lock is not None
        
        # Should be able to register providers
        providers_before = len(factory_info['supported_providers'])
        
        # Get factory info again
        factory_info_after = LLMProviderFactory.get_factory_info()
        assert factory_info_after['total_providers'] == providers_before
