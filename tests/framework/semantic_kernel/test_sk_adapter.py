"""
Tests for Semantic Kernel adapter.

Tests the Semantic Kernel framework integration with llm_interface providers.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from llm_interface.src.manager import LLMManager
from llm_interface.src.core.config import LLMConfig, ProviderType
from llm_interface.src.framework.exceptions import FrameworkConfigurationError

# Skip all tests if Semantic Kernel is not available
try:
    from llm_interface.src.framework.semantic_kernel import SemanticKernelAdapter
    SK_AVAILABLE = True
except ImportError:
    SK_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not SK_AVAILABLE,
    reason="Semantic Kernel not installed"
)


class TestSemanticKernelAdapter:
    """Test suite for SemanticKernelAdapter."""
    
    def test_adapter_initialization(self):
        """Test adapter initializes correctly."""
        manager = LLMManager()
        adapter = SemanticKernelAdapter(manager)
        
        assert adapter.framework_name == "semantic_kernel"
        assert adapter.framework_version == "1.0.0"
        assert adapter.llm_manager is manager
        assert isinstance(adapter.kernels, dict)
        assert isinstance(adapter.plugins, dict)
        assert isinstance(adapter.functions, dict)
    
    def test_create_kernel_with_openai(self):
        """Test creating kernel with OpenAI provider."""
        manager = LLMManager()
        
        config = LLMConfig(
            provider=ProviderType.OPENAI,
            model_name="gpt-4",
            api_key="test-key"
        )
        manager.add_provider("openai", config)
        
        adapter = SemanticKernelAdapter(manager)
        
        with patch('llm_interface.src.framework.semantic_kernel.adapter.Kernel') as mock_kernel_class:
            with patch('llm_interface.src.framework.semantic_kernel.adapter.OpenAIChatCompletion') as mock_service:
                mock_kernel = MagicMock()
                mock_kernel_class.return_value = mock_kernel
                
                kernel = adapter.create_kernel("openai")
                
                assert mock_kernel_class.called
                assert "openai" in adapter.kernels
                assert mock_service.called
                assert mock_kernel.add_service.called
    
    def test_create_kernel_with_azure(self):
        """Test creating kernel with Azure provider."""
        manager = LLMManager()
        
        config = LLMConfig(
            provider=ProviderType.AZURE,
            model_name="gpt-4",
            api_key="azure-key",
            azure_endpoint="https://test.openai.azure.com",
            azure_deployment="gpt-4",
            api_version="2024-02-15-preview"
        )
        manager.add_provider("azure", config)
        
        adapter = SemanticKernelAdapter(manager)
        
        with patch('llm_interface.src.framework.semantic_kernel.adapter.Kernel') as mock_kernel_class:
            with patch('llm_interface.src.framework.semantic_kernel.adapter.AzureChatCompletion') as mock_service:
                mock_kernel = MagicMock()
                mock_kernel_class.return_value = mock_kernel
                
                kernel = adapter.create_kernel("azure")
                
                assert mock_kernel_class.called
                assert "azure" in adapter.kernels
                assert mock_service.called
                
                # Verify Azure-specific config
                call_kwargs = mock_service.call_args[1]
                assert call_kwargs["deployment_name"] == "gpt-4"
                assert call_kwargs["endpoint"] == "https://test.openai.azure.com"
    
    def test_create_kernel_provider_not_found(self):
        """Test error when provider not found."""
        manager = LLMManager()
        adapter = SemanticKernelAdapter(manager)
        
        with pytest.raises(FrameworkConfigurationError, match="Provider 'nonexistent' not found"):
            adapter.create_kernel("nonexistent")
    
    def test_create_kernel_unsupported_provider(self):
        """Test error with unsupported provider type."""
        manager = LLMManager()
        
        config = LLMConfig(
            provider=ProviderType.ANTHROPIC,  # Not supported by SK
            model_name="claude-3-sonnet",
            api_key="test-key"
        )
        manager.add_provider("anthropic", config)
        
        adapter = SemanticKernelAdapter(manager)
        
        with pytest.raises(FrameworkConfigurationError, match="not supported by Semantic Kernel"):
            adapter.create_kernel("anthropic")
    
    def test_create_semantic_function(self):
        """Test creating a semantic function."""
        manager = LLMManager()
        
        config = LLMConfig(
            provider=ProviderType.OPENAI,
            model_name="gpt-4",
            api_key="test-key"
        )
        manager.add_provider("openai", config)
        
        adapter = SemanticKernelAdapter(manager)
        
        with patch('llm_interface.src.framework.semantic_kernel.adapter.Kernel') as mock_kernel_class:
            mock_kernel = MagicMock()
            mock_kernel_class.return_value = mock_kernel
            mock_function = MagicMock()
            mock_kernel.add_function.return_value = mock_function
            
            kernel = adapter.create_kernel("openai")
            
            function = adapter.create_semantic_function(
                kernel=kernel,
                prompt="Translate to {{$language}}: {{$text}}",
                function_name="translate",
                plugin_name="text",
                description="Translates text",
                max_tokens=200,
                temperature=0.5
            )
            
            assert mock_kernel.add_function.called
            assert "text.translate" in adapter.functions or "translate" in adapter.functions
    
    def test_create_native_function(self):
        """Test creating a native function."""
        manager = LLMManager()
        
        config = LLMConfig(
            provider=ProviderType.OPENAI,
            model_name="gpt-4",
            api_key="test-key"
        )
        manager.add_provider("openai", config)
        
        adapter = SemanticKernelAdapter(manager)
        
        with patch('llm_interface.src.framework.semantic_kernel.adapter.Kernel') as mock_kernel_class:
            mock_kernel = MagicMock()
            mock_kernel_class.return_value = mock_kernel
            mock_function = MagicMock()
            mock_kernel.add_function.return_value = mock_function
            
            kernel = adapter.create_kernel("openai")
            
            def test_func(text: str) -> str:
                return text.upper()
            
            function = adapter.create_native_function(
                kernel=kernel,
                function=test_func,
                function_name="to_upper",
                plugin_name="text",
                description="Converts to uppercase"
            )
            
            assert mock_kernel.add_function.called
            assert "text.to_upper" in adapter.functions or "to_upper" in adapter.functions
    
    def test_create_plugin(self):
        """Test creating a plugin."""
        manager = LLMManager()
        
        config = LLMConfig(
            provider=ProviderType.OPENAI,
            model_name="gpt-4",
            api_key="test-key"
        )
        manager.add_provider("openai", config)
        
        adapter = SemanticKernelAdapter(manager)
        
        with patch('llm_interface.src.framework.semantic_kernel.adapter.Kernel') as mock_kernel_class:
            mock_kernel = MagicMock()
            mock_kernel_class.return_value = mock_kernel
            mock_plugin = MagicMock()
            mock_kernel.plugins.get.return_value = mock_plugin
            
            kernel = adapter.create_kernel("openai")
            
            def func1(x: int) -> int:
                return x * 2
            
            def func2(x: int) -> int:
                return x + 1
            
            plugin = adapter.create_plugin(
                kernel=kernel,
                plugin_name="math",
                functions={"double": func1, "increment": func2}
            )
            
            assert mock_kernel.add_function.call_count == 2
            assert "math" in adapter.plugins or mock_kernel.plugins.get.called
    
    @pytest.mark.asyncio
    async def test_invoke_function_async(self):
        """Test async function invocation."""
        manager = LLMManager()
        
        config = LLMConfig(
            provider=ProviderType.OPENAI,
            model_name="gpt-4",
            api_key="test-key"
        )
        manager.add_provider("openai", config)
        
        adapter = SemanticKernelAdapter(manager)
        
        with patch('llm_interface.src.framework.semantic_kernel.adapter.Kernel') as mock_kernel_class:
            mock_kernel = MagicMock()
            mock_kernel_class.return_value = mock_kernel
            mock_function = MagicMock()
            mock_kernel.invoke = AsyncMock(return_value="result")
            
            kernel = adapter.create_kernel("openai")
            adapter.functions["test_func"] = mock_function
            
            result = await adapter.invoke_function(
                kernel=kernel,
                function="test_func",
                arguments={"arg": "value"}
            )
            
            assert mock_kernel.invoke.called
    
    def test_invoke_function_sync(self):
        """Test sync function invocation wrapper."""
        manager = LLMManager()
        
        config = LLMConfig(
            provider=ProviderType.OPENAI,
            model_name="gpt-4",
            api_key="test-key"
        )
        manager.add_provider("openai", config)
        
        adapter = SemanticKernelAdapter(manager)
        
        with patch('llm_interface.src.framework.semantic_kernel.adapter.Kernel') as mock_kernel_class:
            mock_kernel = MagicMock()
            mock_kernel_class.return_value = mock_kernel
            mock_function = MagicMock()
            
            # Mock the async invoke to return a simple value
            async def mock_invoke(*args, **kwargs):
                return "sync_result"
            
            with patch.object(adapter, 'invoke_function', side_effect=mock_invoke):
                kernel = adapter.create_kernel("openai")
                adapter.functions["test_func"] = mock_function
                
                result = adapter.invoke_function_sync(
                    kernel=kernel,
                    function="test_func",
                    arguments={"arg": "value"}
                )
                
                assert result == "sync_result"
    
    def test_create_model(self):
        """Test create_model (alias for create_kernel)."""
        manager = LLMManager()
        
        config = LLMConfig(
            provider=ProviderType.OPENAI,
            model_name="gpt-4",
            api_key="test-key"
        )
        manager.add_provider("openai", config)
        
        adapter = SemanticKernelAdapter(manager)
        
        with patch('llm_interface.src.framework.semantic_kernel.adapter.Kernel') as mock_kernel_class:
            mock_kernel = MagicMock()
            mock_kernel_class.return_value = mock_kernel
            
            kernel = adapter.create_model("openai")
            
            assert mock_kernel_class.called
            assert "openai" in adapter.kernels
    
    def test_validate_config(self):
        """Test configuration validation."""
        manager = LLMManager()
        adapter = SemanticKernelAdapter(manager)
        
        # Valid config
        valid_config = {
            "kernels": [
                {"provider_name": "openai"}
            ]
        }
        assert adapter.validate_config(valid_config) is True
        
        # Invalid config (not a dict)
        with pytest.raises(FrameworkConfigurationError, match="must be a dictionary"):
            adapter.validate_config("not a dict")
        
        # Invalid kernels config
        invalid_config = {"kernels": "not a list"}
        with pytest.raises(FrameworkConfigurationError, match="must be list or dict"):
            adapter.validate_config(invalid_config)
    
    def test_get_kernel(self):
        """Test retrieving a kernel by ID."""
        manager = LLMManager()
        adapter = SemanticKernelAdapter(manager)
        
        mock_kernel = MagicMock()
        adapter.kernels["test"] = mock_kernel
        
        assert adapter.get_kernel("test") is mock_kernel
        
        with pytest.raises(KeyError, match="Kernel 'nonexistent' not found"):
            adapter.get_kernel("nonexistent")
    
    def test_list_kernels(self):
        """Test listing all kernels."""
        manager = LLMManager()
        adapter = SemanticKernelAdapter(manager)
        
        adapter.kernels["kernel1"] = MagicMock()
        adapter.kernels["kernel2"] = MagicMock()
        
        kernels = adapter.list_kernels()
        assert len(kernels) == 2
        assert "kernel1" in kernels
        assert "kernel2" in kernels
    
    def test_get_plugin(self):
        """Test retrieving a plugin by name."""
        manager = LLMManager()
        adapter = SemanticKernelAdapter(manager)
        
        mock_plugin = MagicMock()
        adapter.plugins["test"] = mock_plugin
        
        assert adapter.get_plugin("test") is mock_plugin
        
        with pytest.raises(KeyError, match="Plugin 'nonexistent' not found"):
            adapter.get_plugin("nonexistent")
    
    def test_list_plugins(self):
        """Test listing all plugins."""
        manager = LLMManager()
        adapter = SemanticKernelAdapter(manager)
        
        adapter.plugins["plugin1"] = MagicMock()
        adapter.plugins["plugin2"] = MagicMock()
        
        plugins = adapter.list_plugins()
        assert len(plugins) == 2
        assert "plugin1" in plugins
        assert "plugin2" in plugins
    
    def test_get_function(self):
        """Test retrieving a function by name."""
        manager = LLMManager()
        adapter = SemanticKernelAdapter(manager)
        
        mock_function = MagicMock()
        adapter.functions["test.func"] = mock_function
        
        assert adapter.get_function("test.func") is mock_function
        
        with pytest.raises(KeyError, match="Function 'nonexistent' not found"):
            adapter.get_function("nonexistent")
    
    def test_list_functions(self):
        """Test listing all functions."""
        manager = LLMManager()
        adapter = SemanticKernelAdapter(manager)
        
        adapter.functions["func1"] = MagicMock()
        adapter.functions["plugin.func2"] = MagicMock()
        
        functions = adapter.list_functions()
        assert len(functions) == 2
        assert "func1" in functions
        assert "plugin.func2" in functions
    
    def test_list_providers(self):
        """Test listing providers from manager."""
        manager = LLMManager()
        
        config1 = LLMConfig(
            provider=ProviderType.OPENAI,
            model_name="gpt-4",
            api_key="key1"
        )
        config2 = LLMConfig(
            provider=ProviderType.AZURE,
            model_name="gpt-4",
            api_key="key2",
            azure_endpoint="https://test.openai.azure.com",
            azure_deployment="gpt-4"
        )
        
        manager.add_provider("openai", config1)
        manager.add_provider("azure", config2)
        
        adapter = SemanticKernelAdapter(manager)
        providers = adapter.list_providers()
        
        assert len(providers) == 2
        assert "openai" in providers
        assert "azure" in providers
    
    def test_get_provider_info(self):
        """Test getting provider information."""
        manager = LLMManager()
        
        config = LLMConfig(
            provider=ProviderType.OPENAI,
            model_name="gpt-4",
            api_key="key"
        )
        manager.add_provider("openai", config)
        
        adapter = SemanticKernelAdapter(manager)
        info = adapter.get_provider_info()
        
        assert "openai" in info
        assert info["openai"]["type"] == "openai"
        assert info["openai"]["model"] == "gpt-4"
