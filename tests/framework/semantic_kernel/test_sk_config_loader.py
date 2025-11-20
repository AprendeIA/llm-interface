"""
Tests for Semantic Kernel config loader.

Tests the YAML configuration loading and validation for Semantic Kernel.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from llm_interface.src.framework.exceptions import FrameworkConfigurationError

# Skip all tests if Semantic Kernel is not available
try:
    from llm_interface.src.framework.semantic_kernel.config_loader import SemanticKernelConfigLoader
    SK_AVAILABLE = True
except ImportError:
    SK_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not SK_AVAILABLE,
    reason="Semantic Kernel not installed"
)


class TestSemanticKernelConfigLoader:
    """Test suite for SemanticKernelConfigLoader."""
    
    def test_validate_kernel_config_valid(self):
        """Test validation of valid kernel config."""
        loader = SemanticKernelConfigLoader()
        
        config = {
            "provider_name": "openai"
        }
        
        # Should not raise
        loader.validate_kernel_config(config)
    
    def test_validate_kernel_config_invalid_type(self):
        """Test validation rejects non-dict config."""
        loader = SemanticKernelConfigLoader()
        
        with pytest.raises(FrameworkConfigurationError, match="must be a dictionary"):
            loader.validate_kernel_config("not a dict")
    
    def test_validate_kernel_config_missing_provider(self):
        """Test validation rejects config without provider_name."""
        loader = SemanticKernelConfigLoader()
        
        config = {}
        
        with pytest.raises(FrameworkConfigurationError, match="must specify 'provider_name'"):
            loader.validate_kernel_config(config)
    
    def test_validate_function_config_semantic_valid(self):
        """Test validation of valid semantic function config."""
        loader = SemanticKernelConfigLoader()
        
        config = {
            "name": "translate",
            "type": "semantic",
            "prompt": "Translate: {{$text}}"
        }
        
        loader.validate_function_config(config)
    
    def test_validate_function_config_native_valid(self):
        """Test validation of valid native function config."""
        loader = SemanticKernelConfigLoader()
        
        config = {
            "name": "to_upper",
            "type": "native",
            "callable": "str.upper"
        }
        
        loader.validate_function_config(config)
    
    def test_validate_function_config_invalid_type(self):
        """Test validation rejects non-dict config."""
        loader = SemanticKernelConfigLoader()
        
        with pytest.raises(FrameworkConfigurationError, match="must be a dictionary"):
            loader.validate_function_config("not a dict")
    
    def test_validate_function_config_missing_name(self):
        """Test validation rejects config without name."""
        loader = SemanticKernelConfigLoader()
        
        config = {
            "type": "semantic",
            "prompt": "Test"
        }
        
        with pytest.raises(FrameworkConfigurationError, match="must have a 'name'"):
            loader.validate_function_config(config)
    
    def test_validate_function_config_missing_type(self):
        """Test validation rejects config without type."""
        loader = SemanticKernelConfigLoader()
        
        config = {
            "name": "test"
        }
        
        with pytest.raises(FrameworkConfigurationError, match="must specify 'type'"):
            loader.validate_function_config(config)
    
    def test_validate_function_config_invalid_type_value(self):
        """Test validation rejects invalid type value."""
        loader = SemanticKernelConfigLoader()
        
        config = {
            "name": "test",
            "type": "invalid"
        }
        
        with pytest.raises(FrameworkConfigurationError, match="must be 'semantic' or 'native'"):
            loader.validate_function_config(config)
    
    def test_validate_function_config_semantic_missing_prompt(self):
        """Test validation rejects semantic function without prompt."""
        loader = SemanticKernelConfigLoader()
        
        config = {
            "name": "test",
            "type": "semantic"
        }
        
        with pytest.raises(FrameworkConfigurationError, match="must have a 'prompt'"):
            loader.validate_function_config(config)
    
    def test_validate_function_config_native_missing_callable(self):
        """Test validation rejects native function without callable."""
        loader = SemanticKernelConfigLoader()
        
        config = {
            "name": "test",
            "type": "native"
        }
        
        with pytest.raises(FrameworkConfigurationError, match="must have a 'callable'"):
            loader.validate_function_config(config)
    
    def test_validate_plugin_config_valid(self):
        """Test validation of valid plugin config."""
        loader = SemanticKernelConfigLoader()
        
        config = {
            "name": "text_plugin",
            "functions": ["translate", "summarize"]
        }
        
        loader.validate_plugin_config(config)
    
    def test_validate_plugin_config_invalid_type(self):
        """Test validation rejects non-dict config."""
        loader = SemanticKernelConfigLoader()
        
        with pytest.raises(FrameworkConfigurationError, match="must be a dictionary"):
            loader.validate_plugin_config("not a dict")
    
    def test_validate_plugin_config_missing_name(self):
        """Test validation rejects config without name."""
        loader = SemanticKernelConfigLoader()
        
        config = {
            "functions": ["test"]
        }
        
        with pytest.raises(FrameworkConfigurationError, match="must have a 'name'"):
            loader.validate_plugin_config(config)
    
    def test_validate_plugin_config_missing_functions(self):
        """Test validation rejects config without functions."""
        loader = SemanticKernelConfigLoader()
        
        config = {
            "name": "test_plugin"
        }
        
        with pytest.raises(FrameworkConfigurationError, match="must have a 'functions' list"):
            loader.validate_plugin_config(config)
    
    def test_validate_plugin_config_invalid_functions(self):
        """Test validation rejects config with non-list functions."""
        loader = SemanticKernelConfigLoader()
        
        config = {
            "name": "test_plugin",
            "functions": "not a list"
        }
        
        with pytest.raises(FrameworkConfigurationError, match="must have a 'functions' list"):
            loader.validate_plugin_config(config)
    
    def test_create_semantic_function_config(self):
        """Test creating semantic function config."""
        loader = SemanticKernelConfigLoader()
        
        config = loader.create_semantic_function_config(
            prompt="Translate: {{$text}}",
            description="Translates text",
            max_tokens=100,
            temperature=0.7,
            top_p=0.9
        )
        
        assert config.prompt_template_config.template == "Translate: {{$text}}"
        assert config.prompt_template_config.description == "Translates text"
        assert config.prompt_template_config.execution_settings is not None
    
    def test_load_from_yaml_valid_file(self, tmp_path):
        """Test loading valid YAML config."""
        yaml_content = """
kernels:
  - id: main
    provider_name: openai

functions:
  - name: translate
    type: semantic
    prompt: "Translate: {{$text}}"
    plugin: text

plugins:
  - name: text
    functions:
      - translate
"""
        yaml_file = tmp_path / "test_config.yaml"
        yaml_file.write_text(yaml_content)
        
        loader = SemanticKernelConfigLoader()
        config = loader.load_from_yaml(str(yaml_file))
        
        assert "kernels" in config
        assert len(config["kernels"]) == 1
        assert config["kernels"][0]["provider_name"] == "openai"
        
        assert "functions" in config
        assert len(config["functions"]) == 1
        assert config["functions"][0]["name"] == "translate"
        
        assert "plugins" in config
        assert len(config["plugins"]) == 1
        assert config["plugins"][0]["name"] == "text"
    
    def test_load_from_yaml_file_not_found(self):
        """Test error when YAML file doesn't exist."""
        loader = SemanticKernelConfigLoader()
        
        with pytest.raises(FrameworkConfigurationError, match="Configuration file .* not found"):
            loader.load_from_yaml("nonexistent.yaml")
    
    def test_load_from_yaml_invalid_yaml(self, tmp_path):
        """Test error with invalid YAML syntax."""
        yaml_file = tmp_path / "invalid.yaml"
        yaml_file.write_text("invalid: yaml: syntax:")
        
        loader = SemanticKernelConfigLoader()
        
        with pytest.raises(FrameworkConfigurationError, match="Error parsing YAML"):
            loader.load_from_yaml(str(yaml_file))
    
    def test_load_full_config_integration(self, tmp_path):
        """Test full config loading with all components."""
        yaml_content = """
kernels:
  - id: main
    provider_name: openai
  - id: fast
    provider_name: gpt35

functions:
  - name: translate
    type: semantic
    prompt: "Translate to {{$language}}: {{$text}}"
    plugin: text
    description: "Translation function"
    max_tokens: 200
    temperature: 0.5
  
  - name: to_upper
    type: native
    callable: str.upper
    plugin: text

plugins:
  - name: text
    description: "Text processing"
    functions:
      - translate
      - to_upper

execution_settings:
  default_max_tokens: 500
  default_temperature: 0.7
"""
        yaml_file = tmp_path / "full_config.yaml"
        yaml_file.write_text(yaml_content)
        
        loader = SemanticKernelConfigLoader()
        config = loader.load_from_yaml(str(yaml_file))
        
        # Check kernels
        assert len(config["kernels"]) == 2
        assert config["kernels"][0]["id"] == "main"
        assert config["kernels"][1]["id"] == "fast"
        
        # Check functions
        assert len(config["functions"]) == 2
        semantic_func = next(f for f in config["functions"] if f["name"] == "translate")
        assert semantic_func["type"] == "semantic"
        assert "{{$language}}" in semantic_func["prompt"]
        assert semantic_func["max_tokens"] == 200
        
        native_func = next(f for f in config["functions"] if f["name"] == "to_upper")
        assert native_func["type"] == "native"
        
        # Check plugins
        assert len(config["plugins"]) == 1
        assert config["plugins"][0]["name"] == "text"
        assert len(config["plugins"][0]["functions"]) == 2
        
        # Check execution settings
        assert config["execution_settings"]["default_max_tokens"] == 500
    
    def test_get_kernel_config(self, tmp_path):
        """Test retrieving specific kernel config."""
        yaml_content = """
kernels:
  - id: kernel1
    provider_name: openai
  - id: kernel2
    provider_name: azure
"""
        yaml_file = tmp_path / "kernels.yaml"
        yaml_file.write_text(yaml_content)
        
        loader = SemanticKernelConfigLoader()
        config = loader.load_from_yaml(str(yaml_file))
        
        kernel2 = loader.get_kernel_config(config, "kernel2")
        assert kernel2["provider_name"] == "azure"
    
    def test_get_function_config(self, tmp_path):
        """Test retrieving specific function config."""
        yaml_content = """
functions:
  - name: func1
    type: semantic
    prompt: "Test 1"
  - name: func2
    type: semantic
    prompt: "Test 2"
"""
        yaml_file = tmp_path / "functions.yaml"
        yaml_file.write_text(yaml_content)
        
        loader = SemanticKernelConfigLoader()
        config = loader.load_from_yaml(str(yaml_file))
        
        func2 = loader.get_function_config(config, "func2")
        assert func2["prompt"] == "Test 2"
    
    def test_get_plugin_config(self, tmp_path):
        """Test retrieving specific plugin config."""
        yaml_content = """
plugins:
  - name: plugin1
    functions: [func1]
  - name: plugin2
    functions: [func2]
"""
        yaml_file = tmp_path / "plugins.yaml"
        yaml_file.write_text(yaml_content)
        
        loader = SemanticKernelConfigLoader()
        config = loader.load_from_yaml(str(yaml_file))
        
        plugin2 = loader.get_plugin_config(config, "plugin2")
        assert plugin2["functions"] == ["func2"]
    
    def test_load_config_with_defaults(self, tmp_path):
        """Test that config loading applies defaults."""
        yaml_content = """
kernels:
  - provider_name: openai
"""
        yaml_file = tmp_path / "minimal.yaml"
        yaml_file.write_text(yaml_content)
        
        loader = SemanticKernelConfigLoader()
        config = loader.load_from_yaml(str(yaml_file))
        
        # Kernel should get default ID if not specified
        assert "kernels" in config
        assert len(config["kernels"]) == 1
