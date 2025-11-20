"""
Test ConfigLoader module with environment variable expansion
"""

import os
import pytest
import tempfile
import yaml
from pathlib import Path

from llm_interface import ConfigLoader, LLMConfig, ProviderType


class TestConfigLoaderEnvExpansion:
    """Test environment variable expansion in ConfigLoader"""
    
    def test_expand_env_vars_with_braces(self):
        """Test expanding ${VAR} format"""
        os.environ['TEST_API_KEY'] = 'test-key-123'
        
        result = ConfigLoader._expand_env_vars('${TEST_API_KEY}')
        assert result == 'test-key-123'
        
        # Clean up
        del os.environ['TEST_API_KEY']
    
    def test_expand_env_vars_without_braces(self):
        """Test expanding $VAR format"""
        os.environ['TEST_MODEL'] = 'gpt-4'
        
        result = ConfigLoader._expand_env_vars('$TEST_MODEL')
        assert result == 'gpt-4'
        
        # Clean up
        del os.environ['TEST_MODEL']
    
    def test_expand_env_vars_in_string(self):
        """Test expanding variables within a larger string"""
        os.environ['TEST_ENDPOINT'] = 'https://api.example.com'
        
        result = ConfigLoader._expand_env_vars('${TEST_ENDPOINT}/v1/chat')
        assert result == 'https://api.example.com/v1/chat'
        
        # Clean up
        del os.environ['TEST_ENDPOINT']
    
    def test_expand_env_vars_missing_var(self):
        """Test that missing variables are left as-is"""
        result = ConfigLoader._expand_env_vars('${NONEXISTENT_VAR}')
        assert result == '${NONEXISTENT_VAR}'
    
    def test_expand_env_vars_in_dict(self):
        """Test expanding variables in a dictionary"""
        os.environ['TEST_KEY'] = 'my-secret-key'
        
        input_dict = {
            'api_key': '${TEST_KEY}',
            'model': 'gpt-4',
            'nested': {
                'key': '${TEST_KEY}'
            }
        }
        
        result = ConfigLoader._expand_env_vars(input_dict)
        
        assert result['api_key'] == 'my-secret-key'
        assert result['model'] == 'gpt-4'
        assert result['nested']['key'] == 'my-secret-key'
        
        # Clean up
        del os.environ['TEST_KEY']
    
    def test_expand_env_vars_in_list(self):
        """Test expanding variables in a list"""
        os.environ['TEST_ITEM'] = 'expanded-value'
        
        input_list = ['${TEST_ITEM}', 'plain-value', '${TEST_ITEM}']
        result = ConfigLoader._expand_env_vars(input_list)
        
        assert result == ['expanded-value', 'plain-value', 'expanded-value']
        
        # Clean up
        del os.environ['TEST_ITEM']
    
    def test_from_yaml_with_env_vars(self):
        """Test loading YAML with environment variable expansion"""
        # Set up environment
        os.environ['TEST_OPENAI_KEY'] = 'sk-test-123'
        os.environ['TEST_MODEL_NAME'] = 'gpt-4'
        
        # Create temporary YAML file
        yaml_content = """
providers:
  openai:
    provider: "openai"
    model_name: "${TEST_MODEL_NAME}"
    api_key: "${TEST_OPENAI_KEY}"
    temperature: 0.7
    max_tokens: 1000
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name
        
        try:
            # Load configuration
            configs = ConfigLoader.from_yaml(temp_path)
            
            # Verify expansion
            assert 'openai' in configs
            assert configs['openai'].model_name == 'gpt-4'
            assert configs['openai'].api_key == 'sk-test-123'
            assert configs['openai'].provider == ProviderType.OPENAI
            
        finally:
            # Clean up
            os.unlink(temp_path)
            del os.environ['TEST_OPENAI_KEY']
            del os.environ['TEST_MODEL_NAME']
    
    def test_from_dict_with_env_vars(self):
        """Test loading from dict with environment variable expansion"""
        os.environ['TEST_AZURE_KEY'] = 'azure-key-456'
        
        config_dict = {
            'providers': {
                'azure': {
                    'provider': 'azure',
                    'model_name': 'gpt-4',
                    'api_key': '${TEST_AZURE_KEY}',
                    'azure_endpoint': 'https://test.openai.azure.com',
                    'azure_deployment': 'gpt-4-deployment'
                }
            }
        }
        
        configs = ConfigLoader.from_dict(config_dict)
        
        assert configs['azure'].api_key == 'azure-key-456'
        
        # Clean up
        del os.environ['TEST_AZURE_KEY']
    
    def test_expand_multiple_vars_in_one_string(self):
        """Test expanding multiple variables in a single string"""
        os.environ['TEST_HOST'] = 'example.com'
        os.environ['TEST_PORT'] = '8080'
        
        result = ConfigLoader._expand_env_vars('https://${TEST_HOST}:${TEST_PORT}/api')
        assert result == 'https://example.com:8080/api'
        
        # Clean up
        del os.environ['TEST_HOST']
        del os.environ['TEST_PORT']


class TestConfigLoaderThreadSafety:
    """Test thread safety of ConfigLoader (it's stateless, so should be safe)"""
    
    def test_concurrent_loading(self):
        """Test that ConfigLoader can be used concurrently"""
        import threading
        
        os.environ['CONCURRENT_TEST_KEY'] = 'test-value'
        
        results = []
        errors = []
        
        def load_config():
            try:
                config_dict = {
                    'providers': {
                        'test': {
                            'provider': 'openai',
                            'model_name': 'gpt-3.5-turbo',
                            'api_key': '${CONCURRENT_TEST_KEY}'
                        }
                    }
                }
                configs = ConfigLoader.from_dict(config_dict)
                results.append(configs['test'].api_key)
            except Exception as e:
                errors.append(e)
        
        # Create multiple threads
        threads = [threading.Thread(target=load_config) for _ in range(10)]
        
        # Start all threads
        for t in threads:
            t.start()
        
        # Wait for all threads
        for t in threads:
            t.join()
        
        # Verify results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 10
        assert all(r == 'test-value' for r in results)
        
        # Clean up
        del os.environ['CONCURRENT_TEST_KEY']
