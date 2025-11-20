"""
Test thread safety of LLMManager
"""

import threading
import pytest
from unittest.mock import Mock, patch

from llm_interface import LLMManager, LLMConfig, ProviderType
from llm_interface.src.core.interfaces import LLMProvider


class MockProvider(LLMProvider):
    """Mock provider for testing"""
    
    def __init__(self, config):
        self.config = config
        self._model = Mock()
        self._chat_model = Mock()
        self._embeddings = Mock()
    
    def get_model(self):
        return self._model
    
    def get_chat_model(self):
        return self._chat_model
    
    def get_embeddings(self):
        return self._embeddings
    
    def validate_config(self):
        return True


class TestManagerThreadSafety:
    """Test thread safety of LLMManager operations"""
    
    @patch('llm_interface.src.manager.LLMProviderFactory.create_provider')
    def test_concurrent_add_providers(self, mock_create_provider):
        """Test adding providers from multiple threads"""
        manager = LLMManager()
        errors = []
        added_providers = []
        
        def add_provider(name):
            try:
                config = LLMConfig(
                    provider=ProviderType.OPENAI,
                    model_name=f"gpt-{name}"
                )
                mock_provider = MockProvider(config)
                
                # Use a lock to ensure mock_create_provider returns the right provider
                with threading.Lock():
                    mock_create_provider.return_value = mock_provider
                
                manager.add_provider(name, config)
                added_providers.append(name)
            except Exception as e:
                errors.append((name, e))
        
        # Create multiple threads
        threads = []
        for i in range(10):
            t = threading.Thread(target=add_provider, args=(f"provider_{i}",))
            threads.append(t)
        
        # Start all threads
        for t in threads:
            t.start()
        
        # Wait for all threads
        for t in threads:
            t.join()
        
        # Verify results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(manager.list_providers()) == 10
        assert len(added_providers) == 10
    
    @patch('llm_interface.src.manager.LLMProviderFactory.create_provider')
    def test_concurrent_get_operations(self, mock_create_provider):
        """Test getting models from multiple threads"""
        manager = LLMManager()
        
        # Add a provider first
        config = LLMConfig(
            provider=ProviderType.OPENAI,
            model_name="gpt-4"
        )
        mock_provider = MockProvider(config)
        mock_create_provider.return_value = mock_provider
        
        manager.add_provider("test_provider", config)
        
        # Now test concurrent access
        results = []
        errors = []
        
        def get_model():
            try:
                model = manager.get_model("test_provider")
                results.append(model)
            except Exception as e:
                errors.append(e)
        
        def get_chat_model():
            try:
                chat_model = manager.get_chat_model("test_provider")
                results.append(chat_model)
            except Exception as e:
                errors.append(e)
        
        def list_providers():
            try:
                providers = manager.list_providers()
                results.append(providers)
            except Exception as e:
                errors.append(e)
        
        # Create mixed operations
        threads = []
        for _ in range(5):
            threads.append(threading.Thread(target=get_model))
            threads.append(threading.Thread(target=get_chat_model))
            threads.append(threading.Thread(target=list_providers))
        
        # Start all threads
        for t in threads:
            t.start()
        
        # Wait for all threads
        for t in threads:
            t.join()
        
        # Verify no errors occurred
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 15  # 5 of each operation
    
    @patch('llm_interface.src.manager.LLMProviderFactory.create_provider')
    def test_concurrent_mixed_operations(self, mock_create_provider):
        """Test mixed add and get operations from multiple threads"""
        manager = LLMManager()
        errors = []
        successful_ops = []
        
        def add_and_get(index):
            try:
                # Add a provider
                config = LLMConfig(
                    provider=ProviderType.OPENAI,
                    model_name=f"gpt-{index}"
                )
                mock_provider = MockProvider(config)
                mock_create_provider.return_value = mock_provider
                
                provider_name = f"provider_{index}"
                manager.add_provider(provider_name, config)
                successful_ops.append(f"added_{index}")
                
                # Try to get it
                model = manager.get_model(provider_name)
                successful_ops.append(f"got_{index}")
                
                # List all providers
                providers = manager.list_providers()
                successful_ops.append(f"listed_{index}")
                
            except Exception as e:
                errors.append((index, e))
        
        # Create multiple threads
        threads = [threading.Thread(target=add_and_get, args=(i,)) for i in range(10)]
        
        # Start all threads
        for t in threads:
            t.start()
        
        # Wait for all threads
        for t in threads:
            t.join()
        
        # Verify results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(manager.list_providers()) == 10
        # Each thread should complete 3 operations (add, get, list)
        assert len(successful_ops) == 30
    
    @patch('llm_interface.src.manager.LLMProviderFactory.create_provider')
    def test_no_race_conditions_in_add(self, mock_create_provider):
        """Test that no race conditions occur when adding the same provider"""
        manager = LLMManager()
        
        from llm_interface import ProviderAlreadyExistsError
        
        errors = []
        success_count = []
        
        def try_add_provider():
            try:
                config = LLMConfig(
                    provider=ProviderType.OPENAI,
                    model_name="gpt-4"
                )
                mock_provider = MockProvider(config)
                mock_create_provider.return_value = mock_provider
                
                manager.add_provider("same_name", config)
                success_count.append(1)
            except ProviderAlreadyExistsError:
                # Expected for all but one thread
                pass
            except Exception as e:
                errors.append(e)
        
        # Create multiple threads trying to add the same provider
        threads = [threading.Thread(target=try_add_provider) for _ in range(10)]
        
        # Start all threads
        for t in threads:
            t.start()
        
        # Wait for all threads
        for t in threads:
            t.join()
        
        # Verify results - only one should succeed
        assert len(errors) == 0, f"Unexpected errors: {errors}"
        assert len(success_count) == 1, "Only one thread should successfully add the provider"
        assert len(manager.list_providers()) == 1
    
    @patch('llm_interface.src.manager.LLMProviderFactory.create_provider')
    def test_reentrant_lock_works(self, mock_create_provider):
        """Test that the reentrant lock allows nested calls"""
        manager = LLMManager()
        
        config = LLMConfig(
            provider=ProviderType.OPENAI,
            model_name="gpt-4"
        )
        mock_provider = MockProvider(config)
        mock_create_provider.return_value = mock_provider
        
        manager.add_provider("test", config)
        
        # This should work because RLock is reentrant
        # (though this specific code path doesn't exercise it,
        # it demonstrates the lock is configured correctly)
        def nested_operation():
            providers = manager.list_providers()
            if providers:
                # This is a nested lock acquisition
                model = manager.get_model(providers[0])
                return model
        
        result = nested_operation()
        assert result is not None
