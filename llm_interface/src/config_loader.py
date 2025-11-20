import os
import re
import yaml
from typing import Dict, Any, Optional
from pathlib import Path
from .core.config import LLMConfig, ProviderType

class ConfigLoader:
    """Load LLM configurations from various sources"""
    
    @staticmethod
    def _expand_env_vars(value: Any) -> Any:
        """Expand environment variables in string values.
        
        Supports both ${VAR_NAME} and $VAR_NAME formats.
        If variable is not found, returns the original placeholder.
        
        Args:
            value: The value to expand (can be string, dict, list, or other)
            
        Returns:
            Value with environment variables expanded
        """
        if isinstance(value, str):
            # Pattern matches ${VAR} or $VAR formats
            pattern = r'\$\{([^}]+)\}|\$([A-Z_][A-Z0-9_]*)'
            
            def replacer(match):
                var_name = match.group(1) or match.group(2)
                return os.getenv(var_name, match.group(0))
            
            return re.sub(pattern, replacer, value)
        elif isinstance(value, dict):
            return {k: ConfigLoader._expand_env_vars(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [ConfigLoader._expand_env_vars(item) for item in value]
        return value
    
    @staticmethod
    def from_yaml(file_path: str) -> Dict[str, LLMConfig]:
        """Load configurations from YAML file with environment variable expansion"""
        try:
            with open(file_path, 'r') as file:
                config_data = yaml.safe_load(file)
            
            if not isinstance(config_data, dict):
                raise ValueError("YAML file must contain a dictionary")
            
            # Expand environment variables in the config data
            config_data = ConfigLoader._expand_env_vars(config_data)
            
            return ConfigLoader._parse_config_data(config_data)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML format: {e}")
    
    @staticmethod
    def from_dict(config_dict: Dict[str, Any]) -> Dict[str, LLMConfig]:
        """Load configurations from dictionary with environment variable expansion"""
        # Expand environment variables in the config data
        config_dict = ConfigLoader._expand_env_vars(config_dict)
        return ConfigLoader._parse_config_data(config_dict)
    
    @staticmethod
    def from_env(prefix: str = "LLM_") -> Dict[str, LLMConfig]:
        """Load configurations from environment variables"""
        providers = {}
        
        # Look for environment variables with the prefix
        for key, value in os.environ.items():
            if key.startswith(prefix):
                # Parse environment variable structure
                # LLM_OPENAI_API_KEY, LLM_OPENAI_MODEL, etc.
                parts = key[len(prefix):].split('_')
                if len(parts) >= 2:
                    provider_name = parts[0].lower()
                    config_key = '_'.join(parts[1:]).lower()
                    
                    if provider_name not in providers:
                        providers[provider_name] = {}
                    providers[provider_name][config_key] = value
        
        return ConfigLoader._parse_config_data({"providers": providers})
    
    @staticmethod
    def _parse_config_data(config_data: Dict[str, Any]) -> Dict[str, LLMConfig]:
        """Parse raw configuration data into LLMConfig objects"""
        providers = {}
        providers_config = config_data.get('providers', {})
        
        if not isinstance(providers_config, dict):
            raise ValueError("Providers configuration must be a dictionary")
        
        for name, config in providers_config.items():
            if not isinstance(config, dict):
                raise ValueError(f"Provider '{name}' configuration must be a dictionary")
            
            providers[name] = ConfigLoader._create_llm_config(name, config)
        
        return providers
    
    @staticmethod
    def _create_llm_config(name: str, config: Dict[str, Any]) -> LLMConfig:
        """Create LLMConfig from dictionary"""
        try:
            # Handle provider type
            provider_str = config.get('provider', '').lower()
            provider_type = ConfigLoader._get_provider_type(provider_str)
            
            # Handle API key with environment variable fallback
            api_key = config.get('api_key')
            if not api_key and provider_str:
                env_var_map = {
                    'openai': 'OPENAI_API_KEY',
                    'azure': 'AZURE_OPENAI_API_KEY',
                    'anthropic': 'ANTHROPIC_API_KEY',
                    # 'gemini': 'GEMINI_API_KEY'  # Not yet implemented
                }
                env_var = env_var_map.get(provider_str)
                if env_var:
                    api_key = os.getenv(env_var)
            
            # Create LLMConfig
            return LLMConfig(
                provider=provider_type,
                model_name=config.get('model_name', ''),
                api_key=api_key,
                base_url=config.get('base_url'),
                temperature=float(config.get('temperature', 0.7)),
                max_tokens=int(config.get('max_tokens', 1000)),
                azure_endpoint=config.get('azure_endpoint'),
                azure_deployment=config.get('azure_deployment'),
                azure_api_version=config.get('azure_api_version')
            )
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid configuration for provider '{name}': {e}")
    
    @staticmethod
    def _get_provider_type(provider_str: str) -> ProviderType:
        """Convert string to ProviderType enum"""
        provider_map = {
            'openai': ProviderType.OPENAI,
            'azure': ProviderType.AZURE,
            'ollama': ProviderType.OLLAMA,
            'anthropic': ProviderType.ANTHROPIC,
            # 'gemini': ProviderType.GEMINI  # Not yet implemented
        }
        
        if provider_str not in provider_map:
            raise ValueError(f"Unsupported provider: {provider_str}")
        
        return provider_map[provider_str]
    
    @staticmethod
    def merge_configs(*configs: Dict[str, LLMConfig]) -> Dict[str, LLMConfig]:
        """Merge multiple configuration dictionaries"""
        merged = {}
        for config in configs:
            merged.update(config)
        return merged

# Convenience functions
def load_config(file_path: Optional[str] = None, 
                config_dict: Optional[Dict] = None,
                use_env: bool = False) -> Dict[str, LLMConfig]:
    """Load configuration from multiple sources"""
    configs = []
    
    if file_path:
        configs.append(ConfigLoader.from_yaml(file_path))
    
    if config_dict:
        configs.append(ConfigLoader.from_dict(config_dict))
    
    if use_env:
        configs.append(ConfigLoader.from_env())
    
    if not configs:
        # Try default locations
        default_paths = ['config.yaml', 'llm_config.yaml', '.llm_config.yaml']
        for path in default_paths:
            if Path(path).exists():
                configs.append(ConfigLoader.from_yaml(path))
                break
    
    if not configs:
        raise ValueError("No configuration source provided and no default config found")
    
    return ConfigLoader.merge_configs(*configs)
