"""
Configuration loader for AutoGen framework.

Handles loading and validation of AutoGen-specific configurations.
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
import yaml
from ...config_loader import ConfigLoader
from ..exceptions import FrameworkConfigurationError


class AutoGenConfigLoader:
    """
    Load and validate AutoGen-specific configurations.
    
    Supports loading agent configurations, group chat settings,
    and conversation parameters from YAML or dictionaries.
    
    Example:
        >>> loader = AutoGenConfigLoader()
        >>> config = loader.from_yaml("autogen_config.yaml")
        >>> agents_config = config["agents"]
    """
    
    def __init__(self, base_config_loader: Optional[ConfigLoader] = None):
        """
        Initialize AutoGen config loader.
        
        Args:
            base_config_loader: Optional base ConfigLoader for provider configs
        """
        self.base_loader = base_config_loader or ConfigLoader()
    
    @staticmethod
    def from_yaml(file_path: str) -> Dict[str, Any]:
        """
        Load AutoGen configuration from YAML file.
        
        Args:
            file_path: Path to YAML configuration file
            
        Returns:
            Configuration dictionary
            
        Raises:
            FrameworkConfigurationError: If file not found or invalid YAML
            
        Example:
            >>> config = AutoGenConfigLoader.from_yaml("config.yaml")
        """
        try:
            path = Path(file_path)
            if not path.exists():
                raise FrameworkConfigurationError(
                    f"Configuration file not found: {file_path}"
                )
            
            with open(path, 'r') as f:
                config = yaml.safe_load(f)
            
            if not isinstance(config, dict):
                raise FrameworkConfigurationError(
                    "Configuration must be a dictionary"
                )
            
            return config
            
        except yaml.YAMLError as e:
            raise FrameworkConfigurationError(
                f"Invalid YAML in {file_path}: {str(e)}"
            ) from e
        except Exception as e:
            if isinstance(e, FrameworkConfigurationError):
                raise
            raise FrameworkConfigurationError(
                f"Failed to load configuration: {str(e)}"
            ) from e
    
    @staticmethod
    def from_dict(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and return AutoGen configuration from dictionary.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Validated configuration
            
        Raises:
            FrameworkConfigurationError: If configuration is invalid
        """
        if not isinstance(config, dict):
            raise FrameworkConfigurationError(
                "Configuration must be a dictionary"
            )
        
        return config
    
    @staticmethod
    def validate_agent_config(agent_config: Dict[str, Any]) -> bool:
        """
        Validate individual agent configuration.
        
        Args:
            agent_config: Agent configuration dictionary
            
        Returns:
            True if valid
            
        Raises:
            FrameworkConfigurationError: If configuration is invalid
        """
        required_fields = ["name", "provider_name"]
        
        for field in required_fields:
            if field not in agent_config:
                raise FrameworkConfigurationError(
                    f"Agent config missing required field: {field}"
                )
        
        # Validate human_input_mode if present
        if "human_input_mode" in agent_config:
            valid_modes = ["ALWAYS", "NEVER", "TERMINATE"]
            if agent_config["human_input_mode"] not in valid_modes:
                raise FrameworkConfigurationError(
                    f"Invalid human_input_mode. Must be one of: {valid_modes}"
                )
        
        return True
    
    @staticmethod
    def validate_group_chat_config(group_chat_config: Dict[str, Any]) -> bool:
        """
        Validate group chat configuration.
        
        Args:
            group_chat_config: Group chat configuration dictionary
            
        Returns:
            True if valid
            
        Raises:
            FrameworkConfigurationError: If configuration is invalid
        """
        required_fields = ["name", "agents"]
        
        for field in required_fields:
            if field not in group_chat_config:
                raise FrameworkConfigurationError(
                    f"Group chat config missing required field: {field}"
                )
        
        # Validate agents list
        agents = group_chat_config["agents"]
        if not isinstance(agents, list):
            raise FrameworkConfigurationError(
                "Group chat 'agents' must be a list"
            )
        
        if len(agents) < 2:
            raise FrameworkConfigurationError(
                "Group chat must have at least 2 agents"
            )
        
        # Validate speaker_selection_method if present
        if "speaker_selection_method" in group_chat_config:
            valid_methods = ["auto", "manual", "random", "round_robin"]
            if group_chat_config["speaker_selection_method"] not in valid_methods:
                raise FrameworkConfigurationError(
                    f"Invalid speaker_selection_method. Must be one of: {valid_methods}"
                )
        
        return True
    
    def load_full_config(self, file_path: str) -> Dict[str, Any]:
        """
        Load complete configuration including providers and AutoGen settings.
        
        Args:
            file_path: Path to configuration file
            
        Returns:
            Complete configuration dictionary with:
                - providers: Provider configurations
                - autogen: AutoGen-specific settings
                
        Example:
            >>> loader = AutoGenConfigLoader()
            >>> config = loader.load_full_config("full_config.yaml")
            >>> provider_configs = config["providers"]
            >>> autogen_config = config["autogen"]
        """
        config = self.from_yaml(file_path)
        
        # Validate structure
        if "providers" not in config and "autogen" not in config:
            raise FrameworkConfigurationError(
                "Configuration must contain 'providers' or 'autogen' section"
            )
        
        # Validate agents if present
        if "autogen" in config and "agents" in config["autogen"]:
            agents = config["autogen"]["agents"]
            if isinstance(agents, list):
                for agent_config in agents:
                    self.validate_agent_config(agent_config)
            elif isinstance(agents, dict):
                for agent_name, agent_config in agents.items():
                    if "name" not in agent_config:
                        agent_config["name"] = agent_name
                    self.validate_agent_config(agent_config)
        
        # Validate group chats if present
        if "autogen" in config and "group_chats" in config["autogen"]:
            group_chats = config["autogen"]["group_chats"]
            if isinstance(group_chats, list):
                for gc_config in group_chats:
                    self.validate_group_chat_config(gc_config)
            elif isinstance(group_chats, dict):
                for gc_name, gc_config in group_chats.items():
                    if "name" not in gc_config:
                        gc_config["name"] = gc_name
                    self.validate_group_chat_config(gc_config)
        
        return config
    
    @staticmethod
    def create_agent_config(
        name: str,
        provider_name: str,
        system_message: Optional[str] = None,
        human_input_mode: str = "NEVER",
        max_consecutive_auto_reply: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create agent configuration dictionary.
        
        Args:
            name: Agent name
            provider_name: Provider to use
            system_message: System message
            human_input_mode: Input mode
            max_consecutive_auto_reply: Max auto replies
            **kwargs: Additional parameters
            
        Returns:
            Agent configuration dictionary
        """
        config = {
            "name": name,
            "provider_name": provider_name,
            "human_input_mode": human_input_mode,
        }
        
        if system_message:
            config["system_message"] = system_message
        
        if max_consecutive_auto_reply is not None:
            config["max_consecutive_auto_reply"] = max_consecutive_auto_reply
        
        config.update(kwargs)
        return config
    
    @staticmethod
    def create_group_chat_config(
        name: str,
        agents: List[str],
        max_round: int = 10,
        speaker_selection_method: str = "auto",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create group chat configuration dictionary.
        
        Args:
            name: Group chat name
            agents: List of agent names
            max_round: Maximum rounds
            speaker_selection_method: Selection method
            **kwargs: Additional parameters
            
        Returns:
            Group chat configuration dictionary
        """
        config = {
            "name": name,
            "agents": agents,
            "max_round": max_round,
            "speaker_selection_method": speaker_selection_method,
        }
        
        config.update(kwargs)
        return config
