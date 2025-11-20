"""
AutoGen framework adapter.

Provides integration between llm_interface providers and AutoGen agents.
"""

from typing import Dict, Any, List, Optional, Union
from ...manager import LLMManager
from ..base import FrameworkAdapter
from ..exceptions import FrameworkConfigurationError, FrameworkExecutionError

try:
    from autogen import ConversableAgent, GroupChat, GroupChatManager
    AUTOGEN_AVAILABLE = True
except ImportError:
    AUTOGEN_AVAILABLE = False
    # Create placeholder types for when autogen is not installed
    class ConversableAgent:  # type: ignore
        """Placeholder for autogen.ConversableAgent"""
        pass
    
    class GroupChat:  # type: ignore
        """Placeholder for autogen.GroupChat"""
        pass
    
    class GroupChatManager:  # type: ignore
        """Placeholder for autogen.GroupChatManager"""
        pass


class AutoGenAdapter(FrameworkAdapter):
    """
    Adapter for Microsoft AutoGen framework.
    
    Enables creation of AutoGen agents using unified provider interface,
    supporting multi-agent conversations and group chats.
    
    Example:
        >>> manager = LLMManager()
        >>> manager.add_provider("openai", openai_config)
        >>> adapter = AutoGenAdapter(manager)
        >>> agent = adapter.create_agent(
        ...     name="assistant",
        ...     provider_name="openai",
        ...     system_message="You are a helpful assistant"
        ... )
    """
    
    def __init__(self, llm_manager: LLMManager):
        """
        Initialize AutoGen adapter.
        
        Args:
            llm_manager: LLMManager instance with configured providers
            
        Raises:
            FrameworkConfigurationError: If AutoGen is not installed
        """
        if not AUTOGEN_AVAILABLE:
            raise FrameworkConfigurationError(
                "AutoGen is not installed. Install with: pip install pyautogen"
            )
        super().__init__(llm_manager)
        self.agents: Dict[str, ConversableAgent] = {}
        self.group_chats: Dict[str, GroupChat] = {}
    
    @property
    def framework_name(self) -> str:
        """Return framework identifier."""
        return "autogen"
    
    @property
    def framework_version(self) -> str:
        """Return minimum supported framework version."""
        return "0.2.0"
    
    def create_agent(
        self,
        name: str,
        provider_name: str,
        system_message: Optional[str] = None,
        human_input_mode: str = "NEVER",
        max_consecutive_auto_reply: Optional[int] = None,
        code_execution_config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> ConversableAgent:
        """
        Create an AutoGen ConversableAgent with specified provider.
        
        Args:
            name: Agent name/identifier
            provider_name: Name of provider from LLMManager
            system_message: System message for agent behavior
            human_input_mode: When to request human input ("ALWAYS", "NEVER", "TERMINATE")
            max_consecutive_auto_reply: Max auto-replies before stopping
            code_execution_config: Configuration for code execution
            **kwargs: Additional ConversableAgent parameters
            
        Returns:
            Configured ConversableAgent instance
            
        Raises:
            FrameworkConfigurationError: If provider not found or config invalid
            
        Example:
            >>> agent = adapter.create_agent(
            ...     name="coder",
            ...     provider_name="gpt4",
            ...     system_message="You are an expert Python developer",
            ...     max_consecutive_auto_reply=10
            ... )
        """
        try:
            # Get provider config
            if provider_name not in self.llm_manager.providers:
                raise FrameworkConfigurationError(
                    f"Provider '{provider_name}' not found in LLMManager"
                )
            
            provider = self.llm_manager.providers[provider_name]
            config = provider.config
            
            # Build AutoGen LLM config
            llm_config = {
                "model": config.model_name,
                "temperature": config.temperature,
            }
            
            # Add API key based on provider type
            provider_type = str(config.provider).lower()
            if provider_type == "openai":
                llm_config["api_key"] = config.api_key
            elif provider_type == "azure":
                llm_config["api_key"] = config.api_key
                llm_config["api_type"] = "azure"
                llm_config["api_base"] = config.azure_endpoint
                llm_config["api_version"] = config.api_version or "2024-02-15-preview"
            elif provider_type == "anthropic":
                llm_config["api_key"] = config.api_key
                llm_config["model"] = config.model_name or "claude-3-sonnet-20240229"
            
            # Add optional parameters
            if config.max_tokens:
                llm_config["max_tokens"] = config.max_tokens
            
            # Create agent
            agent = ConversableAgent(
                name=name,
                system_message=system_message or f"You are {name}",
                llm_config=llm_config,
                human_input_mode=human_input_mode,
                max_consecutive_auto_reply=max_consecutive_auto_reply,
                code_execution_config=code_execution_config or False,
                **kwargs
            )
            
            # Store agent reference
            self.agents[name] = agent
            return agent
            
        except Exception as e:
            if isinstance(e, FrameworkConfigurationError):
                raise
            raise FrameworkConfigurationError(
                f"Failed to create AutoGen agent '{name}': {str(e)}"
            ) from e
    
    def create_user_proxy(
        self,
        name: str,
        human_input_mode: str = "NEVER",
        max_consecutive_auto_reply: Optional[int] = None,
        code_execution_config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> ConversableAgent:
        """
        Create a UserProxyAgent (agent without LLM).
        
        Args:
            name: Agent name
            human_input_mode: When to request input
            max_consecutive_auto_reply: Max auto-replies
            code_execution_config: Code execution settings
            **kwargs: Additional parameters
            
        Returns:
            ConversableAgent configured as user proxy
            
        Example:
            >>> user_proxy = adapter.create_user_proxy(
            ...     name="user",
            ...     code_execution_config={"work_dir": "coding"}
            ... )
        """
        agent = ConversableAgent(
            name=name,
            llm_config=False,  # No LLM for user proxy
            human_input_mode=human_input_mode,
            max_consecutive_auto_reply=max_consecutive_auto_reply,
            code_execution_config=code_execution_config or False,
            **kwargs
        )
        
        self.agents[name] = agent
        return agent
    
    def create_group_chat(
        self,
        name: str,
        agents: List[ConversableAgent],
        max_round: int = 10,
        admin_name: Optional[str] = None,
        speaker_selection_method: str = "auto",
        allow_repeat_speaker: bool = True,
        **kwargs
    ) -> GroupChat:
        """
        Create a group chat with multiple agents.
        
        Args:
            name: Group chat identifier
            agents: List of participating agents
            max_round: Maximum conversation rounds
            admin_name: Name of admin agent (if any)
            speaker_selection_method: How to select next speaker
            allow_repeat_speaker: Whether same agent can speak consecutively
            **kwargs: Additional GroupChat parameters
            
        Returns:
            GroupChat instance
            
        Example:
            >>> group_chat = adapter.create_group_chat(
            ...     name="dev_team",
            ...     agents=[coder, reviewer, tester],
            ...     max_round=20
            ... )
        """
        try:
            group_chat = GroupChat(
                agents=agents,
                messages=[],
                max_round=max_round,
                admin_name=admin_name,
                speaker_selection_method=speaker_selection_method,
                allow_repeat_speaker=allow_repeat_speaker,
                **kwargs
            )
            
            self.group_chats[name] = group_chat
            return group_chat
            
        except Exception as e:
            raise FrameworkConfigurationError(
                f"Failed to create group chat '{name}': {str(e)}"
            ) from e
    
    def create_group_chat_manager(
        self,
        group_chat: GroupChat,
        provider_name: str,
        system_message: Optional[str] = None,
        **kwargs
    ) -> GroupChatManager:
        """
        Create manager for group chat using specified provider.
        
        Args:
            group_chat: GroupChat to manage
            provider_name: Provider for manager's LLM
            system_message: Manager's system message
            **kwargs: Additional GroupChatManager parameters
            
        Returns:
            GroupChatManager instance
            
        Example:
            >>> manager = adapter.create_group_chat_manager(
            ...     group_chat=group_chat,
            ...     provider_name="gpt4",
            ...     system_message="Coordinate the discussion"
            ... )
        """
        try:
            # Get provider config
            if provider_name not in self.llm_manager.providers:
                raise FrameworkConfigurationError(
                    f"Provider '{provider_name}' not found"
                )
            
            provider = self.llm_manager.providers[provider_name]
            config = provider.config
            
            # Build LLM config
            llm_config = {
                "model": config.model_name,
                "temperature": config.temperature,
                "api_key": config.api_key,
            }
            
            if config.max_tokens:
                llm_config["max_tokens"] = config.max_tokens
            
            # Handle Azure
            provider_type = str(config.provider).lower()
            if provider_type == "azure":
                llm_config["api_type"] = "azure"
                llm_config["api_base"] = config.azure_endpoint
                llm_config["api_version"] = config.api_version or "2024-02-15-preview"
            
            manager = GroupChatManager(
                groupchat=group_chat,
                llm_config=llm_config,
                system_message=system_message or "Group chat manager",
                **kwargs
            )
            
            return manager
            
        except Exception as e:
            if isinstance(e, FrameworkConfigurationError):
                raise
            raise FrameworkConfigurationError(
                f"Failed to create group chat manager: {str(e)}"
            ) from e
    
    def initiate_chat(
        self,
        sender: ConversableAgent,
        recipient: ConversableAgent,
        message: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Initiate a conversation between two agents.
        
        Args:
            sender: Agent initiating the conversation
            recipient: Agent receiving the message
            message: Initial message
            **kwargs: Additional chat parameters
            
        Returns:
            Chat result dictionary
            
        Example:
            >>> result = adapter.initiate_chat(
            ...     sender=user_proxy,
            ...     recipient=assistant,
            ...     message="Write a Python function to sort a list"
            ... )
        """
        try:
            result = sender.initiate_chat(
                recipient=recipient,
                message=message,
                **kwargs
            )
            return result
            
        except Exception as e:
            raise FrameworkExecutionError(
                f"Chat execution failed: {str(e)}"
            ) from e
    
    def create_model(self, provider_name: str, **kwargs):
        """
        Create a model instance (AutoGen uses agents, not standalone models).
        
        For AutoGen, use create_agent() instead.
        
        Args:
            provider_name: Provider name
            **kwargs: Additional parameters
            
        Returns:
            None (use create_agent instead)
        """
        raise NotImplementedError(
            "AutoGen uses agents instead of standalone models. "
            "Use create_agent() or create_user_proxy() instead."
        )
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate AutoGen-specific configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            True if valid
            
        Raises:
            FrameworkConfigurationError: If configuration is invalid
        """
        if not isinstance(config, dict):
            raise FrameworkConfigurationError("Config must be a dictionary")
        
        # Check for agents configuration
        if "agents" in config:
            agents_config = config["agents"]
            if not isinstance(agents_config, (list, dict)):
                raise FrameworkConfigurationError(
                    "agents config must be list or dict"
                )
        
        return True
    
    def get_agent(self, name: str) -> ConversableAgent:
        """
        Get a previously created agent by name.
        
        Args:
            name: Agent name
            
        Returns:
            ConversableAgent instance
            
        Raises:
            KeyError: If agent not found
        """
        if name not in self.agents:
            raise KeyError(f"Agent '{name}' not found")
        return self.agents[name]
    
    def list_agents(self) -> List[str]:
        """
        List all created agent names.
        
        Returns:
            List of agent names
        """
        return list(self.agents.keys())
    
    def get_group_chat(self, name: str) -> GroupChat:
        """
        Get a previously created group chat by name.
        
        Args:
            name: Group chat name
            
        Returns:
            GroupChat instance
            
        Raises:
            KeyError: If group chat not found
        """
        if name not in self.group_chats:
            raise KeyError(f"Group chat '{name}' not found")
        return self.group_chats[name]
    
    def list_group_chats(self) -> List[str]:
        """
        List all created group chat names.
        
        Returns:
            List of group chat names
        """
        return list(self.group_chats.keys())
