"""LangChain-specific configuration loader.

Extends the base config loader with LangChain-specific settings and patterns.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pathlib import Path
import yaml

from llm_interface.src.framework.exceptions import FrameworkConfigurationError


@dataclass
class LangChainNodeConfig:
    """Configuration for a LangChain workflow node.
    
    Attributes:
        name: Node identifier
        type: Node type (e.g., 'prompt', 'chain', 'tool')
        provider: LLM provider for this node
        prompt_template: Template for prompt nodes
        metadata: Additional node configuration
    """
    name: str
    type: str
    provider: str
    prompt_template: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LangChainWorkflowConfig:
    """Configuration for a LangChain workflow.
    
    Attributes:
        name: Workflow name
        description: Human-readable description
        nodes: List of LangChainNodeConfig
        edges: List of edges between nodes
        default_provider: Default provider if not specified
        metadata: Additional configuration
    """
    name: str
    description: str
    nodes: List[LangChainNodeConfig] = field(default_factory=list)
    edges: List[tuple] = field(default_factory=list)
    default_provider: str = "openai"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> tuple[bool, str]:
        """Validate workflow configuration.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not self.name:
            return False, "Workflow name required"
        
        if not self.nodes:
            return False, "At least one node required"
        
        # Validate nodes have required fields
        for node in self.nodes:
            if not node.name:
                return False, "Node name required"
            if not node.type:
                return False, f"Node '{node.name}' missing type"
            if not node.provider and not self.default_provider:
                return False, f"Node '{node.name}' missing provider"
        
        # Validate edges reference existing nodes
        node_names = {node.name for node in self.nodes}
        for source, target in self.edges:
            if source not in node_names:
                return False, f"Edge source '{source}' not found in nodes"
            if target not in node_names:
                return False, f"Edge target '{target}' not found in nodes"
        
        return True, ""


class LangChainConfigLoader:
    """Loader for LangChain workflow configurations.
    
    Supports loading from YAML files, dictionaries, and building programmatically.
    
    Example:
        >>> loader = LangChainConfigLoader()
        >>> config = loader.load_from_file("workflows.yaml")
        >>> config = loader.load_from_dict({
        ...     "name": "my_workflow",
        ...     "nodes": [...]
        ... })
    """
    
    @staticmethod
    def load_from_file(filepath: str) -> LangChainWorkflowConfig:
        """Load configuration from YAML file.
        
        Args:
            filepath: Path to YAML configuration file
            
        Returns:
            LangChainWorkflowConfig instance
            
        Raises:
            FrameworkConfigurationError: If file not found or invalid
        """
        path = Path(filepath)
        if not path.exists():
            raise FrameworkConfigurationError(
                f"Configuration file not found: {filepath}",
                framework="LangChain"
            )
        
        try:
            with open(path, 'r') as f:
                data = yaml.safe_load(f)
            return LangChainConfigLoader.load_from_dict(data)
        except yaml.YAMLError as e:
            raise FrameworkConfigurationError(
                f"Invalid YAML in {filepath}: {str(e)}",
                framework="LangChain"
            ) from e
        except Exception as e:
            raise FrameworkConfigurationError(
                f"Error loading configuration: {str(e)}",
                framework="LangChain"
            ) from e
    
    @staticmethod
    def load_from_dict(data: Dict[str, Any]) -> LangChainWorkflowConfig:
        """Load configuration from dictionary.
        
        Args:
            data: Configuration dictionary
            
        Returns:
            LangChainWorkflowConfig instance
            
        Raises:
            FrameworkConfigurationError: If configuration is invalid
        """
        try:
            # Parse nodes
            nodes_data = data.get("nodes", [])
            nodes = []
            for node_data in nodes_data:
                node = LangChainNodeConfig(
                    name=node_data.get("name", ""),
                    type=node_data.get("type", ""),
                    provider=node_data.get("provider", ""),
                    prompt_template=node_data.get("prompt_template"),
                    metadata=node_data.get("metadata", {})
                )
                nodes.append(node)
            
            # Parse edges
            edges_data = data.get("edges", [])
            edges = [tuple(edge) for edge in edges_data]
            
            # Create config
            config = LangChainWorkflowConfig(
                name=data.get("name", ""),
                description=data.get("description", ""),
                nodes=nodes,
                edges=edges,
                default_provider=data.get("default_provider", "openai"),
                metadata=data.get("metadata", {})
            )
            
            # Validate
            is_valid, error = config.validate()
            if not is_valid:
                raise FrameworkConfigurationError(error, framework="LangChain")
            
            return config
        except FrameworkConfigurationError:
            raise
        except Exception as e:
            raise FrameworkConfigurationError(
                f"Error parsing configuration: {str(e)}",
                framework="LangChain"
            ) from e
    
    @staticmethod
    def to_dict(config: LangChainWorkflowConfig) -> Dict[str, Any]:
        """Convert configuration to dictionary.
        
        Args:
            config: LangChainWorkflowConfig instance
            
        Returns:
            Dictionary representation
        """
        return {
            "name": config.name,
            "description": config.description,
            "default_provider": config.default_provider,
            "nodes": [
                {
                    "name": node.name,
                    "type": node.type,
                    "provider": node.provider,
                    "prompt_template": node.prompt_template,
                    "metadata": node.metadata,
                }
                for node in config.nodes
            ],
            "edges": [list(edge) for edge in config.edges],
            "metadata": config.metadata,
        }
    
    @staticmethod
    def to_yaml(config: LangChainWorkflowConfig) -> str:
        """Convert configuration to YAML string.
        
        Args:
            config: LangChainWorkflowConfig instance
            
        Returns:
            YAML string representation
        """
        data = LangChainConfigLoader.to_dict(config)
        return yaml.dump(data, default_flow_style=False, sort_keys=False)
    
    @staticmethod
    def save_to_file(config: LangChainWorkflowConfig, filepath: str) -> None:
        """Save configuration to YAML file.
        
        Args:
            config: LangChainWorkflowConfig instance
            filepath: Path to save to
            
        Raises:
            FrameworkConfigurationError: If save fails
        """
        try:
            path = Path(filepath)
            path.parent.mkdir(parents=True, exist_ok=True)
            yaml_content = LangChainConfigLoader.to_yaml(config)
            path.write_text(yaml_content)
        except Exception as e:
            raise FrameworkConfigurationError(
                f"Error saving configuration: {str(e)}",
                framework="LangChain"
            ) from e


class LangChainConfigBuilder:
    """Programmatic builder for LangChain configurations.
    
    Example:
        >>> builder = LangChainConfigBuilder("analysis_workflow")
        >>> builder.add_node("input", "prompt", "openai", "Analyze: {text}")
        >>> builder.add_node("extract", "chain", "openai")
        >>> builder.add_edge("input", "extract")
        >>> config = builder.build()
    """
    
    def __init__(
        self,
        name: str,
        description: str = "",
        default_provider: str = "openai"
    ):
        """Initialize builder.
        
        Args:
            name: Workflow name
            description: Workflow description
            default_provider: Default provider name
        """
        self.name = name
        self.description = description
        self.default_provider = default_provider
        self._nodes: List[LangChainNodeConfig] = []
        self._edges: List[tuple] = []
        self._metadata: Dict[str, Any] = {}
    
    def add_node(
        self,
        name: str,
        node_type: str,
        provider: Optional[str] = None,
        prompt_template: Optional[str] = None,
        **metadata: Any
    ) -> "LangChainConfigBuilder":
        """Add a node to the workflow.
        
        Args:
            name: Node name
            node_type: Node type
            provider: LLM provider (uses default if not specified)
            prompt_template: Prompt template for prompt nodes
            **metadata: Additional node metadata
            
        Returns:
            Self for chaining
            
        Raises:
            FrameworkConfigurationError: If node already exists
        """
        if any(n.name == name for n in self._nodes):
            raise FrameworkConfigurationError(
                f"Node '{name}' already exists",
                framework="LangChain"
            )
        
        node = LangChainNodeConfig(
            name=name,
            type=node_type,
            provider=provider or self.default_provider,
            prompt_template=prompt_template,
            metadata=metadata
        )
        self._nodes.append(node)
        return self
    
    def add_edge(self, source: str, target: str) -> "LangChainConfigBuilder":
        """Add an edge between nodes.
        
        Args:
            source: Source node name
            target: Target node name
            
        Returns:
            Self for chaining
            
        Raises:
            FrameworkConfigurationError: If nodes not found
        """
        node_names = {n.name for n in self._nodes}
        if source not in node_names:
            raise FrameworkConfigurationError(
                f"Source node '{source}' not found",
                framework="LangChain"
            )
        if target not in node_names:
            raise FrameworkConfigurationError(
                f"Target node '{target}' not found",
                framework="LangChain"
            )
        
        self._edges.append((source, target))
        return self
    
    def set_metadata(self, key: str, value: Any) -> "LangChainConfigBuilder":
        """Set metadata value.
        
        Args:
            key: Metadata key
            value: Metadata value
            
        Returns:
            Self for chaining
        """
        self._metadata[key] = value
        return self
    
    def build(self) -> LangChainWorkflowConfig:
        """Build the configuration.
        
        Returns:
            LangChainWorkflowConfig instance
        """
        return LangChainWorkflowConfig(
            name=self.name,
            description=self.description,
            nodes=self._nodes,
            edges=self._edges,
            default_provider=self.default_provider,
            metadata=self._metadata
        )
