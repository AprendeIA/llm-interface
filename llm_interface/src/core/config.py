from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict

class ProviderType(Enum):
    """Supported LLM provider types.
    
    Only providers with full implementations are included.
    To add a new provider, implement the provider class in llm_interface/src/providers/
    and register it in the factory.
    """
    OPENAI = "openai"
    AZURE = "azure"
    OLLAMA = "ollama"
    ANTHROPIC = "anthropic"
    # GEMINI = "gemini"  # Not yet implemented

class LLMConfig(BaseModel):
    """Configuration for LLM providers with comprehensive validation.
    
    This Pydantic model provides runtime validation for all configuration
    parameters, ensuring type safety and business rules are enforced.
    """
    
    provider: ProviderType = Field(
        ...,
        description="The LLM provider type (OpenAI, Azure, Anthropic, etc.)"
    )
    model_name: str = Field(
        ...,
        min_length=1,
        description="Name of the model to use (e.g., 'gpt-4', 'claude-3-sonnet')"
    )
    api_key: Optional[str] = Field(
        default=None,
        description="API key for authentication (or use environment variables)"
    )
    base_url: Optional[str] = Field(
        default=None,
        description="Custom base URL for API endpoints"
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature (0.0 = deterministic, 2.0 = very random)"
    )
    max_tokens: int = Field(
        default=1000,
        gt=0,
        description="Maximum number of tokens to generate"
    )
    
    # Azure-specific fields
    azure_endpoint: Optional[str] = Field(
        default=None,
        description="Azure OpenAI endpoint URL (required for Azure provider)"
    )
    azure_deployment: Optional[str] = Field(
        default=None,
        description="Azure OpenAI deployment name (required for Azure provider)"
    )
    azure_api_version: Optional[str] = Field(
        default=None,
        description="Azure OpenAI API version (e.g., '2023-12-01-preview')"
    )
    
    @field_validator('model_name')
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        """Ensure model name is not empty or whitespace only."""
        if not v or not v.strip():
            raise ValueError("model_name cannot be empty or whitespace")
        return v.strip()
    
    @field_validator('base_url')
    @classmethod
    def validate_base_url(cls, v: Optional[str]) -> Optional[str]:
        """Validate base URL format if provided."""
        if v:
            v = v.strip()
            if not (v.startswith('http://') or v.startswith('https://')):
                raise ValueError(
                    f"base_url must start with 'http://' or 'https://'. Got: {v}"
                )
            # Remove trailing slash for consistency
            return v.rstrip('/')
        return v
    
    @field_validator('azure_endpoint')
    @classmethod
    def validate_azure_endpoint(cls, v: Optional[str]) -> Optional[str]:
        """Validate Azure endpoint URL format if provided."""
        if v:
            v = v.strip()
            if not (v.startswith('http://') or v.startswith('https://')):
                raise ValueError(
                    f"azure_endpoint must start with 'http://' or 'https://'. Got: {v}"
                )
            return v.rstrip('/')
        return v
    
    @model_validator(mode='after')
    def validate_azure_config(self) -> 'LLMConfig':
        """Validate Azure-specific configuration requirements."""
        if self.provider == ProviderType.AZURE:
            if not self.azure_endpoint:
                raise ValueError(
                    "azure_endpoint is required when provider is 'azure'"
                )
            if not self.azure_deployment:
                raise ValueError(
                    "azure_deployment is required when provider is 'azure'"
                )
        return self
    
    @model_validator(mode='after')
    def validate_api_key_for_providers(self) -> 'LLMConfig':
        """Validate that API key is provided for cloud providers (optional for Ollama)."""
        # Note: We don't enforce API key here because it might come from environment
        # The actual validation happens in the provider classes
        return self
    
    model_config = ConfigDict(
        use_enum_values=False,  # Keep enum types, don't convert to values
        validate_assignment=True,  # Validate on attribute assignment
        arbitrary_types_allowed=False,
        json_schema_extra={
            "examples": [
                {
                    "provider": "openai",
                    "model_name": "gpt-4",
                    "api_key": "sk-...",
                    "temperature": 0.7,
                    "max_tokens": 1000
                },
                {
                    "provider": "azure",
                    "model_name": "gpt-4",
                    "azure_endpoint": "https://your-resource.openai.azure.com",
                    "azure_deployment": "gpt-4-deployment",
                    "api_key": "your-azure-key",
                    "temperature": 0.7
                }
            ]
        }
    )
