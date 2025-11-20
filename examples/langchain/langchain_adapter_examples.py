"""LangChain adapter usage examples.

Demonstrates how to use the LangChain adapter with the unified LLM Interface.
"""

from llm_interface import LLMManager, LLMConfig, ProviderType
from llm_interface.src.framework.langchain import (
    LangChainAdapter,
    WorkflowFactory,
    WorkflowType,
)


def example_basic_adapter():
    """Example 1: Basic adapter setup and usage."""
    print("=" * 60)
    print("Example 1: Basic LangChain Adapter Setup")
    print("=" * 60)
    
    # Initialize manager
    manager = LLMManager()
    
    # Configure provider (in real usage, set API key via environment)
    config = LLMConfig(
        provider=ProviderType.OPENAI,
        model_name="gpt-4",
        api_key="your-api-key"  # Use env var in production
    )
    
    manager.add_provider("primary", config)
    
    # Create and register adapter
    langchain_adapter = LangChainAdapter(manager)
    manager.register_framework(langchain_adapter)
    
    print(f"Registered framework: {langchain_adapter.framework_name}")
    print(f"Available providers: {langchain_adapter.list_providers()}")
    
    # Validate configuration
    is_valid, message = langchain_adapter.validate_config()
    print(f"Configuration valid: {is_valid}")
    if not is_valid:
        print(f"Error: {message}")


def example_create_chain():
    """Example 2: Creating LangChain chains."""
    print("\n" + "=" * 60)
    print("Example 2: Creating LangChain Chains")
    print("=" * 60)
    
    manager = LLMManager()
    config = LLMConfig(
        provider=ProviderType.OPENAI,
        model_name="gpt-3.5-turbo",
        api_key="your-api-key"
    )
    manager.add_provider("default", config)
    
    adapter = LangChainAdapter(manager)
    
    # Create a simple summarization chain
    try:
        chain = adapter.create_chain(
            "default",
            "Summarize the following text in one paragraph:\n\n{text}"
        )
        print(f"✓ Created summarization chain")
        
        # Example usage (would need actual API key):
        # result = chain.invoke({"text": "Your long text here..."})
        # print(f"Summary: {result['text']}")
        
    except Exception as e:
        print(f"✗ Error creating chain: {e}")


def example_multi_provider():
    """Example 3: Using multiple providers."""
    print("\n" + "=" * 60)
    print("Example 3: Multi-Provider Setup")
    print("=" * 60)
    
    manager = LLMManager()
    
    # Add multiple providers
    openai_config = LLMConfig(
        provider=ProviderType.OPENAI,
        model_name="gpt-4",
        api_key="sk-..."
    )
    manager.add_provider("openai", openai_config)
    
    anthropic_config = LLMConfig(
        provider=ProviderType.ANTHROPIC,
        model_name="claude-3-sonnet",
        api_key="sk-ant-..."
    )
    manager.add_provider("anthropic", anthropic_config)
    
    adapter = LangChainAdapter(manager)
    
    print("Available providers:")
    for provider in adapter.list_providers():
        info = adapter.get_provider_info(provider)
        print(f"  - {provider}: {info.get('provider_type', 'unknown')}")
    
    print("\nAvailable models:")
    models = adapter.get_available_models()
    for provider, model_list in models.items():
        print(f"  {provider}: {model_list}")


def example_workflow_chain():
    """Example 4: Creating a chain workflow."""
    print("\n" + "=" * 60)
    print("Example 4: Chain Workflow")
    print("=" * 60)
    
    # Create workflow builder
    workflow = WorkflowFactory.create(
        WorkflowType.CHAIN,
        "text_processing",
        "openai"
    )
    
    # Define processing functions
    def extract_entities(text: str) -> str:
        """Extract entities from text."""
        return f"Entities from: {text}"
    
    def analyze_sentiment(entities: str) -> str:
        """Analyze sentiment."""
        return f"Sentiment analysis of: {entities}"
    
    def generate_summary(analysis: str) -> str:
        """Generate summary."""
        return f"Summary of: {analysis}"
    
    # Build workflow
    (workflow
     .add_node("extract", extract_entities)
     .add_node("sentiment", analyze_sentiment)
     .add_node("summary", generate_summary)
     .add_edge("extract", "sentiment")
     .add_edge("sentiment", "summary"))
    
    compiled = workflow.build()
    print(f"✓ Created workflow with {len(compiled)} nodes")
    print(f"  Nodes: {list(compiled.keys())}")


def example_workflow_conditional():
    """Example 5: Creating a conditional workflow."""
    print("\n" + "=" * 60)
    print("Example 5: Conditional Workflow")
    print("=" * 60)
    
    # Create conditional workflow
    workflow = WorkflowFactory.create(
        WorkflowType.CONDITIONAL,
        "routing_workflow",
        "openai"
    )
    
    def classify_query(query: str) -> str:
        """Classify query type."""
        return "technical" if "?" in query else "general"
    
    def handle_technical(query: str) -> str:
        """Handle technical queries."""
        return f"Technical response: {query}"
    
    def handle_general(query: str) -> str:
        """Handle general queries."""
        return f"General response: {query}"
    
    def is_technical(state: dict) -> bool:
        """Check if query is technical."""
        return state.get("type") == "technical"
    
    # Build workflow
    (workflow
     .add_node("classify", classify_query)
     .add_node("technical", handle_technical)
     .add_node("general", handle_general)
     .add_edge("classify", "technical", is_technical)
     .add_edge("classify", "general"))
    
    compiled = workflow.build()
    print(f"✓ Created conditional workflow")
    print(f"  Nodes: {list(compiled['nodes'].keys())}")
    print(f"  Routes: {len(compiled['edges'])}")


def example_workflow_parallel():
    """Example 6: Creating a parallel workflow."""
    print("\n" + "=" * 60)
    print("Example 6: Parallel Workflow")
    print("=" * 60)
    
    # Create parallel workflow
    workflow = WorkflowFactory.create(
        WorkflowType.PARALLEL,
        "analysis_workflow",
        "openai"
    )
    
    def sentiment_analysis(text: str) -> str:
        """Analyze sentiment."""
        return "positive" if text.count("good") > 0 else "neutral"
    
    def entity_extraction(text: str) -> str:
        """Extract entities."""
        return f"Entities: {text[:20]}..."
    
    def keyword_extraction(text: str) -> str:
        """Extract keywords."""
        return f"Keywords: {text.split()[:3]}"
    
    # Build workflow - these will execute in parallel
    (workflow
     .add_node("sentiment", sentiment_analysis)
     .add_node("entities", entity_extraction)
     .add_node("keywords", keyword_extraction))
    
    compiled = workflow.build()
    print(f"✓ Created parallel workflow")
    print(f"  Nodes: {list(compiled['nodes'].keys())}")
    print(f"  Parallel groups: {compiled['parallel_groups']}")


def example_adapter_caching():
    """Example 7: Using adapter caching."""
    print("\n" + "=" * 60)
    print("Example 7: Model and Chain Caching")
    print("=" * 60)
    
    manager = LLMManager()
    config = LLMConfig(
        provider=ProviderType.OPENAI,
        model_name="gpt-3.5-turbo",
        api_key="your-api-key"
    )
    manager.add_provider("default", config)
    
    adapter = LangChainAdapter(manager)
    
    try:
        # First call creates and caches
        print("Creating model (1st time)...")
        model1 = adapter.get_model("default", use_cache=True)
        print("✓ Model created and cached")
        
        # Second call retrieves from cache
        print("Retrieving model (2nd time)...")
        model2 = adapter.get_model("default", use_cache=True)
        print("✓ Model retrieved from cache")
        
        # Clear cache
        adapter.clear_cache()
        print("✓ Cache cleared")
        
    except Exception as e:
        print(f"Note: {e}")


def example_embeddings():
    """Example 8: Creating embeddings models."""
    print("\n" + "=" * 60)
    print("Example 8: Embeddings Models")
    print("=" * 60)
    
    manager = LLMManager()
    config = LLMConfig(
        provider=ProviderType.OPENAI,
        model_name="gpt-3.5-turbo",
        api_key="your-api-key"
    )
    manager.add_provider("default", config)
    
    adapter = LangChainAdapter(manager)
    
    try:
        # Create embeddings model
        embeddings = adapter.create_embeddings("default")
        print(f"✓ Created embeddings model: {type(embeddings).__name__}")
        
        # Would embed text in real usage:
        # embedded = embeddings.embed_query("Sample text")
        # print(f"Embedding dimension: {len(embedded)}")
        
    except Exception as e:
        print(f"Note: {e}")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("LangChain Adapter Examples")
    print("=" * 60 + "\n")
    
    # Note: These examples demonstrate structure, actual execution
    # requires valid API keys and installed dependencies
    
    example_basic_adapter()
    example_create_chain()
    example_multi_provider()
    example_workflow_chain()
    example_workflow_conditional()
    example_workflow_parallel()
    example_adapter_caching()
    example_embeddings()
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
