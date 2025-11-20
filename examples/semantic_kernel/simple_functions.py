"""
Simple Semantic Kernel functions example.

Demonstrates basic semantic and native functions using llm_interface providers.
"""

import os
import asyncio
from llm_interface.src.manager import LLMManager
from llm_interface.src.core.config import LLMConfig, ProviderType
from llm_interface.src.framework.semantic_kernel import SemanticKernelAdapter


async def main():
    """Run basic semantic and native function examples."""
    
    # Initialize LLM Manager with provider
    manager = LLMManager()
    
    # Configure OpenAI provider
    openai_config = LLMConfig(
        provider=ProviderType.OPENAI,
        model_name="gpt-4",
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0.7,
        max_tokens=500
    )
    manager.add_provider("gpt4", openai_config)
    
    # Create Semantic Kernel adapter
    adapter = SemanticKernelAdapter(manager)
    
    print("=" * 80)
    print("Semantic Kernel Simple Functions Example")
    print("=" * 80)
    print(f"Using provider: {openai_config.model_name}")
    print()
    
    # Create kernel
    kernel = adapter.create_kernel("gpt4", kernel_id="main")
    print(f"✅ Created kernel: main")
    
    # Example 1: Semantic Function - Translation
    print("\n" + "=" * 80)
    print("Example 1: Semantic Function - Translation")
    print("=" * 80)
    
    translate_func = adapter.create_semantic_function(
        kernel=kernel,
        prompt="Translate the following text to {{$language}}:\n\n{{$text}}",
        function_name="translate",
        plugin_name="text",
        description="Translates text to specified language",
        max_tokens=200,
        temperature=0.5
    )
    
    result = await adapter.invoke_function(
        kernel=kernel,
        function=translate_func,
        arguments={
            "text": "Hello, how are you today?",
            "language": "Spanish"
        }
    )
    
    print(f"Input: 'Hello, how are you today?'")
    print(f"Language: Spanish")
    print(f"Translation: {result}")
    
    # Example 2: Semantic Function - Summarization
    print("\n" + "=" * 80)
    print("Example 2: Semantic Function - Summarization")
    print("=" * 80)
    
    summarize_func = adapter.create_semantic_function(
        kernel=kernel,
        prompt="Summarize the following text in 2-3 sentences:\n\n{{$text}}",
        function_name="summarize",
        plugin_name="text",
        description="Summarizes long text",
        max_tokens=150
    )
    
    long_text = """
    Semantic Kernel is an open-source SDK that lets you easily build agents that 
    can call your existing code. As a highly extensible SDK, you can use Semantic 
    Kernel with models from OpenAI, Azure OpenAI, Hugging Face, and more. By 
    combining your existing C#, Python, and Java code with these models, you can 
    build agents that answer questions and automate processes.
    """
    
    result = await adapter.invoke_function(
        kernel=kernel,
        function=summarize_func,
        arguments={"text": long_text}
    )
    
    print(f"Input: {long_text.strip()[:100]}...")
    print(f"Summary: {result}")
    
    # Example 3: Native Function - String Manipulation
    print("\n" + "=" * 80)
    print("Example 3: Native Function - String Manipulation")
    print("=" * 80)
    
    def to_upper_case(text: str) -> str:
        """Convert text to uppercase."""
        return text.upper()
    
    upper_func = adapter.create_native_function(
        kernel=kernel,
        function=to_upper_case,
        function_name="to_upper",
        plugin_name="text",
        description="Converts text to uppercase"
    )
    
    result = await adapter.invoke_function(
        kernel=kernel,
        function=upper_func,
        arguments={"text": "hello world"}
    )
    
    print(f"Input: 'hello world'")
    print(f"Output: {result}")
    
    # Example 4: Chaining Functions
    print("\n" + "=" * 80)
    print("Example 4: Chaining Functions")
    print("=" * 80)
    
    # First translate, then convert to uppercase
    translation = await adapter.invoke_function(
        kernel=kernel,
        function="text.translate",
        arguments={
            "text": "Good morning!",
            "language": "French"
        }
    )
    
    final_result = await adapter.invoke_function(
        kernel=kernel,
        function="text.to_upper",
        arguments={"text": str(translation)}
    )
    
    print(f"1. Original: 'Good morning!'")
    print(f"2. Translated to French: {translation}")
    print(f"3. Uppercase: {final_result}")
    
    # List all created functions
    print("\n" + "=" * 80)
    print("Created Functions")
    print("=" * 80)
    functions = adapter.list_functions()
    for func_name in functions:
        print(f"  - {func_name}")
    
    print()
    print("=" * 80)
    print("Example Complete!")
    print("=" * 80)


async def example_with_multiple_providers():
    """Example using different providers for different kernels."""
    
    manager = LLMManager()
    
    # Add multiple providers
    openai_config = LLMConfig(
        provider=ProviderType.OPENAI,
        model_name="gpt-4",
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0.7
    )
    manager.add_provider("openai", openai_config)
    
    # Add Azure if available
    if os.getenv("AZURE_OPENAI_KEY"):
        azure_config = LLMConfig(
            provider=ProviderType.AZURE,
            model_name="gpt-4",
            api_key=os.getenv("AZURE_OPENAI_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            azure_deployment="gpt-4",
            temperature=0.7
        )
        manager.add_provider("azure", azure_config)
    
    adapter = SemanticKernelAdapter(manager)
    
    print("=" * 80)
    print("Multi-Provider Semantic Kernel Example")
    print("=" * 80)
    
    # Create kernel with OpenAI
    kernel1 = adapter.create_kernel("openai", kernel_id="openai_kernel")
    print(f"✅ Created kernel: openai_kernel")
    
    # Create function on first kernel
    func1 = adapter.create_semantic_function(
        kernel=kernel1,
        prompt="Write a haiku about {{$topic}}",
        function_name="haiku",
        description="Writes a haiku"
    )
    
    result1 = await adapter.invoke_function(
        kernel=kernel1,
        function=func1,
        arguments={"topic": "artificial intelligence"}
    )
    
    print(f"\nOpenAI Haiku about AI:")
    print(result1)
    
    # If Azure is available, create second kernel
    if "azure" in manager.list_providers():
        kernel2 = adapter.create_kernel("azure", kernel_id="azure_kernel")
        print(f"\n✅ Created kernel: azure_kernel")
        
        func2 = adapter.create_semantic_function(
            kernel=kernel2,
            prompt="Write a limerick about {{$topic}}",
            function_name="limerick",
            description="Writes a limerick"
        )
        
        result2 = await adapter.invoke_function(
            kernel=kernel2,
            function=func2,
            arguments={"topic": "machine learning"}
        )
        
        print(f"\nAzure Limerick about ML:")
        print(result2)
    
    # List all kernels
    print(f"\nCreated Kernels: {adapter.list_kernels()}")
    
    print()
    print("=" * 80)
    print("Multi-Provider Example Complete!")
    print("=" * 80)


if __name__ == "__main__":
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Set it with: export OPENAI_API_KEY='your-key-here'")
        exit(1)
    
    # Run main example
    asyncio.run(main())
    
    print("\n\n")
    
    # Run multi-provider example
    asyncio.run(example_with_multiple_providers())
